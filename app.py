from flask import Flask, render_template, request, url_for
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
from bs4 import BeautifulSoup
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Retrieve API keys
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")

# Load BLIP model for captioning
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# --- Menu ---
@app.route("/")
def index():
    return render_template("index.html")

# --- Emotion Detector ---
@app.route("/emotion_selection")
def emotion_selection():
    """Render the main page with user selection and emotion options."""
    # Fetch random users from API
    resp = requests.get("https://randomuser.me/api/?results=10")
    users = [u["name"]["first"] + " " + u["name"]["last"] for u in resp.json()["results"]]
    return render_template("emotion.html", users=users)

@app.route("/recommendations", methods=["GET", "POST"])
def recommendations():
    person = request.form.get("person")
    emotion = request.form.get("emotion")

    resp = requests.get("https://randomuser.me/api/?results=10")
    users = [u["name"]["first"] + " " + u["name"]["last"] for u in resp.json()["results"]]
    recommendations = {}

    # Recipe suggestion
    emotion_to_tag = {
        "sad": "comfort food",
        "happy": "dessert",
        "fear": "soup",
        "anger": "spicy",
        "depression": "chocolate",
        "nervous": "snack"
    }
    tag = emotion_to_tag.get(emotion, "main course")  # fallback if emotion not mapped
    url = f"https://api.spoonacular.com/recipes/random?tags={tag}&apiKey={SPOONACULAR_API_KEY}"
    recipe_resp = requests.get(url)
    data = recipe_resp.json()

    if "recipes" in data and data["recipes"]:
        recommendations["recipe"] = data["recipes"][0]["title"]
        recommendations["recipe_url"] = data["recipes"][0]["sourceUrl"]

    # Book suggestion
    url = f"https://www.googleapis.com/books/v1/volumes?q={emotion}&key={GOOGLE_BOOKS_API_KEY}"
    book_resp = requests.get(url)
    recommendations["book"] = book_resp.json()["items"][0]["volumeInfo"]["title"]

    # Song suggestion
    song_resp = requests.get(f"https://itunes.apple.com/search?term={emotion}&entity=song&limit=1")
    recommendations["song"] = song_resp.json()["results"][0]["trackName"]
    recommendations["song_artist"] = song_resp.json()["results"][0]["artistName"]
    return render_template("emotion.html", users=users, recommendations=recommendations, person=person, emotion=emotion)

# --- Local Image Captioning ---
@app.route("/image_upload")
def image_upload():
    """Render the main page with image selection."""
    return render_template("caption.html")

@app.route("/caption_generation", methods=["GET", "POST"])
def caption_generation():
    caption = None
    image_url = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join("static/uploads", filename)
            file.save(filepath)
            image = Image.open(file).convert("RGB")
            inputs = processor(images=image, text="", return_tensors="pt")
            outputs = model.generate(**inputs, max_length=50)
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            # Generate URL to display image
            image_url = url_for("static", filename=f"uploads/{filename}")
    return render_template("caption.html",image_url=image_url, caption=caption)

# --- URL Image Captioning ---
@app.route("/url_selection")
def url_selection():
    """Render the main page with URL address selection."""
    return render_template("automatic_url.html")

@app.route("/url_caption", methods=["GET", "POST"])
def url_caption():
    captions =[]
    if request.method == "POST":
        url = request.form["url_addr"]
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        # list of <img> tags
        img_elements = soup.find_all("img")
        print(f"Found {len(img_elements)} <img> tags")

        for idx, img_element in enumerate(img_elements, start=1):
            # Try different attributes
            img_url = img_element.get("src") or img_element.get("data-src")
            if not img_url and img_element.has_attr("srcset"):
                img_url = img_element["srcset"].split()[0]

            # if no URL is found, skip this image
            if not img_url:
                continue

            # Skip SVGs directly
            # Skips SVGs (vector graphics) because Pillow can’t open them.
            if img_url.endswith(".svg") or ".svg" in img_url:
                continue

            # Fix relative URLs
            # relative path into full URL
            if img_url.startswith("//"):
                img_url = "https:" + img_url
            elif img_url.startswith("/"):
                img_url = url.rstrip("/") + img_url
            elif not img_url.startswith("http"):
                continue

            try:
                # Fetches the image from the web.
                r = requests.get(img_url, timeout=10, headers=headers)
                raw_image = Image.open(BytesIO(r.content))

                # Skip very small images
                # Ignores tiny icons, logos, or thumbnails.
                if raw_image.size[0] * raw_image.size[1] < 200:
                    continue

                raw_image = raw_image.convert("RGB")

                # Process the image with a text prompt
                inputs = processor(images=raw_image, text="", return_tensors="pt")
                # Generates a caption (up to 50 tokens).
                out = model.generate(**inputs, max_new_tokens=50)

                # Decodes the caption into readable text.
                caption = processor.decode(out[0], skip_special_tokens=True)

                # Collect dictionary values
                captions.append({"url": img_url, "caption": caption})

                if len(captions) >= 5:  # stop after 5
                    break

            except Exception as e:
                print(f"[{idx}] Error: {e}")
                continue

    return render_template("automatic_url.html", captions=captions)

if __name__ == "__main__":
    app.run(debug=True)
