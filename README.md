# PolyAPI AI Multi-App

A modular AI web application and RESTful microservice built with **Python** and **Flask** that combines computer vision, emotion detection, and automated web scraping into a single platform.

## Key Features
* **AI Image Captioning:** Uses Hugging Face's pre-trained BLIP (`BlipForConditionalGeneration` & `AutoProcessor`) model to generate natural language descriptions from uploaded images.
* **URL-Based Content Extraction:** Automatically scrapes and processes images from external web pages using `BeautifulSoup` and `Pillow`.
* **Emotion & Sentiment Detection:** Analyzes input content to classify emotional tone through clean API endpoints.

## Tech Stack
* **Backend & API:** Python, Flask, Requests
* **AI & Deep Learning:** PyTorch, Hugging Face `transformers` (BLIP)
* **Web Scraping & Image Processing:** BeautifulSoup4, Pillow
* **Configuration:** `python-dotenv`, Git

## Local Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kathambari-Kumar/PolyAPI_AI
   cd PolyAPI_AI
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
**Acknowledgments & Licensing**

BLIP (Bootstrapping Language - Image Pretraining) Provided via Hugging Face Transformers. 
Licensed under the MIT License - free to use, modify, and distribute with attribution.

AutoProcessor & Transformers Library Part of the Hugging Face ecosystem. 
Licensed under Apache 2.0 - permissive for both research and commercial use.

PyTorch (torch) Deep learning framework powering BLIP. 
Licensed under BSD-style license.

Other dependencies (Flask, BeautifulSoup, Pillow, Requests, python-dotenv) All are open-source libraries under permissive licenses (BSD, MIT, Apache 2.0).
