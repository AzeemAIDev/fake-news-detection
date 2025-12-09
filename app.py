import pickle
import nltk
import re
import sys
import os
import uvicorn

from pydantic import BaseModel
from nltk.corpus import stopwords
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
from fastapi.templating import Jinja2Templates 
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
from tensorflow.keras.preprocessing.sequence import pad_sequences 

# --- 1. FastAPI Application Setup 
# Create a FastAPI instance with metadata for documentation
app = FastAPI(
    title="Fake News Detection API",
    description="API for classifying news as Real or Fake using an LSTM model.",
    version="1.0.0"
)

# --- 2. Template Setup 
# Set up Jinja2 templates for serving HTML pages.
# Ensure a 'templates' folder exists alongside this app.py file.
templates = Jinja2Templates(directory="templates")

# --- 3. Model and Tokenizer Loading 
# Define paths relative to this script file
model_path = os.path.join(os.path.dirname(__file__), "model", "news_detection_model.keras")
tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer", "tokenizer.pkl")

model = None
tokenizer = None

# Load the pre-trained LSTM model
try:
    model = load_model(model_path, compile=False)
    print("Model Loaded Successfully.")
except Exception as e:
    print(f"FATAL: Error in model loading from '{model_path}': {e}")
    sys.exit(1)

# Load the tokenizer used during model training
try:
    with open(tokenizer_path , "rb") as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer Loaded Successfully from '{tokenizer_path}'")
except Exception as e:
    print(f"FATAL: Error in tokenizer loading from '{tokenizer_path}': {e}")
    sys.exit(1)

# --- 4. NLP Preprocessing Setup ---
# Load stopwords from NLTK, downloading if necessary
try:
    nltk.data.find('corpora/stopwords')
    stop_words = set(stopwords.words("english"))
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
except Exception as e:
    print(f"Error initializing stopwords: {e}")
    stop_words = set() 

# Function to clean input text: remove non-alphabetic chars, lowercase, remove stopwords
def clean_text(text):
    """ Cleans text by removing punctuation, lowercasing, and removing stopwords. """
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# --- 5. Request/Response Model 
# Define Pydantic model for input validation
class news_input(BaseModel):
    title : str
    text : str

# --- 6. API Endpoints ---

# A. Root endpoint: Serves HTML UI for user interaction
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """ Renders the main index.html file for the web UI. """
    return templates.TemplateResponse("index.html", {"request": request})

# B. Prediction endpoint: Handles POST requests for news classification
@app.post("/predict")
def predict_news(data : news_input):
    """ Predicts whether the provided news article is Real or Fake. """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or Tokenizer not loaded. Check server logs.")

    try:
        # Combine title and text, then clean the input
        full_text = data.title + " " + data.text
        clean = clean_text(full_text)

        # Convert text to sequences using the tokenizer
        seq = tokenizer.texts_to_sequences([clean])
        MAX_LEN = 200  
        pad_seq = pad_sequences(seq , maxlen=MAX_LEN)

        # Perform prediction using the LSTM model
        prediction_result = model.predict(pad_seq)
        pred_probs = float(prediction_result[0][0])  # Probability output for binary classification

        # Apply threshold to determine class
        pred = "Real News" if pred_probs >= 0.5 else "Fake News"

        # Return prediction and probability
        return {"Prediction" : pred , "Probability" : pred_probs}

    except Exception as e:
        print(f"Prediction processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed due to internal error: {e}")

# --- 7. Run Uvicorn Server 
if __name__ == "__main__":
    # Start FastAPI server locally on the port 8000
    uvicorn.run(app , host="127.0.0.1" , port=8000)
