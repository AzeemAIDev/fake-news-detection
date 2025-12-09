from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
import pickle
import nltk
import re
import sys
import os 

# --- 1. FastAPI Setup ---
# All indentation here is now standard Python space characters
app = FastAPI(title="Fake News Detection API",
              description="API for classifying news as Real or Fake using an LSTM model.",
              version="1.0.0.1")

# Allow all origins for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- 2. Model and Tokenizer Loading ---
model_path = "fake_news_detection_model.keras"
tokenizer_path = "tokenizer.pkl"

model = None
tokenizer = None

# Load Model
try:
    model = load_model(model_path, compile=False) 
    print(f"Model Loaded Successfully: {model}")
except Exception as e:
    print(f"FATAL: Error in model loading from '{model_path}': {e}")
    sys.exit(1)

# Load Tokenizer
try:
    with open(tokenizer_path , "rb") as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer Loaded Successfully from '{tokenizer_path}'")
except Exception as e:
    print(f"FATAL: Error in tokenizer loading from '{tokenizer_path}': {e}")
    sys.exit(1)


# --- 3. NLP Preprocessing Setup ---
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

def clean_text(text):
    """ Cleans text: removes punctuation, lowercases, and removes stop words. """
    text = re.sub(r"[^a-zA-Z\s]" , " " , text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# --- 4. Request/Response Models ---
class news_input(BaseModel):
    title : str
    text : str


# --- 5. API Endpoints ---
@app.post("/predict")
def predict_news(data : news_input):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or Tokenizer not loaded. Check server logs.")

    try:
        full_text = data.title + " " + data.text
        clean = clean_text(full_text)

        seq = tokenizer.texts_to_sequences([clean])
        MAX_LEN = 200 
        pad_seq = pad_sequences(seq , maxlen=MAX_LEN)

        prediction_result = model.predict(pad_seq)
        
        pred_probs = float(prediction_result[0][0]) 
        
        pred = "Real News" if pred_probs >= 0.5 else "Fake News"

        return {"Prediction" : pred , "Probability" : pred_probs, "is_fake": pred_probs < 0.5}

    except Exception as e:
        print(f"Prediction processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed due to internal error: {e}")


# --- 6. HTML Content (Client UI) ---

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake News Detector - NLP Project</title>
    <!-- Tailwind CSS for beautiful, responsive design -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
        
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f0f4f8; 
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .card {
            background-color: white;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
            max-width: 900px;
            width: 100%;
        }

        .result-container {
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275); 
            transform: translateY(20px);
            opacity: 0;
            height: 0;
            overflow: hidden;
            padding: 0 2rem; 
        }
        .result-container.show {
            transform: translateY(0);
            opacity: 1;
            height: auto;
            padding: 1.5rem 2rem;
            margin-top: 1.5rem;
        }
    </style>
</head>
<body class="bg-gray-100">

    <div class="card p-8 md:p-12 rounded-3xl">
        <h1 class="text-4xl md:text-5xl font-extrabold text-gray-900 mb-2">
            📰 Fake News Analyzer
        </h1>
        <p class="text-gray-500 mb-10 text-lg">Use the power of NLP to check if an article is Real or Fake.</p>

        <form id="predictionForm" onsubmit="handlePrediction(event)">
            
            <div class="mb-6">
                <label for="title" class="block text-base font-semibold text-gray-700 mb-2">Article Title</label>
                <input type="text" id="title" name="title" required placeholder="Enter the headline"
                       class="w-full p-4 border-2 border-gray-300 rounded-xl focus:ring-4 focus:ring-blue-200 focus:border-blue-500 transition duration-200 shadow-sm">
            </div>

            <div class="mb-8">
                <label for="text" class="block text-base font-semibold text-gray-700 mb-2">Article Body Text</label>
                <textarea id="text" name="text" rows="10" required placeholder="Paste the complete text here (longer input improves accuracy)..."
                          class="w-full p-4 border-2 border-gray-300 rounded-xl resize-y focus:outline-none focus:ring-4 focus:ring-blue-200 focus:border-blue-500 transition duration-200 shadow-sm"></textarea>
            </div>

            <button type="submit" id="submitButton"
                    class="w-full bg-blue-600 text-white font-extrabold text-lg py-4 rounded-xl hover:bg-blue-700 transition duration-300 shadow-lg shadow-blue-500/50 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed">
                <span id="buttonText">Classify Article</span>
                <svg id="loadingSpinner" class="animate-spin -ml-1 mr-3 h-6 w-6 text-white hidden" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
            </button>
        </form>

        <!-- Result Display Container -->
        <div id="resultContainer" class="result-container text-white rounded-xl">
            <h2 id="resultHeading" class="text-3xl font-extrabold mb-1"></h2>
            <p id="resultProbability" class="text-md opacity-90"></p>
        </div>

        <!-- Custom Alert/Message Box -->
        <div id="messageBox" class="hidden fixed inset-0 bg-gray-900 bg-opacity-70 flex items-center justify-center p-4 z-50">
            <div class="bg-white p-8 rounded-xl shadow-2xl max-w-lg w-full transform transition duration-300 scale-100">
                <h3 class="text-2xl font-bold text-red-600 mb-4">Connection Error!</h3>
                <p id="messageText" class="text-gray-700 mb-6 text-lg"></p>
                <button onclick="hideMessage()" class="w-full bg-red-600 text-white font-semibold py-3 rounded-lg hover:bg-red-700 transition duration-300 shadow-md">Close</button>
            </div>
        </div>

    </div>

    <script>
        // API URL is relative, compatible with Docker/Uvicorn on 0.0.0.0
        const API_URL = '/predict'; 
        
        const resultContainer = document.getElementById('resultContainer');
        const resultHeading = document.getElementById('resultHeading');
        const resultProbability = document.getElementById('resultProbability');
        const submitButton = document.getElementById('submitButton');
        const buttonText = document.getElementById('buttonText');
        const loadingSpinner = document.getElementById('loadingSpinner');
        const messageBox = document.getElementById('messageBox');
        const messageText = document.getElementById('messageText');

        function showLoading(isLoading) {
            submitButton.disabled = isLoading;
            loadingSpinner.classList.toggle('hidden', !isLoading);
            buttonText.textContent = isLoading ? 'Analyzing...' : 'Classify Article';
        }

        function showMessage(text) {
            messageText.innerHTML = text; 
            messageBox.classList.remove('hidden');
        }

        function hideMessage() {
            messageBox.classList.add('hidden');
        }

        function displayResult(prediction, probability) {
            let bgColor;
            let label;
            
            const probPercent = (probability * 100).toFixed(2);
            
            if (prediction === "Real News") {
                bgColor = 'bg-green-600';
                label = 'REAL NEWS';
            } else {
                bgColor = 'bg-red-600';
                label = 'FAKE NEWS';
            }

            resultContainer.classList.remove('show');
            
            setTimeout(() => {
                resultContainer.className = 'result-container ' + bgColor; 
                resultHeading.textContent = `Verdict: ${label}`;
                resultProbability.textContent = `Confidence: ${probPercent}% (Prediction Score: ${probability.toFixed(4)})`;

                setTimeout(() => {
                    resultContainer.classList.add('show');
                }, 50);
            }, 500); 
        }

        async function handlePrediction(event) {
            event.preventDefault();
            showLoading(true);

            const title = document.getElementById('title').value;
            const text = document.getElementById('text').value;

            const data = { title, text };
            
            if (title.trim() === "" || text.trim() === "") {
                showMessage("Please enter both the Title and the Article Text to analyze.");
                showLoading(false);
                return;
            }

            try {
                const response = await fetch(API_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data),
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || `Server Error (${response.status})`);
                }

                const result = await response.json();
                
                displayResult(result.Prediction, result.Probability);

            } catch (error) {
                console.error("Fetch Error:", error);
                // Solution for Docker/local:
                showMessage(`Could not connect to the API or an error occurred: 
                <br><br><strong>Error:</strong> ${error.message}.
                <br><br><strong>Solution:</strong> Please ensure the Python server is running (host='0.0.0.0') and the port is correctly open (e.g., Docker port mapping <strong>8000:8000</strong>).`);
            } finally {
                showLoading(false);
            }
        }
    </script>

</body>
</html>
"""

# --- 7. Serve HTML Interface on Root Path ---

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """ Serves the embedded HTML/JS frontend interface. """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    # HOST FIX: Using "0.0.0.0" for Docker compatibility
    uvicorn.run(app , host="0.0.0.0" , port=8000)