<h1> Fake News Detection</h1>
<p><b>A Machine Learning & NLP Project to Classify News as Real or Fake</b></p>

<hr>

<h2> Overview</h2>
<p>Fake news is a major problem in the digital age — harmful misinformation spreads quickly on social media and news platforms.<br>
This project uses <b>Natural Language Processing (NLP)</b> and <b>Machine Learning</b> to detect whether a news article is <i>fake</i> or <i>real</i> based on its text content.</p>

<hr>

<h2> Features</h2>
<ul>
<li> Text preprocessing using tokenization, cleaning, and vectorization</li>
<li> Machine Learning/DL model trained to classify news articles</li>
<li> Predict fake or real news with simple input text</li>
<li> Easy integration with web UI or API</li>
</ul>

<hr>

<h2> Technologies Used</h2>
<p>
<img src="https://img.shields.io/badge/Python-3.11-blue">
<img src="https://img.shields.io/badge/NLTK-3.7-yellow">
<img src="https://img.shields.io/badge/scikit--learn-1.2-green">
<img src="https://img.shields.io/badge/TensorFlow-2.13-orange">
</p>
<ul>
<li>Python</li>
<li>Natural Language Processing (NLP)</li>
<li>TF‑IDF / Vectorization</li>
<li>Deep Learning Models</li>
</ul>

<hr>

<h2>📂 Project Structure</h2>
<pre>
fake-news-detection/
│
│
├── codes/
│   └── train_model.ipynb
│
├── model/
│   └── news_detection_model.keras
│
├── templates/
│   └── index.html
│
├── tokenizer/
│   └── tokenizer.pkl
│
├── README.md 
├── app.py
└── requirements.txt
</pre>

<hr>

<h2>🧠 Model Details</h2>
<ul>
<li><b>Text Preprocessing:</b> Cleaning text (lowercase, remove symbols), tokenization, stopword removal, TF‑IDF vectorization</li>
<li><b>Model Training:</b> Train/test split, Sequential model </li>
<li><b>Evaluation:</b> Accuracy, precision, recall, F1-score, and saved model for future predictions</li>
</ul>

<hr>

<h2>📥 Installation & Setup</h2>

<h3>1️⃣ Clone the repository</h3>
<pre>
git clone https://github.com/AzeemAIDev/fake-news-detection.git
</pre>

<h3>2️⃣ Navigate into project directory</h3>
<pre>
cd fake-news-detection
</pre>

<h3>3️⃣ Install dependencies</h3>
<pre>
pip install -r requirements.txt
</pre>

<hr>

<h2>▶️ How to Run</h2>
<p>If the project has a Jupyter notebook:</p>
<ul>
<li>Open the notebook in the <code>notebooks/</code> folder</li>
<li>Run all cells for preprocessing, training and evaluation</li>
</ul>
<p>If the project provides a script or API:</p>
<pre>
uvicorn app:app --reload --port 8000
</pre>
<p>Then navigate to your browser on the provided link and enter news text to get <b>fake</b> or <b>real</b> prediction.</p>

<hr>

<h2>📊 Dataset</h2>
<ul>
<li><b>Dataset:</b> Labeled news articles in CSV format (real vs fake)</li>
<li>Columns: <code>title</code>, <code>text</code>, <code>label</code></li>
<li><a href="https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets" target="_blank">Kaggle Dataset Link</a></li>
</ul>

<hr>

<h2>⭐ Author</h2>
<p><b>Muhammad Azeem</b><br>
Machine Learning & AI Learner<br>
GitHub: <a href="https://github.com/AzeemAIDev" target="_blank">AzeemAIDev</a></p>
