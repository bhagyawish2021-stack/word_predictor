# 🧠 Next Word Predictor

A smart **Next Word Prediction Web App** built using **Python, Flask, and NLP concepts**.
This project predicts the next possible words based on user input using an **N-gram Language Model**.

Perfect for learning:

* Natural Language Processing (NLP)
* Flask API development
* Machine Learning basics
* Text prediction systems

---

## 🚀 Features

* 🔮 Predicts next possible words
* ⚡ Real-time suggestions using Flask API
* 🧠 N-gram based language model
* 📊 Confidence score for predictions
* 💾 Model save & load functionality
* 🌐 Web-based interface
* 🛠 REST API endpoints

---

## 🛠 Technologies Used

* Python
* Flask
* HTML/CSS/JavaScript
* NLP (Natural Language Processing)
* Pickle (Model Storage)

---

## 📂 Project Structure

```bash
project/
│
├── app.py
├── predictor/
│   ├── __init__.py
│   ├── predictor.py
│   └── model.pkl
│
├── dataset/
│   └── corpus.txt
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
└── requirements.txt
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/next-word-predictor.git
cd next-word-predictor
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate virtual environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python app.py
```

The app will run at:

```bash
http://127.0.0.1:5000
```

---

## 🧠 How It Works

The project uses an **N-gram Language Model**.

Example:

Input:

```text
I am going to
```

Predictions:

```text
school
sleep
work
eat
```

The model analyzes previous word sequences from the training corpus and predicts the most probable next word.

---

## 📡 API Endpoints

### Predict Next Word

```http
POST /api/predict
```

Request:

```json
{
  "text": "I am going to"
}
```

Response:

```json
{
  "suggestions": [
    {
      "word": "school",
      "confidence": 0.85
    }
  ]
}
```

---

### Get Model Stats

```http
GET /api/stats
```

---

### Health Check

```http
GET /api/health
```

---

## 📚 Learning Concepts

This project helps understand:

* NLP Basics
* N-grams
* Flask APIs
* Backend Development
* Machine Learning Workflow
* Text Processing

---

## 🌟 Future Improvements

* Deep Learning based prediction
* Transformer / LSTM integration
* Better UI/UX
* Multi-language support
* Voice typing support
* Auto sentence completion

---

## 🤝 Contributing

Contributions are always welcome!

If you'd like to improve this project:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

---

## 👩‍💻 Author

**Bhagyasri Korlam**

Passionate about:

* Artificial Intelligence
* Full Stack Development
* NLP Projects
* Web Development

Projects worked on:

* AI Typing Analyzer
* Aquaculture Agency Website
* Word Predictor

---

## ⭐ Support

If you like this project:

🌟 Star the repository
🍴 Fork the project
📢 Share with others

---

## 📜 License

This project is open-source and available under the MIT License.

---

Based on your Flask app and predictor engine implementation. 
