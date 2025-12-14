# 💬 Comment Analyzer

A **Chrome extension + backend system** that analyzes **YouTube video comments in near real time**, classifies sentiment (Positive / Neutral / Negative), and visualizes insights using clean charts.

This project is built with an **honest data-science mindset**: instead of forcing exaggerated positivity or negativity, it reflects the reality that **most YouTube comments are neutral or low-signal**.

---

## 🚀 What Problem This Solves

YouTube comment sections are noisy:

* Fan spam ("Team X ❤️")
* One-word reactions ("Hi", "Yes")
* Emojis, timestamps, and low-effort text

Most tools overpromise sentiment accuracy and mislead users.

**Comment Analyzer focuses on signal over hype** — showing what people *actually* feel, not what looks exciting.

---

## 🧠 Key Features

* 🔍 **Real-time comment fetching** using YouTube Data API v3
* 🧠 **Sentiment classification**: Positive, Neutral, Negative
* 📊 **Interactive charts** showing sentiment distribution
* 💬 **Top comments view** with sentiment labels
* ⚡ Lightweight Chrome extension UI
* 🔌 Backend-powered analysis (scalable design)

---

## 📊 Example Insight

In real YouTube data, sentiment often looks like:

* **Neutral:** Majority (fan tags, short replies, emojis)
* **Positive:** Genuine appreciation, excitement
* **Negative:** Rare but high-signal criticism

This project intentionally preserves that distribution instead of distorting it.

---

## 🛠️ Tech Stack

### Frontend (Chrome Extension)

* JavaScript
* HTML / CSS
* Chart.js

### Backend

* Python
* Flask / FastAPI
* YouTube Data API v3
* NLP-based sentiment model

### Data & MLOps

* pandas, numpy
* Custom preprocessing pipeline
* DVC (Data Version Control)
* Docker

---

## 🧩 Architecture Overview

1. Chrome extension captures YouTube video ID
2. Backend fetches comments via YouTube API
3. Text is cleaned (emojis, spam, noise handling)
4. Sentiment model classifies each comment
5. Aggregated results returned to extension
6. Charts + comment table rendered in UI

---

## 🧪 Sentiment Philosophy (Important)

> Neutral does **not** mean useless.

Neutral comments represent:

* Audience presence
* Fanbase size
* Engagement volume

Positive and negative comments represent **emotional signal**.

This separation is deliberate and interview-defensible.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/comment-analyzer.git
cd comment-analyzer
```

### 2️⃣ Backend setup

```bash
pip install -r requirements.txt
python app.py
```

### 3️⃣ Chrome Extension setup

1. Open `chrome://extensions/`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select the `extension/` directory

---

## 🔐 API Configuration

* Create a YouTube Data API v3 key
* Add it to backend environment variables
* Respect quota limits (default: 10,000 units/day)

---

## 📈 Future Improvements

* Spam / bot comment detection
* Emotion-level classification (joy, anger, sarcasm)
* Time-series sentiment tracking
* Creator dashboard view
* Model fine-tuning on YouTube-specific data

---

## 👨‍💻 Author

**Rohit Kiran**
Final-year Computer Engineering student
Focused on Data Science, Machine Learning, and Applied NLP

---

## ⭐ Why This Project Matters

This is not a flashy demo.

It demonstrates:

* Real-world data realism
* API usage at scale
* NLP preprocessing depth
* Honest ML evaluation
* End-to-end product thinking

Exactly what serious data science roles look for.

---

If you find this useful, feel free to ⭐ the repo or fork it.
