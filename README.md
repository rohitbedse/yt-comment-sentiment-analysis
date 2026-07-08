<div align="center">

# 🎯 YouTube Comment Sentiment Analysis

### An End-to-End MLOps Pipeline with Chrome Extension & Real-Time Analytics

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-945DD6?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Classifier-02569B?style=for-the-badge&logo=microsoft&logoColor=white)](https://lightgbm.readthedocs.io)
[![DagsHub](https://img.shields.io/badge/DagsHub-MLOps-FF6F00?style=for-the-badge&logo=data:image/svg+xml;base64,&logoColor=white)](https://dagshub.com)
[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white)](#-chrome-browser-extension)
[![Gemini AI](https://img.shields.io/badge/Gemini%20AI-Integrated-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](#-gemini-ai-integration)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> **A production-grade ML system** that fetches YouTube comments in real-time, classifies sentiment (Positive / Neutral / Negative) using a fine-tuned LightGBM model, and surfaces actionable creator analytics — all through a sleek Chrome extension and a robust Flask REST API.

<br/>

[Features](#-key-features) · [Architecture](#-system-architecture) · [ML Pipeline](#-ml-pipeline) · [API Reference](#-api-endpoints) · [Setup](#-getting-started) · [Notebooks](#-experiment-notebooks)

</div>

---

## 🔥 Key Features

| Feature | Description |
|:--------|:------------|
| 🧠 **3-Class Sentiment Classification** | Classifies comments as **Positive**, **Neutral**, or **Negative** using a production-tuned LightGBM model |
| 🔄 **Automated ML Pipeline** | 5-stage DVC pipeline — from data ingestion to model registration — fully reproducible |
| 📊 **Experiment Tracking** | MLflow + DagsHub integration for logging params, metrics, artifacts, and model versioning |
| 🧩 **Chrome Browser Extension** | One-click analysis of any YouTube video's comment section directly from the browser |
| 📈 **Visual Analytics Dashboard** | Sentiment pie charts, word clouds, monthly sentiment trend graphs |
| 🤖 **Gemini AI Integration** | AI-powered smart reply drafts & content idea generation based on audience comments |
| 🏥 **Community Health Score** | Composite scoring system factoring toxicity, engagement depth, and returning fans |
| 🔍 **Creator Analytics Suite** | Question extractor, controversy scorer, top fan detector, complaint clusters, hype moment finder |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA & ML PIPELINE                          │
│                                                                     │
│   Raw Data ──▶ Preprocessing ──▶ Feature Engineering ──▶ Training  │
│       │            │                    │                    │       │
│    (DVC)     (NLP Pipeline)      (TF-IDF Trigrams)     (LightGBM)  │
│                                                             │       │
│                                    Model Evaluation ◀───────┘       │
│                                         │                           │
│                                  MLflow Registry                    │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SERVING & APPLICATION                           │
│                                                                     │
│   MLflow Model Registry                                            │
│         │                                                           │
│         ▼                                                           │
│   Flask REST API (8 Endpoints + Analytics Suite)                    │
│         │                                                           │
│         ├──▶ Chrome Extension (Manifest V3)                        │
│         │       └── Real-time Sentiment Analysis                   │
│         │       └── Visual Dashboards                              │
│         │       └── Creator Analytics                              │
│         │                                                           │
│         └──▶ Gemini AI (Content Ideas + Smart Replies)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 ML Pipeline

### DVC-Managed Stages

The entire ML workflow is orchestrated through **DVC (Data Version Control)** with 5 reproducible stages:

```yaml
Stage 1: Data Ingestion       → Train/test split (80/20) with seed for reproducibility
Stage 2: Data Preprocessing   → Advanced NLP pipeline with custom stopword handling
Stage 3: Model Building       → TF-IDF (trigrams) + LightGBM training
Stage 4: Model Evaluation     → Multi-metric evaluation + MLflow logging
Stage 5: Model Registration   → Automated model versioning in MLflow Registry
```

### NLP Preprocessing Pipeline

```
Raw Comment
    │
    ├── Lowercase normalization
    ├── Whitespace stripping
    ├── Newline character removal
    ├── Special character filtering (regex-based)
    ├── Smart stopword removal (retains: 'not', 'but', 'however', 'no', 'yet')
    └── WordNet lemmatization
```

> **Why retain sentiment words?** Standard stopword removal strips "not", "but", etc. — words critical for sentiment context. For example, *"not good"* becomes *"good"* without this fix, flipping the sentiment entirely.

### Feature Engineering

| Parameter | Value | Rationale |
|:----------|:------|:----------|
| **Vectorizer** | TF-IDF | Captures term importance relative to the corpus |
| **N-gram Range** | (1, 3) | Unigrams + Bigrams + Trigrams for contextual phrases |
| **Max Features** | 5,000 | Prevents dimensionality curse while retaining signal |

### Model Selection & Hyperparameter Tuning

**8+ models were trained, compared, and hyperparameter-tuned** across 13+ experiment notebooks:

| Model | Status | Notes |
|:------|:-------|:------|
| **LightGBM** | ✅ Production | LR=0.09, Depth=7, Estimators=322, L1+L2 regularization |
| XGBoost | Evaluated | HPT optimized |
| SVC | Evaluated | HPT optimized |
| Logistic Regression | Evaluated | HPT optimized |
| KNN | Evaluated | HPT optimized |
| Naive Bayes | Evaluated | HPT optimized |
| Random Forest | Evaluated | HPT optimized |
| Stacking Ensemble | Evaluated | Multi-model combinations |

### Class Imbalance Handling

- `is_unbalance=True` in LightGBM configuration
- `class_weight="balanced"` for proportional loss weighting
- Dedicated experiment notebook for imbalanced data strategies

---

## 🔌 API Endpoints

The Flask REST API serves **12+ endpoints** covering predictions, visualizations, and creator analytics:

### Core Prediction

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/predict` | Classify a batch of comments (returns sentiment labels) |
| `POST` | `/predict_with_timestamps` | Classify with timestamp metadata for trend analysis |

### Visual Analytics

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/generate_chart` | Sentiment distribution pie chart (PNG) |
| `POST` | `/generate_wordcloud` | Word cloud from comment text (PNG) |
| `POST` | `/generate_trend_graph` | Monthly sentiment trend line chart (PNG) |

### Creator Analytics Suite

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/extract_questions` | Extracts & categorizes audience questions |
| `POST` | `/controversy_score` | Measures polarization (0–10 scale) |
| `POST` | `/top_fans` | Identifies most engaged positive commenters |
| `POST` | `/complaint_clusters` | Groups negative feedback into actionable clusters |
| `POST` | `/content_ideas` | 🤖 Gemini-powered video ideas from audience signals |
| `POST` | `/hype_moments` | Detects most-discussed timestamps in comments |
| `POST` | `/smart_replies` | 🤖 Gemini-powered reply drafts for key comments |
| `POST` | `/community_health` | Composite health score (toxicity + engagement + sentiment) |

---

## 🤖 Gemini AI Integration

Two endpoints leverage **Google's Gemini 2.0 Flash** for generative capabilities:

- **Content Ideas** — Analyzes audience questions and comment themes to suggest video topics that would resonate with the audience
- **Smart Replies** — Generates warm, professional reply drafts for negative and question-based comments

Both endpoints include **rule-based fallbacks** if the Gemini API is unavailable, ensuring the system remains functional without external dependencies.

---

## 🧩 Chrome Browser Extension

A **Manifest V3** Chrome extension that integrates directly into the YouTube browsing experience:

- **One-click analysis** of any YouTube video's comment section
- **Real-time sentiment classification** powered by the Flask API
- **Visual dashboards** — pie charts, word clouds, trend graphs
- **Creator analytics** — questions, controversies, top fans, complaints
- **AI-powered features** — content ideas and smart reply suggestions
- **Dark theme UI** with modern glassmorphism design

---

## 🛠 Tech Stack

| Category | Technologies |
|:---------|:-------------|
| **ML Framework** | LightGBM, Scikit-learn, NLTK |
| **MLOps** | DVC, MLflow, DagsHub |
| **Backend** | Flask, Flask-CORS |
| **AI/LLM** | Google Gemini 2.0 Flash (REST API) |
| **Frontend** | Chrome Extension (Manifest V3), HTML, CSS, JavaScript |
| **Data Viz** | Matplotlib, Seaborn, WordCloud |
| **NLP** | TF-IDF Vectorizer, WordNet Lemmatizer, NLTK Stopwords |
| **Data** | Pandas, NumPy |
| **Config** | YAML (params.yaml) |
| **Serialization** | Pickle, Joblib |
| **Language** | Python 3.8+ |

---

## 📓 Experiment Notebooks

| # | Notebook | Focus |
|:--|:---------|:------|
| 1 | `yt_comment_analyzer_preprocessing` | Data exploration & preprocessing pipeline |
| 2 | `experiment_1_baseline_model` | Baseline model benchmarking |
| 3 | `experiment_2_bow_tfidf` | Bag of Words vs TF-IDF comparison |
| 4 | `experiment_3_tfidf_(1,3)_max_features` | Trigram feature engineering |
| 5 | `experiment_4_handling_imbalanced_data` | Class imbalance strategies |
| 6 | `experiment_5_xgboost_with_hpt` | XGBoost + hyperparameter tuning |
| 7 | `experiment_5_lightgbm_with_hpt` | LightGBM + hyperparameter tuning |
| 8 | `experiment_5_svc_with_hpt` | SVC + hyperparameter tuning |
| 9 | `experiment_5_lor_with_hpt` | Logistic Regression + HPT |
| 10 | `experiment_5_knn_with_hpt` | KNN + hyperparameter tuning |
| 11 | `experiment_5_naive_bayes_with_hpt` | Naive Bayes + HPT |
| 12 | `experiment_5_random_forest_with_hpt` | Random Forest + HPT |
| 13 | `experiment_6_lightgbm_detailed_hpt` | Deep LightGBM optimization (final) |
| 14 | `lightGBM_final` | Final production model training |
| 15 | `sentiment-analysis-bert-reddit-data` | BERT-based approach exploration |
| 16 | `stacking` | Ensemble stacking experiments |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google Chrome (for the browser extension)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/rohitbedse/yt-comment-sentiment-analysis.git
cd yt-comment-sentiment-analysis

# Create and activate virtual environment
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
myenv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the DVC Pipeline

```bash
# Reproduce the entire ML pipeline
dvc repro
```

This will execute all 5 stages sequentially:
`data_ingestion` → `data_preprocessing` → `model_building` → `model_evaluation` → `model_registration`

### Running the Flask API

```bash
cd flask_app
python app.py
```

The API server starts at `http://localhost:8000`

### Loading the Chrome Extension

1. Open `chrome://extensions/` in Google Chrome
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the `frontend/` directory
5. Navigate to any YouTube video and click the extension icon

---

## 📁 Project Structure

```
yt_comment_sentiment_analysis/
│
├── src/                          # Production source code
│   ├── data/
│   │   ├── data_ingestion.py     # Data loading, cleaning, train/test split
│   │   └── data_preprocessing.py # NLP preprocessing pipeline
│   └── model/
│       ├── model_building.py     # TF-IDF + LightGBM training
│       ├── model_evaluation.py   # Evaluation + MLflow logging
│       └── register_model.py     # Model registry management
│
├── flask_app/
│   └── app.py                    # Flask REST API (12+ endpoints)
│
├── frontend/
│   ├── manifest.json             # Chrome Extension config (Manifest V3)
│   ├── popup.html                # Extension UI
│   └── popup.js                  # Extension logic
│
├── notebooks/                    # 16 experiment notebooks
│
├── data/
│   ├── raw/                      # Original train/test splits
│   └── interim/                  # Preprocessed datasets
│
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                      # Pipeline reproducibility lock
├── params.yaml                   # Hyperparameters & config
├── requirements.txt              # Python dependencies
├── Makefile                      # Build automation
├── setup.py                      # Package setup
├── lgbm_model.pkl                # Trained LightGBM model
├── tfidf_vectorizer.pkl          # Fitted TF-IDF vectorizer
├── experiment_info.json          # MLflow run & model info
└── LICENSE                       # MIT License
```

---

## 📊 Key Metrics

```
✅ 8+ ML models trained & compared
✅ 16 experiment notebooks
✅ 5-stage DVC pipeline
✅ 3-class sentiment classification
✅ 5,000-dimensional feature space (TF-IDF with trigrams)
✅ 12+ REST API endpoints
✅ Real-time Chrome Extension
✅ Gemini AI integration with fallback logic
✅ MLflow experiment tracking & model registry
✅ Production-grade logging & error handling
```

---

## ☁️ AWS Deployment

> [!NOTE]
> **The AWS-hosted deployment of this project has been taken down** due to recurring cloud infrastructure charges. The project was previously deployed and fully functional on AWS, demonstrating end-to-end cloud deployment capabilities.
>
> However, the entire project is **fully functional locally** — you can run the Flask API and Chrome Extension on your machine by following the [Getting Started](#-getting-started) section above.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Rohit Bedse](https://github.com/rohitbedse)**

*If you found this project valuable, consider giving it a ⭐*

</div>
