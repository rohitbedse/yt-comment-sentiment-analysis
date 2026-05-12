# app.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot


from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import os
import json
import pandas as pd
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates

app = Flask(__name__)
CORS(app)

# ── Gemini API helper (used ONLY for Content Ideas + Smart Replies) ──────────
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

def call_gemini(prompt, max_tokens=800):
    """Lightweight Gemini call via REST — no SDK needed."""
    import requests as req
    if not GEMINI_API_KEY:
        return None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}],
                   "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7}}
        r = req.post(url, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        app.logger.error(f"Gemini error: {e}")
    return None


# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_comment(comment):
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


# -------------------------------
# Load Model + Vectorizer
# -------------------------------AC
def load_model_and_vectorizer(model_name, model_version):
    mlflow.set_tracking_uri("https://dagshub.com/rohitbedse/yt-comment-sentiment-analysis.mlflow")

    # Load model
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Get run_id from model version
    client = MlflowClient()
    model_details = client.get_model_version(model_name, model_version)
    run_id = model_details.run_id

    # 🔥 Download vectorizer from artifacts
    vectorizer_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="tfidf_vectorizer.pkl"   # ⚠️ important (no folder)
    )

    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer

model, vectorizer = load_model_and_vectorizer(
    "yt_chrome_plugin_model",
    "2"
)


@app.route('/')
def home():
    return "Welcome to our flask api"


# ----------------------------------------------------------
# PREDICT WITH TIMESTAMPS  (⚠ FIXED: csr_matrix → DataFrame)
# ----------------------------------------------------------
@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        preprocessed_comments = [preprocess_comment(c) for c in comments]

        transformed = vectorizer.transform(preprocessed_comments)

        # FIX: Convert sparse matrix to DataFrame
        df_transformed = pd.DataFrame(
            transformed.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        predictions = model.predict(df_transformed).tolist()
        predictions = [str(p) for p in predictions]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [
        {"comment": c, "sentiment": s, "timestamp": t}
        for c, s, t in zip(comments, predictions, timestamps)
    ]

    return jsonify(response)


# ----------------------------------------------------------
# NORMAL PREDICT  (⚠ FIXED: csr_matrix → DataFrame)
# ----------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)

        # FIX: Convert sparse matrix to DataFrame
        df_transformed = pd.DataFrame(
            transformed.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        predictions = model.predict(df_transformed).tolist()
        predictions = [str(p) for p in predictions]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]
    return jsonify(response)


# --------------------------------
# CHART ENDPOINTS (unchanged)
# --------------------------------
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=140, textprops={'color': 'w'})
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed = [preprocess_comment(c) for c in comments]
        text = ' '.join(preprocessed)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        df['sentiment'] = df['sentiment'].astype(int)

        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for s in [-1, 0, 1]:
            if s not in monthly_percentages.columns:
                monthly_percentages[s] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}

        for s in [-1, 0, 1]:
            plt.plot(monthly_percentages.index,
                     monthly_percentages[s],
                     marker='o',
                     label=sentiment_labels[s],
                     color=colors[s])

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500



# ══════════════════════════════════════════════════════════════════════════════
# CREATOR ANALYTICS ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Question Extractor (rule-based) ───────────────────────────────────────
@app.route('/extract_questions', methods=['POST'])
def extract_questions():
    try:
        comments = request.json.get('comments', [])
        questions = []
        categories = defaultdict(list)

        q_patterns = [
            r'\?',
            r'^(how|what|when|where|why|who|which|can|could|would|should|is|are|do|does|did|will)\b',
        ]

        for c in comments:
            text = c.get('text', '')
            is_q = False
            for pat in q_patterns:
                if re.search(pat, text, re.IGNORECASE):
                    is_q = True
                    break
            if is_q:
                questions.append(text)
                # Simple categorization
                lower = text.lower()
                if any(w in lower for w in ['tutorial', 'how to', 'teach', 'learn', 'explain']):
                    categories['Tutorial requests'].append(text)
                elif any(w in lower for w in ['camera', 'mic', 'gear', 'equipment', 'setup', 'software', 'app']):
                    categories['Gear questions'].append(text)
                elif any(w in lower for w in ['collab', 'feature', 'together', 'join']):
                    categories['Collab asks'].append(text)
                else:
                    categories['General questions'].append(text)

        return jsonify({
            'total': len(questions),
            'questions': questions[:50],
            'categories': {k: {'count': len(v), 'samples': v[:5]} for k, v in categories.items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── 2. Controversy Score (rule-based) ────────────────────────────────────────
@app.route('/controversy_score', methods=['POST'])
def controversy_score():
    try:
        predictions = request.json.get('predictions', [])
        total = len(predictions)
        if total == 0:
            return jsonify({'score': 0, 'level': 'N/A'})

        pos = sum(1 for p in predictions if str(p.get('sentiment')) == '1')
        neg = sum(1 for p in predictions if str(p.get('sentiment')) == '-1')
        neu = sum(1 for p in predictions if str(p.get('sentiment')) == '0')

        pos_pct = pos / total
        neg_pct = neg / total

        # Controversy = how evenly split pos vs neg (ignoring neutral)
        polar = pos + neg
        if polar == 0:
            score = 0
        else:
            balance = 1 - abs(pos_pct - neg_pct)
            polarization = polar / total
            score = round(balance * polarization * 10, 1)

        level = 'Low controversy' if score < 3 else 'Moderate controversy' if score < 6 else 'High controversy'

        return jsonify({
            'score': score,
            'level': level,
            'positive_pct': round(pos_pct * 100, 1),
            'negative_pct': round(neg_pct * 100, 1),
            'neutral_pct': round(neu / total * 100, 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── 3. Top Fan Detector (rule-based) ────────────────────────────────────────
@app.route('/top_fans', methods=['POST'])
def top_fans():
    try:
        data = request.json
        comments = data.get('comments', [])
        predictions = data.get('predictions', [])

        fan_data = defaultdict(lambda: {'count': 0, 'positive': 0, 'total_words': 0, 'author': ''})

        for c, p in zip(comments, predictions):
            uid = c.get('authorId', 'Unknown')
            author = c.get('authorName', uid[:12])
            fan_data[uid]['count'] += 1
            fan_data[uid]['author'] = author
            fan_data[uid]['total_words'] += len(c.get('text', '').split())
            if str(p.get('sentiment')) == '1':
                fan_data[uid]['positive'] += 1

        fans = []
        for uid, d in fan_data.items():
            if d['count'] >= 2:  # At least 2 comments
                pos_rate = round((d['positive'] / d['count']) * 100)
                fans.append({
                    'author': d['author'],
                    'comment_count': d['count'],
                    'positive_rate': pos_rate,
                    'avg_words': round(d['total_words'] / d['count']),
                    'score': d['count'] * (pos_rate / 100) * 10
                })

        fans.sort(key=lambda x: x['score'], reverse=True)
        return jsonify({'fans': fans[:10]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── 4. Complaint Clusters (rule-based) ──────────────────────────────────────
@app.route('/complaint_clusters', methods=['POST'])
def complaint_clusters():
    try:
        predictions = request.json.get('predictions', [])
        negatives = [p['comment'] for p in predictions if str(p.get('sentiment')) == '-1']

        cluster_keywords = {
            'Audio issues': ['audio', 'sound', 'mic', 'volume', 'loud', 'quiet', 'hear'],
            'Too many ads': ['ad', 'ads', 'sponsor', 'sponsored', 'promotion', 'sellout'],
            'Video too long': ['long', 'boring', 'dragged', 'shorter', 'too long', 'lengthy', 'skip'],
            'Content quality': ['quality', 'effort', 'lazy', 'bad', 'worse', 'terrible', 'awful'],
            'Clickbait': ['clickbait', 'misleading', 'title', 'thumbnail', 'lied', 'fake'],
            'Pacing issues': ['pacing', 'slow', 'fast', 'rushed', 'pace', 'speed'],
        }

        clusters = {}
        for cat, keywords in cluster_keywords.items():
            matching = [c for c in negatives if any(k in c.lower() for k in keywords)]
            if matching:
                clusters[cat] = {'count': len(matching), 'samples': matching[:3]}

        # Catch uncategorized negatives
        categorized = set()
        for cat_data in clusters.values():
            categorized.update(cat_data['samples'])
        uncategorized = [c for c in negatives if c not in categorized]
        if uncategorized:
            clusters['Other complaints'] = {'count': len(uncategorized), 'samples': uncategorized[:3]}

        return jsonify({'clusters': clusters, 'total_negative': len(negatives)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── 5. Content Idea Miner (✨ Gemini-powered) ───────────────────────────────
@app.route('/content_ideas', methods=['POST'])
def content_ideas():
    try:
        comments = request.json.get('comments', [])
        questions = [c for c in comments if '?' in c.get('text', '')]
        sample_comments = [c.get('text', '')[:150] for c in comments[:80]]

        prompt = f"""You are a YouTube content strategist. Based on these audience comments, suggest exactly 5 video ideas that would resonate with this audience.

Top audience questions:
{chr(10).join([q.get('text','')[:100] for q in questions[:15]])}

Sample comments:
{chr(10).join(sample_comments[:30])}

Return ONLY a JSON array of 5 objects with keys: "title" (catchy video title), "reason" (1 sentence why this would work based on audience demand). No markdown, no explanation, just the JSON array."""

        result = call_gemini(prompt, max_tokens=600)
        if result:
            # Clean markdown fences if present
            cleaned = re.sub(r'```json\s*', '', result)
            cleaned = re.sub(r'```\s*', '', cleaned).strip()
            ideas = json.loads(cleaned)
            return jsonify({'ideas': ideas, 'powered_by': 'gemini'})

        # Fallback: rule-based suggestions
        word_freq = Counter()
        for c in comments:
            words = re.findall(r'\b[a-z]{4,}\b', c.get('text', '').lower())
            word_freq.update(words)

        stop = set(stopwords.words('english'))
        topics = [w for w, _ in word_freq.most_common(30) if w not in stop][:5]
        ideas = [{'title': f'Deep dive into: {t.title()}', 'reason': 'Frequently mentioned by your audience'} for t in topics]
        return jsonify({'ideas': ideas, 'powered_by': 'fallback'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── 6. Hype Moment Finder (rule-based) ──────────────────────────────────────
@app.route('/hype_moments', methods=['POST'])
def hype_moments():
    try:
        predictions = request.json.get('predictions', [])
        timestamp_pattern = r'\b(\d{1,2}:\d{2}(?::\d{2})?)\b'

        moments = defaultdict(lambda: {'count': 0, 'positive': 0, 'mentions': []})
        for p in predictions:
            text = p.get('comment', '')
            stamps = re.findall(timestamp_pattern, text)
            sent = str(p.get('sentiment'))
            for ts in stamps:
                moments[ts]['count'] += 1
                if sent == '1':
                    moments[ts]['positive'] += 1
                moments[ts]['mentions'].append(text[:100])

        result = []
        for ts, d in moments.items():
            result.append({
                'timestamp': ts,
                'mention_count': d['count'],
                'positive_count': d['positive'],
                'positive_ratio': round(d['positive'] / d['count'] * 100) if d['count'] else 0,
                'sample_mentions': d['mentions'][:3]
            })

        result.sort(key=lambda x: x['mention_count'], reverse=True)
        return jsonify({'moments': result[:10]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── 7. Smart Reply Drafts (✨ Gemini-powered) ───────────────────────────────
@app.route('/smart_replies', methods=['POST'])
def smart_replies():
    try:
        predictions = request.json.get('predictions', [])

        # Pick top negative + question comments worth replying to
        reply_candidates = []
        for p in predictions:
            text = p.get('comment', '')
            sent = str(p.get('sentiment'))
            if sent == '-1' or '?' in text:
                reply_candidates.append(text)
            if len(reply_candidates) >= 8:
                break

        if not reply_candidates:
            return jsonify({'replies': []})

        prompt = f"""You are a YouTube creator responding to comments. Generate helpful, warm, professional reply drafts.

Comments needing replies:
{chr(10).join([f'{i+1}. "{c[:150]}"' for i, c in enumerate(reply_candidates[:5])])}

Return ONLY a JSON array of objects with keys: "comment" (original comment, abbreviated), "reply" (your suggested reply, max 2 sentences, friendly tone). No markdown, just JSON array."""

        result = call_gemini(prompt, max_tokens=600)
        if result:
            cleaned = re.sub(r'```json\s*', '', result)
            cleaned = re.sub(r'```\s*', '', cleaned).strip()
            replies = json.loads(cleaned)
            return jsonify({'replies': replies[:5], 'powered_by': 'gemini'})

        # Fallback
        fallback = [{'comment': c[:80], 'reply': 'Thanks for your feedback! We appreciate you watching.'} for c in reply_candidates[:3]]
        return jsonify({'replies': fallback, 'powered_by': 'fallback'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── 8. Community Health Score (rule-based) ──────────────────────────────────
@app.route('/community_health', methods=['POST'])
def community_health():
    try:
        data = request.json
        predictions = data.get('predictions', [])
        comments = data.get('comments', [])
        total = len(predictions)
        if total == 0:
            return jsonify({'score': 0})

        pos = sum(1 for p in predictions if str(p.get('sentiment')) == '1')
        neg = sum(1 for p in predictions if str(p.get('sentiment')) == '-1')

        # Toxicity signals
        toxic_words = ['hate', 'stupid', 'idiot', 'trash', 'garbage', 'dumb', 'worst', 'kill', 'die', 'shut up']
        toxic_count = sum(1 for p in predictions if any(w in p.get('comment', '').lower() for w in toxic_words))

        # Engagement depth
        avg_length = np.mean([len(c.get('text', '').split()) for c in comments]) if comments else 0
        depth_score = min(avg_length / 20, 1) * 100  # normalize: 20+ words = max

        # Question ratio (engagement signal)
        q_count = sum(1 for c in comments if '?' in c.get('text', ''))
        q_ratio = q_count / total * 100

        # Unique users ratio
        unique = len(set(c.get('authorId', '') for c in comments))
        returning = total - unique
        returning_ratio = returning / total * 100 if total > 0 else 0

        # Composite score
        sentiment_score = (pos / total) * 40  # 40% weight
        toxicity_penalty = (toxic_count / total) * 30  # 30% penalty
        engagement_bonus = (depth_score / 100) * 15 + (q_ratio / 100) * 15  # 30% bonus

        health = round(min(max(sentiment_score - toxicity_penalty + engagement_bonus + 30, 0), 100))

        if health >= 75:
            label = 'Healthy'
        elif health >= 50:
            label = 'Moderate'
        elif health >= 25:
            label = 'Needs Attention'
        else:
            label = 'Concerning'

        toxicity_level = 'Low' if toxic_count / total < 0.05 else 'Medium' if toxic_count / total < 0.15 else 'High'

        return jsonify({
            'score': health,
            'label': label,
            'toxicity': toxicity_level,
            'question_ratio': round(q_ratio, 1),
            'engagement_depth': round(depth_score, 1),
            'returning_fans': returning,
            'details': {
                'sentiment_contribution': round(sentiment_score),
                'toxicity_penalty': round(toxicity_penalty),
                'engagement_bonus': round(engagement_bonus)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)