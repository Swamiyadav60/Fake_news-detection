from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)

# =========================
# Load ML Model
# =========================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# =========================
# Database setup
# =========================
DB_NAME = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            label TEXT,
            confidence REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# =========================
# Home Page
# =========================
@app.route("/")
def home():
    return render_template("index.html", model_name="Fake News Classifier")

# =========================
# Predict API
# =========================
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({
                "success": False,
                "error": "No text provided"
            })

        transformed = vectorizer.transform([text])
        prediction = model.predict(transformed)[0]

        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(transformed)[0])
        else:
            confidence = 0.85

        label = "fake" if prediction == 0 else "real"

        # Save to DB
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (text, label, confidence, created_at)
            VALUES (?, ?, ?, ?)
        """, (text, label, float(confidence), datetime.now().isoformat()))
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "label": label,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# =========================
# Stats API
# =========================
@app.route("/api/stats")
def stats():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE label='fake'")
    fake = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE label='real'")
    real = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(confidence) FROM predictions")
    avg = cursor.fetchone()[0] or 0

    conn.close()

    return jsonify({
        "success": True,
        "total_predictions": total,
        "fake_count": fake,
        "real_count": real,
        "avg_confidence": avg
    })

# =========================
# History API
# =========================
@app.route("/api/history")
def history():
    limit = request.args.get("limit", 10)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT text, label, confidence
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    predictions = []

    for row in rows:
        predictions.append({
            "text": row[0],
            "label": row[1],
            "confidence": row[2]
        })

    return jsonify({
        "success": True,
        "predictions": predictions
    })

# =========================
# Retrain API
# =========================
@app.route("/api/retrain", methods=["POST"])
def retrain():
    return jsonify({
        "success": True,
        "message": "Retrain placeholder complete"
    })

# =========================
# 404 Handler
# =========================
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

# =========================
# Run
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)