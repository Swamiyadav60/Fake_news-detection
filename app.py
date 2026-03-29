"""
Flask application for Fake News Detection.
"""

import os
import pickle
import logging
from flask import Flask, render_template, request, jsonify

from train_model import ModelTrainer
from db_manager import DatabaseManager
from utils import (
    preprocess_text,
    validate_input,
    truncate_text,
    ModelNotFoundError,
    VectorizerNotFoundError,
    InvalidInputError,
    DatabaseError
)

# ============================================================
# Flask Config
# ============================================================

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Globals
# ============================================================

model = None
vectorizer = None
db_manager = None
model_name = "Unknown"


# ============================================================
# Load Models
# ============================================================

def load_models():
    global model, vectorizer, model_name

    if not os.path.exists("model.pkl"):
        raise ModelNotFoundError("model.pkl not found")

    if not os.path.exists("vectorizer.pkl"):
        raise VectorizerNotFoundError("vectorizer.pkl not found")

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    if os.path.exists("model_metadata.pkl"):
        with open("model_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            model_name = metadata.get("model_name", "Unknown")

    logger.info("Models loaded successfully")


# ============================================================
# Initialize App
# ============================================================

def init_app():
    global db_manager

    logger.info("Initializing app...")

    db_manager = DatabaseManager("database.db")

    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        logger.info("No model found. Training automatically...")
        trainer = ModelTrainer()
        trainer.train_models()

    load_models()

    logger.info("Initialization complete")


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    return render_template("index.html", model_name=model_name)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Request must be JSON"
            }), 400

        data = request.get_json()

        if "text" not in data:
            return jsonify({
                "success": False,
                "error": 'Missing "text" field'
            }), 400

        text = data["text"]

        valid, error = validate_input(text)

        if not valid:
            return jsonify({
                "success": False,
                "error": error
            }), 400

        # Preprocess
        clean_text = preprocess_text(text)

        # Vectorize
        text_vector = vectorizer.transform([clean_text])

        # Predict
        prediction = model.predict(text_vector)[0]

        # Confidence safe for all models
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(text_vector)[0].max())

        elif hasattr(model, "decision_function"):
            score = model.decision_function(text_vector)[0]
            confidence = min(abs(score) / 5, 1.0)

        else:
            confidence = 0.75

        label = "fake" if prediction == 1 else "real"

        # Save history
        try:
            db_manager.log_prediction(text, label, confidence)
        except Exception as db_error:
            logger.warning(f"DB log failed: {db_error}")

        return jsonify({
            "success": True,
            "label": label,
            "confidence": round(confidence, 4),
            "text": truncate_text(text, 200)
        }), 200

    except InvalidInputError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "success": False,
            "error": "Prediction failed"
        }), 500


@app.route("/api/stats")
def stats():
    try:
        stats_data = db_manager.get_statistics()

        return jsonify({
            "success": True,
            **stats_data
        })

    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to load stats"
        }), 500


@app.route("/api/history")
def history():
    try:
        limit = request.args.get("limit", 10, type=int)
        limit = min(limit, 100)

        predictions = db_manager.get_recent_predictions(limit)

        return jsonify({
            "success": True,
            "predictions": predictions
        })

    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to load history"
        }), 500


@app.route("/api/retrain", methods=["POST"])
def retrain():
    try:
        trainer = ModelTrainer()
        trainer.train_models()

        load_models()

        return jsonify({
            "success": True,
            "message": "Retraining completed",
            "model_name": model_name
        })

    except Exception as e:
        logger.error(f"Retrain error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# Error Handlers
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    try:
        init_app()

        print("\n" + "=" * 70)
        print("FAKE NEWS DETECTION WEB APP")
        print("=" * 70)
        print(f"Model: {model_name}")
        print("URL: http://localhost:5000")
        print("=" * 70 + "\n")

        app.run(
            host="127.0.0.1",
            port=5500,
            debug=False
        )

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        print(f"Startup failed: {e}")