from flask import Flask, render_template, request, jsonify
import pickle
import os

print("Starting app...")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Base dir:", BASE_DIR)

try:
    model_path = os.path.join(BASE_DIR, "model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

    print("Loading model...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("Loading vectorizer...")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    print("All files loaded successfully")

except Exception as e:
    print("Startup error:", str(e))
    raise

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]

        result = "Fake News" if prediction == 1 else "Real News"

        return jsonify({
            "success": True,
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)