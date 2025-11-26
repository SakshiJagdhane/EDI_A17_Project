from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load Model
MODEL_PATH = "../model/vitamin_model.keras"

# Load model safely
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D", "Vitamin E"]

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"].read()
        symptoms = request.form.get("symptoms", "")

        img = preprocess_image(file)

        preds = model.predict(img)[0]
        idx = np.argmax(preds)
        confidence = float(preds[idx])
        deficiency = CLASS_NAMES[idx]

        # Simple symptoms match logic
        common_keywords = ["dry", "crack", "pale", "yellow", "pain"]
        symptom_match = "High" if any(k in symptoms.lower() for k in common_keywords) else "Low"

        suggestions_map = {
            "Vitamin A": "Eat carrots, sweet potatoes, leafy greens.",
            "Vitamin B": "Include milk, eggs, whole grains.",
            "Vitamin C": "Eat oranges, strawberries, tomatoes.",
            "Vitamin D": "Morning sunlight, fortified milk.",
            "Vitamin E": "Almonds, peanuts, sunflower seeds."
        }

        suggestions = suggestions_map[deficiency]

        return jsonify({
            "deficiency": deficiency,
            "confidence": f"{confidence:.2f}",
            "symptom_match": symptom_match,
            "suggestions": suggestions
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
