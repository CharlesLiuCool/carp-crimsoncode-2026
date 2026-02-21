"""
Setup:
  pip install flask flask-cors scikit-learn joblib numpy

need to figure out if the pkl file needs to be in the same directory will keep this updated

it will be on  http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

#get the exported model pkl
MODEL_PATH = "diabetes_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Put your exported diabetes_model.pkl next to app.py")

model = joblib.load(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}")

# must match training data
GENDER_MAP = {"male": 0, "female": 1, "other": 2}

def build_features(data):
    # order: [Pregnancies, Glucose, BloodPressure, Insulin, BMI,
    #         DiabetesPedigreeFunction, Age, hypertension, gender]
    return np.array([[
        float(data.get("pregnancies", 0)),
        float(data["glucose"]),
        float(data.get("blood_pressure", 80)),
        float(data.get("insulin", 0)),
        float(data["bmi"]),
        float(data.get("DiabetesPedigreeFunction", 0.0)),
        float(data["age"]),
        int(data.get("hypertension", 0)),
        GENDER_MAP.get(str(data.get("gender", "other")).lower(), 2),
    ]])

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        missing = [f for f in ["age", "bmi", "glucose"] if f not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        features    = build_features(data)
        prediction  = int(model.predict(features)[0])
        probability = round(float(model.predict_proba(features)[0][1]) * 100, 1) #it will multiply by 100 to convert to percentage and round to 1 decimal place
        risk_level  = "Low" if probability < 30 else "Medium" if probability < 60 else "High"

        return jsonify({
            "at_risk":     bool(prediction),
            "probability": probability,
            "risk_level":  risk_level,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)