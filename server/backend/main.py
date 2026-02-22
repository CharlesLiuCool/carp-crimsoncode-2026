"""
Central CARP server backend.

Setup:
    pip install -r requirements.txt

Run (from server/backend/):
    source .venv/bin/activate
    python main.py

Listens on http://localhost:8000
The Vite dev server proxies /api/* here automatically.
"""

import logging
import os

import numpy as np
import torch
from aggregate import aggregate, load_central_model
from flask import Flask, jsonify, request
from flask_cors import CORS
from weights import weights_bp

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

# ── Config ───────────────────────────────────────────────────────────────────

BACKEND_DIR = os.path.dirname(__file__)

app.config["WEIGHTS_UPLOAD_DIR"] = os.path.join(BACKEND_DIR, "uploaded_weights")

# Feature scaling parameters — must match the scaler used during hospital
# client training (fit on the shared merged_diabetes.csv dataset).
SCALE_MEAN = np.array(
    [42.27461331040037, 27.344150779168032, 137.77024124847878], dtype=np.float32
)
SCALE_STD = np.array(
    [22.51730211875095, 6.608171633655254, 40.57052832219186], dtype=np.float32
)

# ── Blueprints ───────────────────────────────────────────────────────────────

app.register_blueprint(weights_bp, url_prefix="/api/weights")

# ── Startup: aggregate any already-uploaded weights ──────────────────────────

with app.app_context():
    upload_dir = app.config["WEIGHTS_UPLOAD_DIR"]
    os.makedirs(upload_dir, exist_ok=True)
    try:
        result = aggregate(upload_dir=upload_dir)
        app.logger.info(
            "Startup aggregation: %d file(s) aggregated, %d skipped",
            result["aggregated"],
            result["skipped"],
        )
    except Exception as exc:
        app.logger.warning("Startup aggregation skipped: %s", exc)


# ── Health ────────────────────────────────────────────────────────────────────


@app.route("/api/health")
def health():
    from aggregate import CENTRAL_WEIGHTS

    return jsonify(
        {
            "status": "ok",
            "central_model": os.path.isfile(CENTRAL_WEIGHTS),
        }
    )


# ── Diagnose ──────────────────────────────────────────────────────────────────


@app.post("/api/diagnose")
def diagnose():
    """
    Run inference on a single patient using the central aggregated model.

    Request body (JSON):
        { "Age": 45, "BMI": 30.5, "Glucose": 140 }

    Response:
        {
            "prediction":  0 | 1,
            "confidence":  float,   # probability of diabetes (0–1)
            "message":     str
        }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"detail": "JSON body required."}), 400

    missing = [f for f in ["Age", "BMI", "Glucose"] if f not in data]
    if missing:
        return jsonify({"detail": f"Missing fields: {missing}"}), 400

    try:
        age = float(data["Age"])
        bmi = float(data["BMI"])
        glucose = float(data["Glucose"])
    except (TypeError, ValueError):
        return jsonify({"detail": "Age, BMI, and Glucose must be numbers."}), 400

    try:
        model = load_central_model()
    except FileNotFoundError as exc:
        return jsonify({"detail": str(exc)}), 503

    # Standardise features using the shared scaler parameters
    raw = np.array([[age, bmi, glucose]], dtype=np.float32)
    scaled = (raw - SCALE_MEAN) / SCALE_STD
    X = torch.tensor(scaled)

    with torch.no_grad():
        logit = model(X)
        prob = torch.sigmoid(logit).item()

    prediction = int(prob >= 0.5)

    return jsonify(
        {
            "prediction": prediction,
            "confidence": round(prob, 4),
            "message": (
                "Elevated diabetes risk detected. Please consult a physician."
                if prediction == 1
                else "No significant diabetes risk detected."
            ),
        }
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8001)
