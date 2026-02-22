import logging
import os

from dotenv import load_dotenv

# Local: server/backend/.env. Docker: project root .env is mounted at /app/.env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv("/app/.env")

import io
import pickle

import numpy as np
import torch
from aggregate import CENTRAL_WEIGHTS, aggregate, load_central_model
from db import get_max_round_id, init_db
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from gemini import analyse_diagnosis
from key_pool import init_key_pool
from weights import weights_bp

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

# ── Config ───────────────────────────────────────────────────────────────────

BACKEND_DIR = os.path.dirname(__file__)

# Feature scaling parameters — must match the scaler used during hospital
# client training (fit on the shared merged_diabetes.csv dataset).
SCALE_MEAN = np.array(
    [42.27461331040037, 27.344150779168032, 137.77024124847878], dtype=np.float32
)
SCALE_STD = np.array(
    [22.51730211875095, 6.608171633655254, 40.57052832219186], dtype=np.float32
)

# Lower threshold reduces false negatives at the cost of more false positives.
# 0.5 = balanced, 0.35 = more sensitive (recommended for medical screening).
CLASSIFICATION_THRESHOLD: float = 0.45

# ── Blueprints ───────────────────────────────────────────────────────────────

app.register_blueprint(weights_bp, url_prefix="/api/weights")

# ── Startup ───────────────────────────────────────────────────────────────────

_started = False


@app.before_request
def startup():
    global _started
    if _started:
        return
    _started = True

    try:
        init_db()
        app.logger.info("Database initialised.")
    except Exception as exc:
        app.logger.error("Database init failed: %s", exc)

    try:
        max_round = get_max_round_id()
        start_round = (max_round + 1) if max_round is not None else 1
        init_key_pool(initial_round_id=start_round)
        app.logger.info("KeyPool initialised: starting at round_id=%d.", start_round)
    except Exception as exc:
        app.logger.error("KeyPool init failed: %s", exc)

    if os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        app.logger.info("AI guidance configured.")
    else:
        app.logger.warning("AI guidance not configured: set GROQ_API_KEY or GEMINI_API_KEY in project root .env when using Docker.")

    try:
        result = aggregate()
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
            "ai_guidance_configured": bool(
                os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY")
            ),
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
    X = torch.tensor(scaled, dtype=torch.float32, requires_grad=True)

    logit = model(X)
    prob = torch.sigmoid(logit).squeeze()
    prob_val = prob.item()
    prediction = int(prob_val >= CLASSIFICATION_THRESHOLD)
    confidence = round(prob_val, 4)

    # SHAP-style feature contributions: gradient of prob w.r.t. input × input (gradient × input attribution)
    model.zero_grad()
    prob.backward()
    grad = X.grad.detach().numpy().squeeze()
    x_np = X.detach().numpy().squeeze()
    contrib_raw = (grad * x_np).tolist()
    feature_names = ["Age", "BMI", "Glucose"]
    feature_contributions = dict(zip(feature_names, [round(c, 4) for c in contrib_raw]))

    # ── LLM analysis (includes feature contributions in prompt) ───────────────
    analysis = None
    analysis_provider = None
    try:
        analysis, analysis_provider = analyse_diagnosis(
            age=age,
            bmi=bmi,
            glucose=glucose,
            prediction=prediction,
            confidence=confidence,
            feature_contributions=feature_contributions,
        )
    except Exception as exc:
        app.logger.warning("LLM analysis failed: %s", exc)

    response = {
        "prediction": prediction,
        "confidence": confidence,
        "feature_contributions": feature_contributions,
        "message": (
            "Elevated diabetes risk detected. Please consult a physician."
            if prediction == 1
            else "No significant diabetes risk detected."
        ),
    }

    if analysis:
        response["analysis"] = analysis
        response["analysis_provider"] = analysis_provider

    return jsonify(response)


# ── Generate AI health guidance (on-demand, uses gemini.py) ────────────────────


@app.post("/api/diagnose/guidance")
def diagnose_guidance():
    """
    Generate AI health guidance for a patient using the same LLM as diagnosis.
    Request body: { "Age", "BMI", "Glucose", "prediction", "confidence", "feature_contributions" }
    (prediction, confidence, feature_contributions typically from a prior /api/diagnose response).
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

    prediction = data.get("prediction", 0)
    confidence = float(data.get("confidence", 0.0))
    feature_contributions = data.get("feature_contributions") or {}

    try:
        analysis, analysis_provider = analyse_diagnosis(
            age=age,
            bmi=bmi,
            glucose=glucose,
            prediction=int(prediction),
            confidence=confidence,
            feature_contributions=feature_contributions,
        )
    except Exception as exc:
        app.logger.warning("AI guidance failed: %s", exc)
        return jsonify({"detail": str(exc)}), 500

    return jsonify({"analysis": analysis, "analysis_provider": analysis_provider})


# ── Export central model ──────────────────────────────────────────────────────


@app.get("/api/model/export")
def export_model():
    """
    Download the aggregated central model as a .pkl file (state_dict).
    Same idea as the hospital client's Export DP Weights, but for the central model.
    """
    if not os.path.isfile(CENTRAL_WEIGHTS):
        return jsonify(
            {
                "detail": "Central model not found. Upload at least one weight file first."
            }
        ), 404

    try:
        state_dict = torch.load(CENTRAL_WEIGHTS, map_location="cpu", weights_only=True)
        buf = io.BytesIO()
        pickle.dump(state_dict, buf, protocol=pickle.HIGHEST_PROTOCOL)
        buf.seek(0)
    except Exception as exc:
        app.logger.exception("Export model failed")
        return jsonify({"detail": str(exc)}), 500

    return send_file(
        buf,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name="carp_central_model.pkl",
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8001)
