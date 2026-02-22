import io
import os
import tempfile

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from hospital_client.backend.train import (
    FEATURES,
    build_scaler_from_merged,
    predict_batch,
    predict_single,
    train,
)

# ─── App ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MERGED_CSV = os.path.join(os.path.dirname(__file__), "..", "merged_diabetes.csv")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

# ─── Startup: ensure scaler exists ───────────────────────────────────────────
with app.app_context():
    if not os.path.isfile(SCALER_PATH):
        if os.path.isfile(MERGED_CSV):
            print("Building scaler from merged_diabetes.csv...")
            build_scaler_from_merged(MERGED_CSV)
        else:
            print(
                "WARNING: scaler.pkl missing and merged_diabetes.csv not found. "
                "Diagnosis will fail until training is run."
            )


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "scaler": os.path.isfile(SCALER_PATH),
            "weights": os.path.isfile(os.path.join(ARTIFACTS_DIR, "weights.pt")),
            "dp_weights": os.path.isfile(os.path.join(ARTIFACTS_DIR, "dp_weights.pt")),
        }
    )


# ─── Train ────────────────────────────────────────────────────────────────────
@app.post("/api/train")
def train_endpoint():
    """
    Train a new MergedModel on the uploaded hospital CSV.

    Form fields:
      file    : CSV with at least Age, BMI, Glucose, Outcome columns
      epsilon : privacy/accuracy slider value (0.1 = max privacy, 10.0 = max accuracy)
    """
    if "file" not in request.files:
        return jsonify({"detail": "No file provided."}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"detail": "File must be a .csv"}), 400

    try:
        epsilon = float(request.form.get("epsilon", 5.0))
    except ValueError:
        return jsonify({"detail": "epsilon must be a number."}), 400

    contents = file.read()

    try:
        df_check = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        return jsonify({"detail": f"Could not parse CSV: {e}"}), 400

    missing = [c for c in FEATURES + ["Outcome"] if c not in df_check.columns]
    if missing:
        return jsonify(
            {
                "detail": f"CSV is missing required columns: {missing}. "
                f"Expected at least: {FEATURES + ['Outcome']}"
            }
        ), 400

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    use_dp = epsilon < 9.5

    try:
        result = train(csv_path=tmp_path, epsilon=epsilon, use_dp=use_dp)
    except Exception as e:
        return jsonify({"detail": str(e)}), 500
    finally:
        os.unlink(tmp_path)

    eps_spent = result.get("epsilon_spent")
    message = (
        f"Model trained successfully with DP-SGD (ε≈{eps_spent:.2f})."
        if eps_spent
        else "Model trained successfully with no differential privacy."
    )

    return jsonify({"message": message, **result})


# ─── Diagnose (single patient JSON) ──────────────────────────────────────────
@app.post("/api/diagnose")
def diagnose_json():
    """
    Diagnose a single patient from a JSON body.

    Body: { "Age": 45, "BMI": 30.5, "Glucose": 140 }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"detail": "JSON body required."}), 400

    missing = [f for f in ["Age", "BMI", "Glucose"] if f not in data]
    if missing:
        return jsonify({"detail": f"Missing fields: {missing}"}), 400

    try:
        result = predict_single(
            age=float(data["Age"]),
            bmi=float(data["BMI"]),
            glucose=float(data["Glucose"]),
            use_dp=False,
        )
    except FileNotFoundError as e:
        return jsonify({"detail": str(e)}), 503
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

    return jsonify(result)


# ─── Diagnose (CSV batch) ─────────────────────────────────────────────────────
@app.post("/api/diagnose/batch")
def diagnose_csv():
    """
    Diagnose a batch of patients from a CSV file.

    CSV must contain columns: Age, BMI, Glucose
    Returns { count, results: [{row, prediction, confidence, message}] }
    """
    if "file" not in request.files:
        return jsonify({"detail": "No file provided."}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"detail": "File must be a .csv"}), 400

    contents = file.read()

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        return jsonify({"detail": f"Could not parse CSV: {e}"}), 400

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return jsonify({"detail": f"CSV is missing columns: {missing}"}), 400

    try:
        results = predict_batch(df, use_dp=False)
    except FileNotFoundError as e:
        return jsonify({"detail": str(e)}), 503
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

    return jsonify({"count": len(results), "results": results})


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
