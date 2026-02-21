import os
import sys
import shutil
import json
import torch
import numpy as np

# Ensure repo root is on path so "backend" is found
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.aggregator import aggregate
from backend.model import HospitalModel  # Make sure this is the merged model

# -------------------- APP SETUP --------------------
app = FastAPI()

# CORS: allow your frontend to call the API
origins = [
    "http://127.0.0.1:8081",
    "http://localhost:8081"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- UPLOAD FOLDER --------------------
UPLOAD_FOLDER = "backend/temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- LOAD PUBLIC MODEL AT STARTUP --------------------
PUBLIC_MODEL_PATH = "backend/public_model/latest_model.pt"
SCALE_PARAMS_PATH = "hospital_client/scale_params.json"

model = None
scale_params = None

if os.path.isfile(PUBLIC_MODEL_PATH):
    model = HospitalModel()
    model.load_state_dict(torch.load(PUBLIC_MODEL_PATH))
    model.eval()

if os.path.isfile(SCALE_PARAMS_PATH):
    with open(SCALE_PARAMS_PATH) as f:
        scale_params = json.load(f)

# -------------------- UPLOAD AND AGGREGATE --------------------
@app.post("/upload")
async def upload_weights(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Aggregate all DP weights into latest public model
        aggregate()

        # Reload the aggregated model
        global model
        if os.path.isfile(PUBLIC_MODEL_PATH):
            model = HospitalModel()
            model.load_state_dict(torch.load(PUBLIC_MODEL_PATH))
            model.eval()

        return {"status": "aggregated"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# -------------------- PREDICTION ENDPOINT --------------------
@app.get("/predict")
def predict():
    if model is None:
        return {"status": "error", "detail": "No public model found."}

    # Example dummy input (same feature order as training)
    raw = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], dtype=np.float64)

    if scale_params is not None:
        mean = np.array(scale_params["mean"])
        std = np.array(scale_params["std"])
        std[std == 0] = 1.0
        raw = (raw - mean) / std

    x = torch.tensor(raw, dtype=torch.float32)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()  # convert logit to probability

    diabetes_percent = round(prob * 100, 2)
    return {"prediction": diabetes_percent, "diabetes_probability_percent": diabetes_percent}