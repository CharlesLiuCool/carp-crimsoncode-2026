from fastapi import FastAPI
import torch
import os
import json
import numpy as np
from backend.model import HospitalModel

app = FastAPI()
public_model_path = "backend/public_model/latest_model.pt"
scale_params_path = "hospital_client/scale_params.json"

model = HospitalModel()
model.load_state_dict(torch.load(public_model_path))
model.eval()

scale_params = None
if os.path.isfile(scale_params_path):
    with open(scale_params_path) as f:
        scale_params = json.load(f)

@app.get("/predict")
def predict():
    # Example dummy input (same feature order as training: Pregnancies, Glucose, ...)
    raw = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], dtype=np.float64)
    if scale_params is not None:
        mean = np.array(scale_params["mean"])
        std = np.array(scale_params["std"])
        std[std == 0] = 1.0
        raw = (raw - mean) / std
    x = torch.tensor(raw, dtype=torch.float32)
    prob = model(x).item()  # probability in [0, 1]
    diabetes_percent = round(prob * 100, 2)
    return {"prediction": diabetes_percent, "diabetes_probability_percent": diabetes_percent}