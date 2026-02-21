import os
import sys
import torch
import numpy as np

# Add repo root to path so sibling 'model/' can be imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from model.model import HospitalModel

# ---------- CONFIG ----------
dp_model_file = "model/private_weights.pt"  # DP-trained model

# ---------- SYNTHETIC TEST PATIENTS ----------
# 1. Obvious diabetes: high glucose, high BMI, older age, high pregnancies
# 2. Obvious non-diabetes: healthy ranges
example_patients = [
    [8, 200, 90, 35, 200, 40.0, 1.5, 55],  # High-risk diabetes
    [1, 90, 70, 20, 50, 22.0, 0.2, 25],    # Healthy patient
]

# ---------- LOAD DP MODEL ----------
if not os.path.exists(dp_model_file):
    raise FileNotFoundError(f"DP model file not found: {dp_model_file}")

dp_model = HospitalModel()
state_dict = torch.load(dp_model_file)

# If keys are prefixed with "_module.", unwrap
if list(state_dict.keys())[0].startswith("_module."):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_module.", "")
        new_state_dict[new_key] = v
    state_dict = new_state_dict

dp_model.load_state_dict(state_dict)
dp_model.eval()

# ---------- TEST ----------
inputs = torch.tensor(example_patients, dtype=torch.float32)

# Optional: if you trained with normalized features, scale test inputs
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(training_X)  # fit on training data
# inputs = torch.tensor(scaler.transform(example_patients), dtype=torch.float32)

with torch.no_grad():
    logits = dp_model(inputs)
    probs = torch.sigmoid(logits)

for i, (logit, prob) in enumerate(zip(logits, probs)):
    print(f"Patient {i+1}: raw logit = {logit.item():.4f}, probability = {prob.item():.4f}")