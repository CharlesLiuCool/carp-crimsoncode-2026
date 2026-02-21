# hospital_client/test_dp_model.py
import os
import sys
import torch
import joblib  # for saving/loading scaler
from model.model import HospitalModel

# ---------------- REPO ROOT ----------------
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ---------------- FILE PATHS ----------------
dp_model_file = os.path.join(repo_root, "model/private_weights.pt")
scaler_file = os.path.join(repo_root, "model/scaler.pkl")  # make sure you save scaler during training

# ---------------- LOAD SCALER ----------------
if not os.path.exists(scaler_file):
    raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
scaler = joblib.load(scaler_file)

# ---------------- LOAD DP MODEL ----------------
state_dict = torch.load(dp_model_file)

# Strip "_module." prefix if present
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("_module."):
        new_key = k[len("_module."):]
    else:
        new_key = k
    new_state_dict[new_key] = v

model = HospitalModel()
model.load_state_dict(new_state_dict)
model.eval()

# ---------------- TEST PATIENTS ----------------
patients = [
    [8, 200, 90, 35, 200, 40.0, 1.5, 55],  # high-risk
    [1, 90, 70, 20, 50, 22.0, 0.2, 25],    # healthy
    [6, 150, 80, 25, 0, 30.0, 0.5, 40]     # medium risk
]

# Scale features
X_test = torch.tensor(scaler.transform(patients), dtype=torch.float32)

# ---------------- PREDICTIONS ----------------
with torch.no_grad():
    logits = model(X_test)
    # optional clamp to avoid extreme values in demo
    logits = torch.clamp(logits, -10, 10)
    probs = torch.sigmoid(logits)

for i, prob in enumerate(probs):
    print(f"Patient {i+1} probability of diabetes: {prob.item():.4f}")