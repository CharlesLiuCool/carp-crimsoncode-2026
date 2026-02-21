import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from backend.model import HospitalModel

# ---------------- CONFIG ----------------
# Choose which DP weights to test
dp_model_file = "hospital_client/dp_weights/hospital_A.csv_dp.pt"
# Optional: non-DP public model for comparison
public_model_file = "backend/public_model/latest_model.pt"

# Example patient data (match your features exactly)
example_patients = [
    [6, 148, 72, 35, 0, 33.6, 0.627, 50],
    [1, 85, 66, 29, 0, 26.6, 0.351, 31],
    [8, 183, 64, 0, 0, 23.3, 0.672, 32],
]

# ---------------- LOAD DP MODEL ----------------
dp_model = HospitalModel()
dp_state_dict = torch.load(dp_model_file)

# unwrap GradSampleModule keys
unwrapped_state_dict = {}
for k, v in dp_state_dict.items():
    if k.startswith("_module."):
        unwrapped_state_dict[k[len("_module."):]] = v
    else:
        unwrapped_state_dict[k] = v

dp_model.load_state_dict(unwrapped_state_dict)
dp_model.eval()
print(f"\nTesting DP model from: {dp_model_file}\n")

inputs = torch.tensor(example_patients, dtype=torch.float32)
with torch.no_grad():
    dp_logits = dp_model(inputs)
    dp_probs = torch.sigmoid(dp_logits)

for i, (logit, prob) in enumerate(zip(dp_logits, dp_probs)):
    print(f"Patient {i+1}: raw logit = {logit.item():.4f}, probability = {prob.item():.4f}")

# ---------------- OPTIONAL: COMPARE NON-DP PUBLIC MODEL ----------------
if os.path.exists(public_model_file):
    public_model = HospitalModel()
    public_state_dict = torch.load(public_model_file)
    public_model.load_state_dict(public_state_dict)
    public_model.eval()

    with torch.no_grad():
        public_logits = public_model(inputs)
        public_probs = torch.sigmoid(public_logits)

    print(f"\nComparison with non-DP public model ({public_model_file}):\n")
    for i, (logit, prob) in enumerate(zip(public_logits, public_probs)):
        print(f"Patient {i+1}: raw logit = {logit.item():.4f}, probability = {prob.item():.4f}")
else:
    print(f"\nNo public non-DP model found at {public_model_file}, skipping comparison.")