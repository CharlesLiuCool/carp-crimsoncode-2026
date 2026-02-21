import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

import torch
from model.model import HospitalModel

# Load from model folder or backend/public_model
weights_path = os.path.join(os.path.dirname(__file__), "latest_model.pt")
if not os.path.isfile(weights_path):
    weights_path = os.path.join(repo_root, "backend/public_model/latest_model.pt")

model = HospitalModel()
model.load_state_dict(torch.load(weights_path))
model.eval()


def predict(x):
    """Return probability of having diabetes as a percentage (0-100)."""
    x = torch.tensor(x).float()
    prob = model(x).item()
    return round(prob * 100, 2)
