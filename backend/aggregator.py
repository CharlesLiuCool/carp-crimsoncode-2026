import os
import sys

# Ensure repo root is on path so "model" package is found
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
from model.model import HospitalModel

def aggregate():
    # Absolute paths for safety
    temp_folder = os.path.abspath("backend/temp")
    public_model_folder = os.path.abspath("backend/public_model")

    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(public_model_folder, exist_ok=True)

    files = [f for f in os.listdir(temp_folder) if f.endswith(".pt")]
    print(f"Files to aggregate: {files}")  # Debug print

    if not files:
        print("No files to aggregate")
        return

    # Initialize a model to get state_dict
    model = HospitalModel()
    state_dict = model.state_dict()

    # Zero out state dict for averaging
    for key in state_dict:
        state_dict[key] = torch.zeros_like(state_dict[key])

    count = 0
    for file in files:
        file_path = os.path.join(temp_folder, file)
        print(f"Loading weights from {file_path}")  # Debug print
        w = torch.load(file_path)
        for key in state_dict:
            state_dict[key] += w[key]
        count += 1

    # Average weights
    for key in state_dict:
        state_dict[key] /= count

    model.load_state_dict(state_dict)

    # Save aggregated public model
    aggregated_path = os.path.join(public_model_folder, "latest_model.pt")
    torch.save(model.state_dict(), aggregated_path)
    print(f"Aggregated {count} models successfully")
    print(f"Aggregated model saved at: {aggregated_path}")