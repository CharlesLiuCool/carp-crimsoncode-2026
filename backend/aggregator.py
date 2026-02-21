import os
import sys

# Ensure repo root is on path so "model" package is found
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
from model.model import HospitalModel

def aggregate():
    folder = "backend/temp"
    os.makedirs(folder, exist_ok=True)
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]

    if not files:
        print("No files to aggregate")
        return

    # Initialize model
    model = HospitalModel()
    state_dict = model.state_dict()

    # Zero out weights for averaging
    for key in state_dict:
        state_dict[key] = torch.zeros_like(state_dict[key])

    count = 0
    for file in files:
        w_path = os.path.join(folder, file)
        w = torch.load(w_path)

        # If the weights were saved from a GradSampleModule, unwrap
        if any(k.startswith("_module.") for k in w.keys()):
            # Map keys from _module.* -> *
            w = {k.replace("_module.", ""): v for k, v in w.items()}

        # Add to accumulator
        for key in state_dict:
            if key in w:
                state_dict[key] += w[key]
            else:
                print(f"Warning: key {key} not found in {file}")
        count += 1

    # Average weights
    for key in state_dict:
        state_dict[key] /= count

    # Load averaged weights into model
    model.load_state_dict(state_dict)

    # Save aggregated public model
    os.makedirs("backend/public_model", exist_ok=True)
    torch.save(model.state_dict(), "backend/public_model/latest_model.pt")
    print(f"Aggregated {count} models successfully")