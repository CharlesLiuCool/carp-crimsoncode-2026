import torch
import os
from backend.model import HospitalModel

def aggregate():
    folder = "backend/temp"
    os.makedirs(folder, exist_ok=True)
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]

    if not files:
        print("No files to aggregate")
        return

    # Load first model to get state_dict shape
    model = HospitalModel()
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = torch.zeros_like(state_dict[key])

    count = 0
    for file in files:
        w = torch.load(os.path.join(folder, file))
        for key in state_dict:
            state_dict[key] += w[key]
        count += 1

    for key in state_dict:
        state_dict[key] /= count

    model.load_state_dict(state_dict)
    os.makedirs("public_model", exist_ok=True)
    torch.save(model.state_dict(), "public_model/latest_model.pt")
    print(f"Aggregated {count} models successfully")