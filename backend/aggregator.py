import os
import torch
import pandas as pd
from backend.model import HospitalModel

def aggregate(csv_folder="hospital_client/hospital_csvs", temp_folder="backend/temp", public_folder="backend/public_model"):
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(public_folder, exist_ok=True)

    # Get all DP weight files
    files = [f for f in os.listdir(temp_folder) if f.endswith(".pt")]
    if not files:
        print("No DP weight files to aggregate in", temp_folder)
        return

    # Optional: get dataset sizes for weighted averaging
    sizes = []
    for f in files:
        csv_name = f.split("_dp.pt")[0]
        csv_path = os.path.join(csv_folder, csv_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            sizes.append(len(df))
        else:
            sizes.append(1)  # default weight if CSV missing

    total_size = sum(sizes)

    # Initialize model
    model = HospitalModel()
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = torch.zeros_like(state_dict[key])

    # Aggregate weights
    for idx, f in enumerate(files):
        w = torch.load(os.path.join(temp_folder, f))
        weight_factor = sizes[idx] / total_size
        for key in state_dict:
            state_dict[key] += w[key] * weight_factor

    model.load_state_dict(state_dict)

    # Save aggregated public model
    save_path = os.path.join(public_folder, "latest_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Aggregated {len(files)} models successfully into {save_path}")

    # Optional test prediction
    model.eval()
    test_input = torch.tensor([[6,148,72,35,0,33.6,0.627,50]], dtype=torch.float32)
    with torch.no_grad():
        pred = model(test_input)
    print(f"Aggregated model test prediction: {pred.item():.4f}")

    # Return model for immediate use
    return model