import os
import torch
import pandas as pd
import requests
from backend.model import HospitalModel

# ---------- HOSPITAL CSVS ----------
hospitals = ["hospital_A.csv", "hospital_B.csv"]

# Loop through each hospital CSV
for hospital_csv in hospitals:
    hospital_path = os.path.join(os.path.dirname(__file__), hospital_csv)
    print(f"Training DP model for {hospital_csv}...")

    # ---------- LOAD DATA ----------
    df = pd.read_csv(hospital_path)
    X = torch.tensor(df.drop("Outcome", axis=1).values, dtype=torch.float32)
    y = torch.tensor(df["Outcome"].values, dtype=torch.float32).unsqueeze(1)

    # ---------- DATASET AND DATALOADER ----------
    from torch.utils.data import TensorDataset, DataLoader
    from torch import nn, optim
    from opacus import PrivacyEngine

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ---------- MODEL, LOSS, OPTIMIZER ----------
    model = HospitalModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # ---------- ATTACH PRIVACY ENGINE ----------
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    # ---------- TRAINING LOOP ----------
    for epoch in range(5):
        for xb, yb in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # ---------- REPORT DP EPSILON ----------
    epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
    print(f"Finished DP training for {hospital_csv}: (ε, δ)=({epsilon:.2f}, 1e-5)")

    # ---------- SAVE DP WEIGHTS ----------
    # Save the inner model's weights to avoid key mismatch issues
    weights_path = os.path.join(os.path.dirname(__file__), f"{hospital_csv}_dp.pt")
    if hasattr(model, "module"):  # unwrap GradSampleModule
        torch.save(model.module.state_dict(), weights_path)
    else:
        torch.save(model.state_dict(), weights_path)
    print(f"Saved DP weights for {hospital_csv}")

    # ---------- UPLOAD WEIGHTS ----------
    url = "http://127.0.0.1:8000/upload"
    with open(weights_path, "rb") as f:
        response = requests.post(url, files={"file": f})
    print(f"Uploaded weights for {hospital_csv}: Status {response.status_code}, Text {response.text}")

# ---------- TEST AGGREGATED PUBLIC MODEL ----------
public_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/public_model/latest_model.pt"))

if os.path.exists(public_model_path):
    model = HospitalModel()
    model.load_state_dict(torch.load(public_model_path))
    model.eval()
    test_input = torch.tensor([[6,148,72,35,0,33.6,0.627,50]], dtype=torch.float32)
    prediction = model(test_input)
    print(f"Test prediction from public model: {prediction.item():.4f}")
else:
    print(f"No aggregated public model found at {public_model_path}")