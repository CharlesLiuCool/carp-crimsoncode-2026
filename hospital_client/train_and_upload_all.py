import os
import torch
import pandas as pd
import requests
from model.model import HospitalModel  # Use the new model

# ---------------- CONFIG ----------------
CONFIG = {
    "batch_size": 32,
    "epochs": 20,
    "lr": 0.01,
    "noise_multiplier": 1.5,  # tradeoff: higher noise = more privacy, lower accuracy
    "max_grad_norm": 1.0,
    "delta": 1e-5,
    "upload_url": "http://127.0.0.1:8000/upload",
    "csv_folder": os.path.join(os.path.dirname(__file__), "hospital_csvs"),
    "dp_save_folder": os.path.join(os.path.dirname(__file__), "dp_weights"),
}

os.makedirs(CONFIG["dp_save_folder"], exist_ok=True)

# ---------------- FIND CSV FILES ----------------
hospital_csvs = [
    f for f in os.listdir(CONFIG["csv_folder"]) if f.endswith(".csv")
]

if not hospital_csvs:
    print("No hospital CSV files found in", CONFIG["csv_folder"])
    exit()

# ---------------- TRAIN DP MODEL FOR EACH HOSPITAL ----------------
for csv_file in hospital_csvs:
    hospital_path = os.path.join(CONFIG["csv_folder"], csv_file)
    print(f"\nTraining DP model for {csv_file}...")

    # Load dataset
    df = pd.read_csv(hospital_path)
    X = torch.tensor(df.drop("Outcome", axis=1).values, dtype=torch.float32)
    y = torch.tensor(df["Outcome"].values, dtype=torch.float32).unsqueeze(1)

    from torch.utils.data import TensorDataset, DataLoader
    from torch import nn, optim
    from opacus import PrivacyEngine

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # Initialize model, criterion, optimizer
    model = HospitalModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # Attach DP engine
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=CONFIG["noise_multiplier"],
        max_grad_norm=CONFIG["max_grad_norm"],
    )

    # Training loop
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {epoch_loss:.4f}")

    # Report DP epsilon
    epsilon = privacy_engine.accountant.get_epsilon(delta=CONFIG["delta"])
    print(f"Finished DP training for {csv_file}: (ε, δ)=({epsilon:.2f}, {CONFIG['delta']})")

    # Save DP weights correctly (unwrap GradSampleModule)
    dp_weights_path = os.path.join(CONFIG["dp_save_folder"], f"{csv_file}_dp.pt")
    if hasattr(model, "module"):  # unwrap if wrapped by Opacus
        torch.save(model.module.state_dict(), dp_weights_path)
    else:
        torch.save(model.state_dict(), dp_weights_path)
    print(f"Saved DP weights to {dp_weights_path}")

    # Upload weights to server
    try:
        with open(dp_weights_path, "rb") as f:
            response = requests.post(CONFIG["upload_url"], files={"file": f})
        print(f"Uploaded weights: Status {response.status_code}, Text {response.text}")
    except Exception as e:
        print(f"Failed to upload weights: {e}")

# ---------------- OPTIONAL: TEST AGGREGATED PUBLIC MODEL ----------------
public_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/public_model/latest_model.pt"))
if os.path.exists(public_model_path):
    test_model = HospitalModel()
    state_dict = torch.load(public_model_path)
    # Unwrap keys if needed
    if any(k.startswith("_module.") for k in state_dict.keys()):
        state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
    test_model.load_state_dict(state_dict)
    test_model.eval()
    test_input = torch.tensor([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], dtype=torch.float32)
    pred = torch.sigmoid(test_model(test_input)).item()  # convert logit -> probability
    print(f"\nTest prediction from public model: {pred:.4f}")
else:
    print(f"No aggregated public model found at {public_model_path}")