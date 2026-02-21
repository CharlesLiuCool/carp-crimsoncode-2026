import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from opacus import PrivacyEngine
from sklearn.preprocessing import StandardScaler
import joblib
from model.model import HospitalModel

# ---------------- CONFIG ----------------
CONFIG = {
    "csv_file": "hospital_client/hospital_csvs/hospital_A.csv",
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-4,
    "noise_multiplier": 0.5,
    "max_grad_norm": 5.0,
    "delta": 1e-5,
    "save_path": "model/private_weights.pt",
    "scaler_path": "model/scaler.pkl"
}

# ---------------- PATH SETUP ----------------
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ---------------- NORMALIZATION FUNCTION ----------------
def normalize_probability_bandaid(probs):
    """
    Rescale probabilities so:
    0.5 -> 0.0
    1.0 -> 1.0
    Clamp outside range
    """
    return torch.clamp((probs - 0.5) / 0.5, 0.0, 1.0)


if __name__ == "__main__":

    # ---------------- LOAD DATA ----------------
    csv_path = os.path.join(repo_root, CONFIG["csv_file"])
    df = pd.read_csv(csv_path)

    X_raw = df.drop("Outcome", axis=1).values
    y = torch.tensor(df["Outcome"].values, dtype=torch.float32).unsqueeze(1)

    # ---------------- NORMALIZE FEATURES ----------------
    scaler = StandardScaler()
    X_scaled = torch.tensor(scaler.fit_transform(X_raw), dtype=torch.float32)

    os.makedirs(os.path.dirname(CONFIG["scaler_path"]), exist_ok=True)
    joblib.dump(scaler, CONFIG["scaler_path"])

    # ---------------- DATALOADER ----------------
    dataset = TensorDataset(X_scaled, y)
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # ---------------- MODEL ----------------
    model = HospitalModel()

    # ---------------- OPTIMIZER ----------------
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    # ---------------- CLASS WEIGHTING ----------------
    n_pos = (y == 1).sum().item()
    n_neg = (y == 0).sum().item()

    pos_weight_val = min(n_neg / max(n_pos, 1), 2.0)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val])
    )

    # ---------------- ATTACH DP ENGINE ----------------
    privacy_engine = PrivacyEngine()

    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=CONFIG["noise_multiplier"],
        max_grad_norm=CONFIG["max_grad_norm"]
    )

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(CONFIG["epochs"]):

        model.train()

        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(logits, y_batch)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        mean_loss = epoch_loss / max(n_batches, 1)

        scheduler.step(mean_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {mean_loss:.4f}")

    # ---------------- SAVE MODEL ----------------
    os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)

    torch.save(model.state_dict(), CONFIG["save_path"])

    print(f"Saved DP weights to {CONFIG['save_path']}")

    # ---------------- PRINT PRIVACY SPENT ----------------
    try:
        epsilon = privacy_engine.accountant.get_epsilon(delta=CONFIG["delta"])
        print(f"(ε, δ)-DP: ({epsilon:.2f}, {CONFIG['delta']})")
    except Exception as e:
        print(f"Could not compute epsilon: {e}")

    # ---------------- TEST INFERENCE ----------------
    test_patients = [
        [8, 200, 90, 35, 200, 40.0, 1.5, 55],
        [1, 90, 70, 20, 50, 22.0, 0.2, 25]
    ]

    test_input = torch.tensor(test_patients, dtype=torch.float32)

    test_input_scaled = torch.tensor(
        scaler.transform(test_input),
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():

        logits = model(test_input_scaled)

        logits = torch.clamp(logits, -10, 10)

        probs = torch.sigmoid(logits)

        # -------- APPLY BANDAID NORMALIZATION --------
        normalized_probs = normalize_probability_bandaid(probs)

    # ---------------- PRINT RESULTS ----------------
    for i in range(len(probs)):

        raw = probs[i].item()

        fixed = normalized_probs[i].item()

        print(f"Patient {i+1}")
        print(f"  Raw probability: {raw:.4f}")
        print(f"  Normalized probability: {fixed:.4f}")