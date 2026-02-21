import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from model.model import HospitalModel

# ---------------- CONFIG ----------------
CONFIG = {
    "csv_file": "hospital_client/hospital_csvs/hospital_A.csv",
    "scale_params_file": "hospital_client/scale_params.json",
    "batch_size": 64,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "noise_multiplier": 0.3,
    "max_grad_norm": 1.0,
    "delta": 1e-5,
    "dropout": 0.0,
    "val_size": 0.2,
    "early_stopping_patience": 15,
    "save_path": "model/private_weights.pt",
    "scaler_path": "model/scaler.pkl",
    "temperature_path": "model/temperature.pt",
}

# ---------------- PATH SETUP ----------------
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


# ---------------- TEMPERATURE SCALING ----------------
def learn_temperature(model, X_val, y_val):
    """
    Learn a single temperature parameter T on the validation set.
    T > 1 softens overconfident predictions; T < 1 sharpens underconfident ones.
    """
    model.eval()
    temperature = nn.Parameter(torch.ones(1))
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        logits = model(X_val)

    def eval_step():
        optimizer.zero_grad()
        scaled = logits / temperature
        loss = criterion(scaled, y_val)
        loss.backward()
        return loss

    optimizer.step(eval_step)
    return max(temperature.item(), 1e-3)


def scale_raw(raw, mean, std):
    """Apply (x - mean) / std to a numpy array of raw patient values."""
    return (np.array(raw, dtype=np.float32) - mean) / std


if __name__ == "__main__":
    # ---------------- LOAD SCALE PARAMS ----------------
    # scale_params.json is produced by split_data.py from the original raw
    # diabetes.csv and contains the true population mean/std. The hospital
    # CSVs have already been standardized with these values, so we do NOT
    # apply a second StandardScaler to the training data. We DO use these
    # params to normalize any raw patient values at inference time.
    scale_params_path = os.path.join(repo_root, CONFIG["scale_params_file"])
    with open(scale_params_path) as f:
        scale_params = json.load(f)

    scale_mean = np.array(scale_params["mean"], dtype=np.float32)
    scale_std = np.array(scale_params["std"], dtype=np.float32)
    feature_cols = scale_params["columns"]
    n_features = len(feature_cols)

    # Save scaler info so the backend API can normalize raw patient inputs
    os.makedirs(os.path.dirname(CONFIG["scaler_path"]), exist_ok=True)
    joblib.dump(
        {"mean": scale_mean, "std": scale_std, "columns": feature_cols},
        CONFIG["scaler_path"],
    )

    # ---------------- LOAD DATA ----------------
    # Hospital CSV is already cleaned (zeros imputed) and standardized by
    # split_data.py, so we load it directly without any further transformation.
    csv_path = os.path.join(repo_root, CONFIG["csv_file"])
    df = pd.read_csv(csv_path)

    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df["Outcome"].values.astype(np.float32)

    # ---------------- TRAIN / VAL SPLIT ----------------
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        X_all,
        y_all,
        test_size=CONFIG["val_size"],
        random_state=42,
        stratify=y_all,
    )

    X_train = torch.tensor(X_train_np)
    X_val = torch.tensor(X_val_np)
    y_train = torch.tensor(y_train_np).unsqueeze(1)
    y_val = torch.tensor(y_val_np).unsqueeze(1)

    print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples")
    print(f"Features ({n_features}): {feature_cols}")

    # ---------------- DATALOADER ----------------
    train_dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # ---------------- MODEL ----------------
    model = HospitalModel(dropout=CONFIG["dropout"], n_features=n_features)

    # ---------------- OPTIMIZER ----------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ---------------- CLASS WEIGHTING ----------------
    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    pos_weight_val = n_neg / max(n_pos, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
    print(f"Class weight  pos={pos_weight_val:.2f}  (neg={n_neg}, pos={n_pos})")

    # ---------------- ATTACH DP ENGINE ----------------
    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=CONFIG["noise_multiplier"],
        max_grad_norm=CONFIG["max_grad_norm"],
    )

    # ---------------- TRAIN LOOP ----------------
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(CONFIG["epochs"]):
        # --- train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        mean_train_loss = epoch_loss / max(n_batches, 1)

        # --- validate ---
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{CONFIG['epochs']} "
                f"- Train Loss: {mean_train_loss:.4f} "
                f"- Val Loss: {val_loss:.4f}"
            )

        # --- early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------------- TEMPERATURE SCALING ----------------
    # Use the unwrapped module for temperature learning and inference
    inference_model = model._module if hasattr(model, "_module") else model

    temperature = learn_temperature(inference_model, X_val, y_val)
    print(f"Learned temperature: {temperature:.4f}")
    torch.save(torch.tensor([temperature]), CONFIG["temperature_path"])

    # ---------------- SAVE MODEL ----------------
    # Unwrap Opacus GradSampleModule (_module prefix) before saving so weights
    # can be loaded into a plain HospitalModel without Opacus installed.
    os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)
    torch.save(inference_model.state_dict(), CONFIG["save_path"])
    print(f"Saved DP weights to {CONFIG['save_path']}")

    # ---------------- PRINT PRIVACY SPENT ----------------
    try:
        epsilon = privacy_engine.accountant.get_epsilon(delta=CONFIG["delta"])
        print(f"(ε, δ)-DP: ({epsilon:.2f}, {CONFIG['delta']})")
    except Exception as e:
        print(f"Could not compute epsilon: {e}")

    # ---------------- TEST INFERENCE ----------------
    # Raw patient values — scaled here using scale_params so the model sees
    # the same standardized distribution it was trained on.
    test_patients_raw = [
        [8, 200, 90, 35, 200, 40.0, 1.5, 55],  # high-risk
        [1, 90, 70, 20, 50, 22.0, 0.2, 25],  # low-risk
    ]

    test_input = torch.tensor(scale_raw(test_patients_raw, scale_mean, scale_std))

    inference_model.eval()
    with torch.no_grad():
        logits = inference_model(test_input)
        calibrated_probs = torch.sigmoid(logits / temperature)

    print("\n--- Test Inference ---")
    labels = ["High-risk", "Low-risk"]
    for i, label in enumerate(labels):
        prob = calibrated_probs[i].item()
        logit = logits[i].item()
        print(
            f"Patient {i + 1} ({label}): "
            f"logit={logit:.3f}  prob={prob:.4f}  "
            f"-> {'Diabetic' if prob >= 0.5 else 'Non-diabetic'}"
        )
