import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from hospital_client.backend.model import MergedModel

# ─── Constants ────────────────────────────────────────────────────────────────
FEATURES = ["Age", "BMI", "Glucose"]
LABEL = "Outcome"
IMPUTE_COLS = ["Glucose", "BMI"]
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in IMPUTE_COLS:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
    return df


def _unwrap(model):
    """Strip Opacus GradSampleModule wrapper if present."""
    return model._module if hasattr(model, "_module") else model


# Safety margin applied to the epsilon budget check during training.
# Training stops when epsilon_spent >= epsilon * (1 - EPSILON_SAFETY_MARGIN),
# leaving headroom so the final epoch's overshoot stays within the target.
# e.g. target=0.63, margin=0.10 → stop at 0.567 → final spend ≈ 0.63
EPSILON_SAFETY_MARGIN: float = 0.10


def _learn_temperature(model, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
    model.eval()
    temperature = nn.Parameter(torch.ones(1))
    opt = optim.LBFGS([temperature], lr=0.01, max_iter=100)
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        logits = model(X_val).detach()

    def step():
        opt.zero_grad()
        loss = criterion(logits / temperature, y_val)
        loss.backward()
        return loss

    opt.step(step)
    return max(float(temperature.item()), 1e-3)


def _quick_metrics(
    model, temperature: float, X_val: torch.Tensor, y_val: torch.Tensor
) -> dict:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_val) / temperature).squeeze().numpy()

    y = y_val.squeeze().numpy()
    preds = (probs >= 0.5).astype(float)

    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())

    acc = (tp + tn) / max(len(y), 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return dict(
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def _training_params_for_size(n_samples: int) -> dict:
    """
    Return epochs, batch_size, lr, patience (and size_tier label) based on dataset size.
    Small datasets: fewer epochs, smaller batches to avoid overfitting and reduce noise.
    Large datasets: more epochs, larger batches for efficiency.
    """
    if n_samples < 5_000:
        return {
            "epochs": 35,
            "batch_size": 64,
            "lr": 2e-3,
            "patience": 6,
            "size_tier": "small",
        }
    if n_samples < 25_000:
        return {
            "epochs": 50,
            "batch_size": 128,
            "lr": 1e-3,
            "patience": 8,
            "size_tier": "medium",
        }
    return {
        "epochs": 60,
        "batch_size": 256,
        "lr": 1e-3,
        "patience": 10,
        "size_tier": "large",
    }


# ─── Scaler bootstrap ────────────────────────────────────────────────────────
def build_scaler_from_merged(merged_csv_path: str) -> None:
    """
    Fit a StandardScaler on the merged dataset's training split and persist
    it to artifacts/.  Call this once if artifacts/scaler.pkl is missing.
    """
    df = pd.read_csv(merged_csv_path)
    df = _impute(df[FEATURES + [LABEL]])
    X = df[FEATURES].values.astype(np.float32)
    y = df[LABEL].values.astype(np.float32)

    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

    scale_params = {
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "columns": FEATURES,
    }
    with open(os.path.join(ARTIFACTS_DIR, "scale_params.json"), "w") as f:
        json.dump(scale_params, f, indent=2)

    print(f"Scaler saved → {ARTIFACTS_DIR}/scaler.pkl")


# ─── Core training function ───────────────────────────────────────────────────
def train(
    csv_path: str,
    epsilon: float = 5.0,
    use_dp: bool = True,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    delta: float = 1e-5,
) -> dict:
    """
    Train a MergedModel on the uploaded CSV.

    Parameters
    ----------
    csv_path    : path to hospital CSV (must contain Age, BMI, Glucose, Outcome)
    epsilon     : UI privacy slider value (0.1 = max privacy, 10.0 = max accuracy)
    use_dp      : whether to apply DP-SGD
    ...         : standard training hyper-parameters

    Returns
    -------
    dict with training metrics, final val loss, and (if DP) privacy spent
    """
    # ── Load & prep ───────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    n_samples = len(df)
    size_params = _training_params_for_size(n_samples)
    epochs = size_params["epochs"]
    batch_size = min(size_params["batch_size"], n_samples)  # cap for tiny datasets
    lr = size_params["lr"]
    patience = size_params["patience"]
    size_tier = size_params["size_tier"]

    df = _impute(df[FEATURES + [LABEL]])
    X_raw = df[FEATURES].values.astype(np.float32)
    y_all = df[LABEL].values.astype(np.float32)

    # Load shared scaler (fit on large merged dataset for consistent feature space)
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(
            "artifacts/scaler.pkl not found. Run build_scaler_from_merged() first."
        )
    scaler = joblib.load(scaler_path)

    X_scaled = scaler.transform(X_raw).astype(np.float32)

    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
        X_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # ── Class balance ─────────────────────────────────────────────────────────
    n_pos = int((y_train_np == 1).sum())
    n_neg = int((y_train_np == 0).sum())

    if use_dp and n_pos > 0 and n_neg > 0:
        # Undersample negatives 1:1 to avoid clipping asymmetry with DP
        rng = np.random.default_rng(42)
        pos_idx = np.where(y_train_np == 1)[0]
        neg_idx = rng.choice(
            np.where(y_train_np == 0)[0], size=len(pos_idx), replace=False
        )
        bal_idx = rng.permutation(np.concatenate([pos_idx, neg_idx]))
        X_train_np = X_train_np[bal_idx]
        y_train_np = y_train_np[bal_idx]
        pos_weight_val = 1.0
    else:
        pos_weight_val = min(n_neg / max(n_pos, 1), 5.0)

    X_train = torch.tensor(X_train_np)
    X_val = torch.tensor(X_val_np)
    y_train = torch.tensor(y_train_np).unsqueeze(1)
    y_val = torch.tensor(y_val_np).unsqueeze(1)

    # ── Model ─────────────────────────────────────────────────────────────────
    dropout = 0.0 if use_dp else 0.2
    model = MergedModel(n_features=len(FEATURES), dropout=dropout)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── Optional DP wrapping ──────────────────────────────────────────────────
    privacy_engine = None
    noise_multiplier = None

    if use_dp:
        privacy_engine = PrivacyEngine()
        # make_private_with_epsilon computes the exact noise multiplier so the
        # epsilon budget is consumed gradually and evenly across all planned
        # epochs — prevents exhausting the budget in epoch 1 on small datasets.
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=1.0,
            epochs=epochs,
        )
        noise_multiplier = optimizer.noise_multiplier

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    epsilon_budget_exhausted = False
    epochs_run = 0

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        epochs_run += 1

        # ── Epsilon budget check (DP only) ────────────────────────────────────
        # make_private_with_epsilon already calibrates noise so the budget is
        # consumed over all planned epochs. This check is a safety net only —
        # it fires if accumulation runs ahead of schedule (e.g. early stopping
        # cut the batch count short). The margin avoids a one-epoch overshoot.
        if use_dp and privacy_engine is not None:
            eps_so_far = privacy_engine.accountant.get_epsilon(delta=delta)
            if eps_so_far >= epsilon * (1 - EPSILON_SAFETY_MARGIN):
                epsilon_budget_exhausted = True
                break

        # Validate
        inf_model = _unwrap(model)
        inf_model.eval()
        with torch.no_grad():
            val_loss = criterion(inf_model(X_val), y_val).item()

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in inf_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        _unwrap(model).load_state_dict(best_state)

    # ── Temperature scaling ───────────────────────────────────────────────────
    inf_model = _unwrap(model)
    temperature = _learn_temperature(inf_model, X_val, y_val)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = _quick_metrics(inf_model, temperature, X_val, y_val)

    # ── Privacy report ────────────────────────────────────────────────────────
    epsilon_spent = None
    if use_dp and privacy_engine is not None:
        try:
            epsilon_spent = round(privacy_engine.accountant.get_epsilon(delta=delta), 4)
        except Exception:
            pass

    # ── Save updated artifacts ────────────────────────────────────────────────
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    weights_file = "dp_weights.pt" if use_dp else "weights.pt"
    temp_file = "dp_temperature.pt" if use_dp else "temperature.pt"

    torch.save(inf_model.state_dict(), os.path.join(ARTIFACTS_DIR, weights_file))
    torch.save(torch.tensor([temperature]), os.path.join(ARTIFACTS_DIR, temp_file))

    return {
        "use_dp": use_dp,
        "noise_multiplier": noise_multiplier,
        "epsilon_spent": epsilon_spent,
        "epsilon_budget_exhausted": epsilon_budget_exhausted,
        "epsilon_target": round(epsilon, 4),
        "epsilon_effective_budget": round(epsilon * (1 - EPSILON_SAFETY_MARGIN), 4),
        "temperature": round(temperature, 4),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "best_val_loss": round(best_val_loss, 4)
        if best_val_loss != float("inf")
        else None,
        "size_tier": size_tier,
        "epochs_run": epochs_run,
        "epochs_planned": epochs,
        "batch_size_used": batch_size,
        **metrics,
    }


# ─── Inference helper ─────────────────────────────────────────────────────────
def load_model_for_inference(use_dp: bool = False):
    """
    Load MergedModel + temperature from artifacts/.
    Returns (model, temperature, scaler).
    """
    weights_file = "dp_weights.pt"
    temp_file = "dp_temperature.pt"

    weights_path = os.path.join(ARTIFACTS_DIR, weights_file)
    temp_path = os.path.join(ARTIFACTS_DIR, temp_file)
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

    for p, name in [
        (weights_path, weights_file),
        (temp_path, temp_file),
        (scaler_path, "scaler.pkl"),
    ]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing artifact: {name}. Run training first.")

    model = MergedModel(n_features=len(FEATURES), dropout=0.0)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    temperature = torch.load(temp_path, weights_only=True).item()
    scaler = joblib.load(scaler_path)

    return model, temperature, scaler


def predict_single(
    age: float,
    bmi: float,
    glucose: float,
    use_dp: bool = False,
) -> dict:
    """Run inference on a single patient. Returns prediction and confidence."""
    model, temperature, scaler = load_model_for_inference(use_dp=use_dp)

    raw = np.array([[age, bmi, glucose]], dtype=np.float32)
    scaled = scaler.transform(raw).astype(np.float32)
    X = torch.tensor(scaled)

    with torch.no_grad():
        logit = model(X)
        prob = torch.sigmoid(logit / temperature).item()

    prediction = int(prob >= 0.5)
    return {
        "prediction": prediction,
        "confidence": round(prob, 4),
        "message": (
            "Elevated diabetes risk detected. Please consult a physician."
            if prediction == 1
            else "No significant diabetes risk detected."
        ),
    }


def predict_batch(df: pd.DataFrame, use_dp: bool = False) -> list[dict]:
    """Run inference on a DataFrame with Age, BMI, Glucose columns."""
    model, temperature, scaler = load_model_for_inference(use_dp=use_dp)

    df = _impute(df)
    raw = df[FEATURES].values.astype(np.float32)
    scaled = scaler.transform(raw).astype(np.float32)
    X = torch.tensor(scaled)

    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits / temperature).squeeze().numpy()

    if probs.ndim == 0:
        probs = probs.reshape(1)

    results = []
    for i, prob in enumerate(probs):
        prob = float(prob)
        pred = int(prob >= 0.5)
        results.append(
            {
                "row": i,
                "prediction": pred,
                "confidence": round(prob, 4),
                "message": (
                    "Elevated diabetes risk detected."
                    if pred == 1
                    else "No significant diabetes risk detected."
                ),
            }
        )
    return results
