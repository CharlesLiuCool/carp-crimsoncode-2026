import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ─── Config ──────────────────────────────────────────────────────────────────
CONFIG = {
    "csv_file": "hospital_client/merged_diabetes.csv",
    "features": ["Age", "BMI", "Glucose"],
    "label": "Outcome",
    "val_size": 0.2,
    # Standard (non-DP) training
    "std_epochs": 60,
    "std_lr": 1e-3,
    "std_weight_decay": 1e-4,
    "std_batch_size": 256,
    "std_dropout": 0.2,
    "std_patience": 10,
    "std_pos_weight_cap": 5.0,
    "std_weights_path": "model/merged_weights.pt",
    # DP training — uses undersampled balanced data so no pos_weight needed
    "dp_epochs": 60,
    "dp_lr": 1e-3,
    "dp_weight_decay": 1e-4,
    "dp_batch_size": 256,
    "dp_dropout": 0.0,
    "dp_noise_multiplier": 0.5,
    "dp_max_grad_norm": 1.0,
    "dp_delta": 1e-5,
    "dp_patience": 12,
    "dp_weights_path": "model/merged_dp_weights.pt",
    # Shared
    "scaler_path": "model/merged_scaler.pkl",
    "scale_params_path": "model/merged_scale_params.json",
    "temperature_path": "model/merged_temperature.pt",
    "dp_temperature_path": "model/merged_dp_temperature.pt",
}

SEP = "─" * 56


# ─── Lightweight 3-feature model ─────────────────────────────────────────────
class MergedModel(nn.Module):
    """
    3 → 16 → 8 → 1  (~209 parameters)
    Intentionally small — 3 features don't need large capacity.
    """

    def __init__(self, n_features=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def impute(df, cols):
    """Replace zeros / NaN with column median (computed on df itself)."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
    return df


def learn_temperature(model, X_val, y_val):
    """Single-parameter temperature scaling on the validation set."""
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


def train_loop(model, loader, optimizer, criterion):
    model.train()
    total_loss, n = 0.0, 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def val_loss_fn(model, X_val, y_val, criterion):
    model.eval()
    with torch.no_grad():
        return criterion(model(X_val), y_val).item()


def run_early_stopping(
    model,
    loader,
    X_val,
    y_val,
    optimizer,
    criterion,
    scheduler,
    epochs,
    patience,
    label,
):
    best_val, counter, best_state = float("inf"), 0, None

    for epoch in range(epochs):
        train_loss = train_loop(model, loader, optimizer, criterion)
        vl = val_loss_fn(model, X_val, y_val, criterion)
        scheduler.step(vl)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  [{label}] Epoch {epoch + 1:>3}/{epochs} "
                f"train={train_loss:.4f}  val={vl:.4f}"
            )

        if vl < best_val:
            best_val, counter = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= patience:
                print(f"  [{label}] Early stopping at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def unwrap(model):
    """Strip Opacus GradSampleModule wrapper if present."""
    return model._module if hasattr(model, "_module") else model


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(unwrap(model).state_dict(), path)
    print(f"  Saved weights → {path}")


# ─── Data preparation ─────────────────────────────────────────────────────────
def prepare_data():
    csv_path = os.path.join(repo_root, CONFIG["csv_file"])
    df = pd.read_csv(csv_path)

    # Drop the source tag
    df = df[CONFIG["features"] + [CONFIG["label"]]].copy()

    # Impute biologically impossible zeros
    df = impute(df, CONFIG["features"])

    X = df[CONFIG["features"]].values.astype(np.float32)
    y = df[CONFIG["label"]].values.astype(np.float32)

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=CONFIG["val_size"], random_state=42, stratify=y
    )

    # Fit scaler on training set only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)

    # Persist scaler and scale params for inference
    os.makedirs(os.path.dirname(CONFIG["scaler_path"]), exist_ok=True)
    joblib.dump(scaler, CONFIG["scaler_path"])

    scale_params = {
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "columns": CONFIG["features"],
    }
    with open(os.path.join(repo_root, CONFIG["scale_params_path"]), "w") as f:
        json.dump(scale_params, f, indent=2)

    X_train_t = torch.tensor(X_train)
    X_val_t = torch.tensor(X_val)
    y_train_t = torch.tensor(y_train).unsqueeze(1)
    y_val_t = torch.tensor(y_val).unsqueeze(1)

    n_pos = int((y_train_t == 1).sum())
    n_neg = int((y_train_t == 0).sum())
    pos_weight = min(n_neg / max(n_pos, 1), CONFIG["std_pos_weight_cap"])

    print(f"  Train: {len(X_train_t):,}  Val: {len(X_val_t):,}")
    print(
        f"  Positive: {n_pos:,} ({100 * n_pos / (n_pos + n_neg):.1f}%)  "
        f"Negative: {n_neg:,}"
    )
    print(
        f"  pos_weight (STD): {pos_weight:.2f}  (capped at {CONFIG['std_pos_weight_cap']})"
    )
    print(f"  Features: {CONFIG['features']}")

    # ── Balanced subset for DP training (undersample negatives 1:1) ──────────
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    rng = np.random.default_rng(42)
    neg_sample = rng.choice(neg_idx, size=len(pos_idx), replace=False)
    bal_idx = np.concatenate([pos_idx, neg_sample])
    rng.shuffle(bal_idx)
    X_train_dp = torch.tensor(X_train[bal_idx])
    y_train_dp = torch.tensor(y_train[bal_idx]).unsqueeze(1)
    print(
        f"  DP balanced train: {len(X_train_dp):,}  "
        f"(pos={len(pos_idx):,}  neg={len(neg_sample):,})"
    )

    return (
        X_train_t,
        X_val_t,
        y_train_t,
        y_val_t,
        pos_weight,
        X_train_dp,
        y_train_dp,
        scaler,
    )


# ─── Standard training ────────────────────────────────────────────────────────
def train_standard(X_train, X_val, y_train, y_val, pos_weight):  # noqa: E501
    print(f"\n{SEP}")
    print("  Standard (non-DP) Training")
    print(SEP)

    n_features = X_train.shape[1]
    model = MergedModel(n_features=n_features, dropout=CONFIG["std_dropout"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {n_features} → 16 → 8 → 1  ({total_params} params)")

    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["std_lr"],
        weight_decay=CONFIG["std_weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=CONFIG["std_batch_size"], shuffle=True)

    model = run_early_stopping(
        model,
        loader,
        X_val,
        y_val,
        optimizer,
        criterion,
        scheduler,
        CONFIG["std_epochs"],
        CONFIG["std_patience"],
        label="STD",
    )

    temperature = learn_temperature(model, X_val, y_val)
    torch.save(
        torch.tensor([temperature]), os.path.join(repo_root, CONFIG["temperature_path"])
    )
    print(f"  Temperature: {temperature:.4f}")

    save_model(model, os.path.join(repo_root, CONFIG["std_weights_path"]))
    return model, temperature


# ─── DP training ─────────────────────────────────────────────────────────────
def train_dp(X_train_dp, X_val, y_train_dp, y_val):
    """
    Train with DP-SGD on a 1:1 balanced subset (undersampled negatives).
    Balancing avoids the clipping asymmetry caused by pos_weight with DP:
    large positive-class gradients are clipped more aggressively than
    negative-class gradients, gutting the minority-class signal.
    With balanced classes no pos_weight is needed and clipping is symmetric.
    """
    print(f"\n{SEP}")
    print("  Differentially Private (DP-SGD) Training")
    print(SEP)

    n_features = X_train_dp.shape[1]
    model = MergedModel(n_features=n_features, dropout=CONFIG["dp_dropout"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {n_features} → 16 → 8 → 1  ({total_params} params)")
    print(
        f"  noise_multiplier={CONFIG['dp_noise_multiplier']}  "
        f"max_grad_norm={CONFIG['dp_max_grad_norm']}  "
        f"batch={CONFIG['dp_batch_size']}"
    )
    print(f"  Balanced train size: {len(X_train_dp):,}  (no pos_weight needed)")

    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["dp_lr"],
        weight_decay=CONFIG["dp_weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.BCEWithLogitsLoss()  # balanced data — no pos_weight

    dataset = TensorDataset(X_train_dp, y_train_dp)
    loader = DataLoader(dataset, batch_size=CONFIG["dp_batch_size"], shuffle=True)

    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=CONFIG["dp_noise_multiplier"],
        max_grad_norm=CONFIG["dp_max_grad_norm"],
    )

    best_val, counter, best_state = float("inf"), 0, None

    for epoch in range(CONFIG["dp_epochs"]):
        train_loss = train_loop(model, loader, optimizer, criterion)
        vl = val_loss_fn(unwrap(model), X_val, y_val, criterion)
        scheduler.step(vl)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            try:
                eps = privacy_engine.accountant.get_epsilon(delta=CONFIG["dp_delta"])
                eps_str = f"  ε={eps:.2f}"
            except Exception:
                eps_str = ""
            print(
                f"  [DP] Epoch {epoch + 1:>3}/{CONFIG['dp_epochs']} "
                f"train={train_loss:.4f}  val={vl:.4f}{eps_str}"
            )

        if vl < best_val:
            best_val, counter = vl, 0
            best_state = {k: v.clone() for k, v in unwrap(model).state_dict().items()}
        else:
            counter += 1
            if counter >= CONFIG["dp_patience"]:
                print(f"  [DP] Early stopping at epoch {epoch + 1}")
                break

    if best_state:
        unwrap(model).load_state_dict(best_state)

    try:
        epsilon = privacy_engine.accountant.get_epsilon(delta=CONFIG["dp_delta"])
        print(f"\n  Final (ε, δ)-DP guarantee: ({epsilon:.2f}, {CONFIG['dp_delta']})")
    except Exception as e:
        print(f"  Could not compute epsilon: {e}")

    inf_model = unwrap(model)
    temperature = learn_temperature(inf_model, X_val, y_val)
    torch.save(
        torch.tensor([temperature]),
        os.path.join(repo_root, CONFIG["dp_temperature_path"]),
    )
    print(f"  Temperature: {temperature:.4f}")

    save_model(model, os.path.join(repo_root, CONFIG["dp_weights_path"]))
    return inf_model, temperature


# ─── Quick eval ──────────────────────────────────────────────────────────────
def quick_eval(model, temperature, X_val, y_val, label):
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        probs = torch.sigmoid(logits / temperature).squeeze().numpy()

    y = y_val.squeeze().numpy()
    preds = (probs >= 0.5).astype(float)

    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())

    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"\n  [{label}] Val metrics @ threshold=0.50")
    print(f"    Accuracy  : {acc * 100:.1f}%")
    print(f"    Precision : {prec * 100:.1f}%")
    print(f"    Recall    : {rec * 100:.1f}%")
    print(f"    F1        : {f1 * 100:.1f}%")
    print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'═' * 56}")
    print("  DiabetesAI — Training on merged_diabetes.csv")
    print(f"{'═' * 56}")

    X_train, X_val, y_train, y_val, pos_weight, X_train_dp, y_train_dp, scaler = (
        prepare_data()
    )

    # Standard model
    std_model, std_temp = train_standard(X_train, X_val, y_train, y_val, pos_weight)
    quick_eval(std_model, std_temp, X_val, y_val, "STD")

    # DP model (balanced undersampled training set)
    dp_model, dp_temp = train_dp(X_train_dp, X_val, y_train_dp, y_val)
    quick_eval(dp_model, dp_temp, X_val, y_val, "DP")

    print(f"\n{'═' * 56}")
    print("  Artifacts saved:")
    for key in (
        "std_weights_path",
        "dp_weights_path",
        "scaler_path",
        "scale_params_path",
        "temperature_path",
        "dp_temperature_path",
    ):
        print(f"    {CONFIG[key]}")
    print(f"{'═' * 56}\n")
