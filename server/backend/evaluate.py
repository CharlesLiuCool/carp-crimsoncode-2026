"""
Evaluate the central aggregated model against a held-out test set.

Metrics returned
----------------
accuracy   : (TP + TN) / total
precision  : TP / (TP + FP)
recall     : TP / (TP + FN)
f1         : 2 * precision * recall / (precision + recall)
tp, tn, fp, fn : raw confusion matrix counts
total      : number of test samples
"""

import os

import numpy as np
import pandas as pd
import torch
from aggregate import load_central_model

BACKEND_DIR = os.path.dirname(__file__)
TEST_DATA_PATH = os.path.join(BACKEND_DIR, "test_data.csv")

# Must match the scaler used during hospital client training
SCALE_MEAN = np.array(
    [42.27461331040037, 27.344150779168032, 137.77024124847878], dtype=np.float32
)
SCALE_STD = np.array(
    [22.51730211875095, 6.608171633655254, 40.57052832219186], dtype=np.float32
)


def _load_test_data() -> tuple[np.ndarray, np.ndarray]:
    """Load and standardise test_data.csv. Returns (X, y) as numpy arrays."""
    if not os.path.isfile(TEST_DATA_PATH):
        raise FileNotFoundError(
            "test_data.csv not found in server/backend/. "
            "Re-run the test set generation script."
        )

    df = pd.read_csv(TEST_DATA_PATH)

    missing = [c for c in ["Age", "BMI", "Glucose", "Outcome"] if c not in df.columns]
    if missing:
        raise ValueError(f"test_data.csv is missing columns: {missing}")

    raw = df[["Age", "BMI", "Glucose"]].values.astype(np.float32)
    X = (raw - SCALE_MEAN) / SCALE_STD
    y = df["Outcome"].values.astype(np.float32)

    return X, y


def evaluate() -> dict:
    """
    Run the central model on test_data.csv and return evaluation metrics.

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1  (all 0–1 floats, 4 d.p.)
        tp, tn, fp, fn                   (int counts)
        total                            (int)
    """
    model = load_central_model()

    X_np, y_np = _load_test_data()
    X = torch.tensor(X_np)

    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).squeeze().numpy()

    preds = (probs >= 0.5).astype(np.float32)

    tp = int(((preds == 1) & (y_np == 1)).sum())
    tn = int(((preds == 0) & (y_np == 0)).sum())
    fp = int(((preds == 1) & (y_np == 0)).sum())
    fn = int(((preds == 0) & (y_np == 1)).sum())
    total = len(y_np)

    accuracy = round((tp + tn) / total, 4)
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }
