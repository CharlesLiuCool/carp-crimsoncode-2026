import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import joblib
import numpy as np
import torch

from model.model import HospitalModel

# ─── Paths ───────────────────────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(repo_root, "model/private_weights.pt")
TEMPERATURE_PATH = os.path.join(repo_root, "model/temperature.pt")
SCALER_PATH = os.path.join(repo_root, "model/scaler.pkl")
TEST_CSV = os.path.join(repo_root, "hospital_client/hospital_csvs/hospital_B.csv")

# ─── Guards ──────────────────────────────────────────────────────────────────
for path, name in [
    (WEIGHTS_PATH, "model/private_weights.pt  (run train_dp.py first)"),
    (TEMPERATURE_PATH, "model/temperature.pt       (run train_dp.py first)"),
    (SCALER_PATH, "model/scaler.pkl           (run train_dp.py first)"),
    (TEST_CSV, "hospital_client/hospital_csvs/hospital_B.csv"),
]:
    if not os.path.isfile(path):
        print(f"Missing: {name}")
        sys.exit(1)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_model():
    scaler_info = joblib.load(SCALER_PATH)
    n_features = len(scaler_info["columns"])

    model = HospitalModel(n_features=n_features, dropout=0.0)
    model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
    model.eval()

    temperature = torch.load(TEMPERATURE_PATH, weights_only=True).item()
    return model, temperature, scaler_info


def load_test_data(scaler_info):
    import pandas as pd

    df = pd.read_csv(TEST_CSV)
    feature_cols = scaler_info["columns"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Outcome"].values.astype(np.float32)
    return torch.tensor(X), torch.tensor(y)


def get_probs(model, temperature, X):
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits / temperature).squeeze().numpy()
    return probs


def confusion_matrix_values(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn


def compute_metrics(y_true, probs, threshold=0.5):
    y_pred = (probs >= threshold).astype(np.float32)
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return dict(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def auc_roc(y_true, probs):
    """Trapezoidal AUC-ROC computed without sklearn."""
    thresholds = np.sort(np.unique(probs))[::-1]
    tprs, fprs = [0.0], [0.0]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    for t in thresholds:
        y_pred = (probs >= t).astype(np.float32)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tprs.append(tp / n_pos if n_pos > 0 else 0.0)
        fprs.append(fp / n_neg if n_neg > 0 else 0.0)

    tprs.append(1.0)
    fprs.append(1.0)

    # Trapezoidal integration
    auc = float(np.trapezoid(tprs, fprs))
    return abs(auc), np.array(fprs), np.array(tprs)


def find_optimal_threshold(y_true, probs):
    """Return the threshold that maximises the F1 score."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.1, 0.91, 0.01):
        m = compute_metrics(y_true, probs, threshold=t)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], t
    return round(best_t, 2), best_f1


# ─── Formatting ──────────────────────────────────────────────────────────────
SEP = "─" * 52
SEP_S = "·" * 52
W = 28  # label column width


def bar(value, width=20):
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def pct(v):
    return f"{v * 100:.1f}%"


def print_section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_confusion_matrix(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    print()
    print("                  Predicted")
    print("                  Pos    Neg")
    print(f"  Actual  Pos  [ {tp:>4}  {fn:>4} ]")
    print(f"          Neg  [ {fp:>4}  {tn:>4} ]")
    print()
    print(f"  Total samples : {total}")
    print(f"  True Positives: {tp}  (correctly flagged as diabetic)")
    print(f"  True Negatives: {tn}  (correctly cleared)")
    print(f"  False Positives:{fp}  (healthy flagged as diabetic)")
    print(f"  False Negatives:{fn}  (diabetic missed)")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'═' * 52}")
    print("  DiabetesAI — Model Accuracy Test")
    print(f"  Test set : hospital_B.csv")
    print(f"{'═' * 52}")

    # Load
    model, temperature, scaler_info = load_model()
    X, y = load_test_data(scaler_info)
    y_np = y.numpy()

    print(f"\n  Samples      : {len(y_np)}")
    print(f"  Positive (1) : {int(y_np.sum())} ({pct(y_np.mean())})")
    print(f"  Negative (0) : {int((y_np == 0).sum())} ({pct(1 - y_np.mean())})")
    print(f"  Temperature  : {temperature:.4f}")

    # Probabilities
    probs = get_probs(model, temperature, X)

    # ── Default threshold (0.5) ───────────────────────────────────────────────
    print_section("Metrics at threshold = 0.50")
    m = compute_metrics(y_np, probs, threshold=0.5)

    metrics = [
        ("Accuracy", m["accuracy"]),
        ("Precision", m["precision"]),
        ("Recall", m["recall"]),
        ("Specificity", m["specificity"]),
        ("F1 Score", m["f1"]),
    ]
    for label, val in metrics:
        print(f"  {label:<{W}} {pct(val):>6}  {bar(val)}")

    print_confusion_matrix(m["tp"], m["tn"], m["fp"], m["fn"])

    # ── AUC-ROC ───────────────────────────────────────────────────────────────
    print_section("AUC-ROC")
    auc, fprs, tprs = auc_roc(y_np, probs)
    print(f"\n  AUC-ROC : {auc:.4f}  {bar(auc)}")
    print()

    # Sparse ROC curve (10 points)
    print(f"  {'FPR':>6}  {'TPR':>6}")
    print(f"  {SEP_S[:16]}")
    indices = np.linspace(0, len(fprs) - 1, 10, dtype=int)
    for i in indices:
        print(f"  {fprs[i]:>6.3f}  {tprs[i]:>6.3f}")

    # ── Optimal threshold ─────────────────────────────────────────────────────
    print_section("Optimal Threshold (max F1)")
    opt_t, opt_f1 = find_optimal_threshold(y_np, probs)
    opt_m = compute_metrics(y_np, probs, threshold=opt_t)

    print(f"\n  Optimal threshold : {opt_t}")
    print()
    opt_metrics = [
        ("Accuracy", opt_m["accuracy"]),
        ("Precision", opt_m["precision"]),
        ("Recall", opt_m["recall"]),
        ("Specificity", opt_m["specificity"]),
        ("F1 Score", opt_m["f1"]),
    ]
    for label, val in opt_metrics:
        print(f"  {label:<{W}} {pct(val):>6}  {bar(val)}")

    print_confusion_matrix(opt_m["tp"], opt_m["tn"], opt_m["fp"], opt_m["fn"])

    # ── Probability distribution ──────────────────────────────────────────────
    print_section("Probability Distribution")
    bins = np.arange(0, 1.1, 0.1)
    pos_probs = probs[y_np == 1]
    neg_probs = probs[y_np == 0]

    print(f"\n  {'Bin':<12} {'Diabetic':>10} {'Healthy':>10}")
    print(f"  {SEP_S[:36]}")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        n_pos = int(((pos_probs >= lo) & (pos_probs < hi)).sum())
        n_neg = int(((neg_probs >= lo) & (neg_probs < hi)).sum())
        label = f"{lo:.1f} – {hi:.1f}"
        print(f"  {label:<12} {n_pos:>10} {n_neg:>10}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 52}")
    print("  Summary")
    print(f"{'═' * 52}")
    print(f"  AUC-ROC            : {auc:.4f}")
    print(f"  F1 @ 0.50          : {pct(m['f1'])}")
    print(f"  F1 @ {opt_t} (optimal) : {pct(opt_f1)}")
    print(f"  Accuracy @ 0.50    : {pct(m['accuracy'])}")
    print(f"  Accuracy @ {opt_t}      : {pct(opt_m['accuracy'])}")
    print(f"{'═' * 52}\n")


if __name__ == "__main__":
    main()
