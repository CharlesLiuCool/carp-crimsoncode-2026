"""
Federated averaging (FedAvg) for the central CARP server model.

How it works
------------
1. Load every valid .pt row from the uploaded_weights PostgreSQL table.
2. Group uploads by round_id (assigned by the KeyPool at upload time).
3. Only include rounds where all ROUND_SIZE (3) slots are present —
   partial rounds are skipped until the missing upload arrives.
4. Average all state-dicts from complete rounds element-wise.
5. Save the result to artifacts/central_model.pt.

Each upload has already had its pairwise mask applied by the KeyPool
before being stored, so the masks cancel exactly across each complete
round and FedAvg produces the correct unmasked average.

Override ROUND_SIZE via the MIN_CONTRIBUTORS environment variable.
"""

import io
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from db import fetch_all_valid_weights
from key_pool import ROUND_SIZE
from model import MergedModel
from torch import optim

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
CENTRAL_WEIGHTS = os.path.join(ARTIFACTS_DIR, "central_model.pt")
CENTRAL_TEMPERATURE = os.path.join(ARTIFACTS_DIR, "central_temperature.pt")

BACKEND_DIR = os.path.dirname(__file__)
TEST_DATA_PATH = os.path.join(BACKEND_DIR, "test_data.csv")

# Must match evaluate.py / main.py
SCALE_MEAN = np.array(
    [42.27461331040037, 27.344150779168032, 137.77024124847878], dtype=np.float32
)
SCALE_STD = np.array(
    [22.51730211875095, 6.608171633655254, 40.57052832219186], dtype=np.float32
)

# Kept for backward-compatibility with weights.py import; value mirrors ROUND_SIZE.
MIN_CONTRIBUTORS: int = ROUND_SIZE


# ── Temperature calibration ───────────────────────────────────────────────────


def _learn_temperature(model: MergedModel) -> float:
    """
    Fit a scalar temperature on the held-out test set via LBFGS so that
    sigmoid(logit / T) produces well-calibrated probabilities.

    Mirrors the same routine used by the hospital client after local training.
    Falls back to T=1.0 if test_data.csv is unavailable.
    """
    if not os.path.isfile(TEST_DATA_PATH):
        logger.warning(
            "test_data.csv not found — skipping temperature calibration, T=1.0"
        )
        return 1.0

    try:
        df = pd.read_csv(TEST_DATA_PATH)
        raw = df[["Age", "BMI", "Glucose"]].values.astype(np.float32)
        X = torch.tensor((raw - SCALE_MEAN) / SCALE_STD, dtype=torch.float32)
        y = torch.tensor(df["Outcome"].values.astype(np.float32)).unsqueeze(1)
    except Exception as exc:
        logger.warning("Temperature calibration skipped (data load error): %s", exc)
        return 1.0

    model.eval()
    temperature = nn.Parameter(torch.ones(1))
    opt = optim.LBFGS([temperature], lr=0.01, max_iter=200)
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        logits = model(X).detach()

    def step():
        opt.zero_grad()
        loss = criterion(logits / temperature, y)
        loss.backward()
        return loss

    opt.step(step)
    T = max(float(temperature.item()), 1e-3)
    logger.info("Temperature calibration complete: T=%.4f", T)
    return T


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_state_from_bytes(data: bytes, label: str) -> dict | None:
    """
    Try to load a state-dict from raw bytes.
    Returns None and logs a warning if the data is unreadable or has the
    wrong keys (e.g. a full checkpoint rather than a bare state-dict).
    """
    try:
        buf = io.BytesIO(data)
        obj = torch.load(buf, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.warning("Could not load %s: %s", label, exc)
        return None

    # Accept a bare state-dict or a dict that contains one under "state_dict"
    if isinstance(obj, dict):
        if "state_dict" in obj:
            obj = obj["state_dict"]
        # Validate that it matches MergedModel
        dummy = MergedModel()
        expected = set(dummy.state_dict().keys())
        if set(obj.keys()) != expected:
            logger.warning(
                "Skipping %s — keys don't match MergedModel (got %s, want %s)",
                label,
                set(obj.keys()),
                expected,
            )
            return None
        return obj

    logger.warning("Skipping %s — not a state-dict", label)
    return None


# ── Core ──────────────────────────────────────────────────────────────────────


def aggregate() -> dict:
    """
    Run FedAvg over all complete rounds in the database.

    A round is complete when all ROUND_SIZE slots have been uploaded.
    Partial rounds (missing one or more slots) are skipped until the
    remaining uploads arrive.  Uploads without a round_id (legacy rows)
    are ignored.

    Raises ValueError if no complete rounds exist yet — the central model
    is not written in that case so the previous valid model is preserved.

    Returns a summary dict:
        {
            "aggregated":       int,   # state-dicts included in FedAvg
            "skipped":          int,   # rows that failed validation
            "complete_rounds":  int,   # number of complete rounds used
            "pending_rounds":   int,   # partial rounds waiting for uploads
            "central_model":    str    # path to saved central_model.pt
        }
    """
    rows = fetch_all_valid_weights()

    # ── Group by round_id ─────────────────────────────────────────────────────
    rounds: dict[int, list[dict]] = {}
    for row in rows:
        rid = row.get("round_id")
        if rid is None:
            continue  # skip legacy uploads without a round assignment
        rounds.setdefault(rid, []).append(row)

    complete = {rid: r for rid, r in rounds.items() if len(r) == ROUND_SIZE}
    pending = {rid: r for rid, r in rounds.items() if len(r) < ROUND_SIZE}

    if not complete:
        pending_count = len(pending)
        raise ValueError(
            f"No complete rounds found (need {ROUND_SIZE} uploads per round). "
            f"{pending_count} partial round(s) are waiting for more uploads."
        )

    # ── Load state-dicts from all complete rounds ─────────────────────────────
    state_dicts = []
    skipped = 0

    for rid in sorted(complete.keys()):
        for row in sorted(complete[rid], key=lambda r: r["slot"] or 0):
            sd = _load_state_from_bytes(row["weights_bytes"], row["saved_as"])
            if sd is None:
                skipped += 1
            else:
                state_dicts.append(sd)

    if not state_dicts:
        raise ValueError(
            "No valid weight files could be loaded from complete rounds. "
            "Ensure hospitals export bare MergedModel state-dicts."
        )

    # ── FedAvg: element-wise mean across all state-dicts ─────────────────────
    avg_state = {}
    n = len(state_dicts)

    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        avg_state[key] = stacked.mean(dim=0)

    # ── Temperature calibration ───────────────────────────────────────────────
    calibrated_model = MergedModel(dropout=0.0)
    calibrated_model.load_state_dict(avg_state)
    calibrated_model.eval()
    temperature = _learn_temperature(calibrated_model)

    # ── Persist ──────────────────────────────────────────────────────────────
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    torch.save(avg_state, CENTRAL_WEIGHTS)
    torch.save(torch.tensor([temperature]), CENTRAL_TEMPERATURE)

    logger.info(
        "FedAvg complete: %d complete round(s), %d file(s) aggregated, "
        "%d skipped, %d partial round(s) pending → %s",
        len(complete),
        n,
        skipped,
        len(pending),
        CENTRAL_WEIGHTS,
    )

    return {
        "aggregated": n,
        "skipped": skipped,
        "complete_rounds": len(complete),
        "pending_rounds": len(pending),
        "central_model": CENTRAL_WEIGHTS,
        "temperature": round(temperature, 4),
    }


# ── Inference helper ──────────────────────────────────────────────────────────


def load_central_model() -> MergedModel:
    """
    Load the central aggregated model.
    Raises FileNotFoundError if the model has not been built yet, which
    happens when fewer than MIN_CONTRIBUTORS valid uploads exist.
    """
    if not os.path.isfile(CENTRAL_WEIGHTS):
        raise FileNotFoundError(
            f"Central model not available. At least one complete round of "
            f"{ROUND_SIZE} uploads is required before the model is built."
        )

    model = MergedModel(dropout=0.0)
    model.load_state_dict(
        torch.load(CENTRAL_WEIGHTS, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def load_central_temperature() -> float:
    """
    Load the calibrated temperature scalar for the central model.
    Returns 1.0 (no-op) if the temperature file has not been saved yet.
    """
    if not os.path.isfile(CENTRAL_TEMPERATURE):
        logger.warning("central_temperature.pt not found — using T=1.0 (uncalibrated)")
        return 1.0
    T = torch.load(CENTRAL_TEMPERATURE, map_location="cpu", weights_only=True)
    return max(float(T.item()), 1e-3)
