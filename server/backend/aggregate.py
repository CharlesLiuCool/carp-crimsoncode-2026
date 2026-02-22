"""
Federated averaging (FedAvg) for the central CARP server model.

How it works
------------
1. Load every valid .pt row from the uploaded_weights PostgreSQL table.
2. Average all state-dicts element-wise (equal weighting).
3. Save the result to artifacts/central_model.pt.
4. A temperature of 1.0 is used — no calibration data is available
   server-side, so raw sigmoid probabilities are returned as-is.

Call aggregate() after every successful weight upload so the central
model always reflects the latest set of hospital contributions.
"""

import io
import logging
import os

import torch
from db import fetch_all_valid_weights
from model import MergedModel

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
CENTRAL_WEIGHTS = os.path.join(ARTIFACTS_DIR, "central_model.pt")

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
    Run FedAvg over every valid weights row in the database.

    Returns a summary dict:
        {
            "aggregated": int,   # number of weight files included
            "skipped":    int,   # rows that failed validation
            "central_model": str # path to saved central_model.pt
        }
    """
    rows = fetch_all_valid_weights()

    if not rows:
        raise ValueError("No weight files found in the database.")

    state_dicts = []
    skipped = 0

    for row in rows:
        sd = _load_state_from_bytes(row["weights_bytes"], row["saved_as"])
        if sd is None:
            skipped += 1
        else:
            state_dicts.append(sd)

    if not state_dicts:
        raise ValueError(
            "No valid weight files could be loaded. "
            "Ensure hospitals export bare MergedModel state-dicts."
        )

    # ── FedAvg: element-wise mean across all state-dicts ─────────────────────
    avg_state = {}
    n = len(state_dicts)

    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        avg_state[key] = stacked.mean(dim=0)

    # ── Persist ──────────────────────────────────────────────────────────────
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    torch.save(avg_state, CENTRAL_WEIGHTS)

    logger.info(
        "FedAvg complete: %d file(s) aggregated, %d skipped → %s",
        n,
        skipped,
        CENTRAL_WEIGHTS,
    )

    return {
        "aggregated": n,
        "skipped": skipped,
        "central_model": CENTRAL_WEIGHTS,
    }


# ── Inference helper ──────────────────────────────────────────────────────────


def load_central_model() -> MergedModel:
    """
    Load the central aggregated model.
    Raises FileNotFoundError if no weights have been aggregated yet.
    """
    if not os.path.isfile(CENTRAL_WEIGHTS):
        raise FileNotFoundError(
            "Central model not found. Upload at least one weight file to generate it."
        )

    model = MergedModel(dropout=0.0)
    model.load_state_dict(
        torch.load(CENTRAL_WEIGHTS, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model
