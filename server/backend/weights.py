import hashlib
import io
import os
from datetime import datetime, timezone

import torch
from aggregate import MIN_CONTRIBUTORS, aggregate
from db import insert_weights
from db import list_weights as db_list_weights
from evaluate import evaluate
from flask import Blueprint, current_app, jsonify, request
from key_pool import get_key_pool
from werkzeug.utils import secure_filename

weights_bp = Blueprint("weights", __name__)

ALLOWED_EXTENSIONS = {".pt", ".pth", ".pkl"}


def _allowed(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    for i in range(0, len(data), 65536):
        h.update(data[i : i + 65536])
    return h.hexdigest()


# ── POST /api/weights/upload ─────────────────────────────────────────────────


@weights_bp.route("/upload", methods=["POST"])
def upload_weights():
    if "file" not in request.files:
        return jsonify({"detail": "No file part in the request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"detail": "No file selected."}), 400

    if not _allowed(file.filename):
        ext = os.path.splitext(file.filename)[1].lower()
        return jsonify(
            {"detail": f'Unsupported file type "{ext}". Allowed: .pt, .pth, .pkl'}
        ), 415

    filename = secure_filename(file.filename)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    saved_as = f"{timestamp}_{filename}"

    # Read bytes into memory
    data = file.read()
    size_kb = round(len(data) / 1024, 2)
    # Checksum is over the original bytes for duplicate detection
    checksum = _sha256_bytes(data)

    # ── Load state-dict and apply pre-computed round key mask ─────────────────
    try:
        state_dict = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            return jsonify({"detail": "Uploaded file is not a valid state-dict."}), 400
    except Exception as exc:
        return jsonify({"detail": f"Could not parse weights file: {exc}"}), 400

    key = get_key_pool().issue_key()
    masked_state = get_key_pool().apply_key(state_dict, key)

    buf = io.BytesIO()
    torch.save(masked_state, buf)
    masked_data = buf.getvalue()

    current_app.logger.info(
        "Mask applied: round=%d slot=%d  original=%.1f KB  masked=%.1f KB",
        key["round_id"],
        key["slot"],
        size_kb,
        round(len(masked_data) / 1024, 2),
    )

    # ── Persist masked weights to database ────────────────────────────────────
    try:
        row_id = insert_weights(
            filename=filename,
            saved_as=saved_as,
            sha256=checksum,
            size_kb=size_kb,
            data=masked_data,
            round_id=key["round_id"],
            slot=key["slot"],
        )
        current_app.logger.info(
            "Weights stored in DB: id=%d  saved_as=%s  round=%d  slot=%d  size=%.1f KB  sha256=%s",
            row_id,
            saved_as,
            key["round_id"],
            key["slot"],
            size_kb,
            checksum,
        )
    except Exception as exc:
        current_app.logger.error("Failed to store weights in DB: %s", exc)
        return jsonify({"detail": f"Database error: {exc}"}), 500

    # ── FedAvg: re-aggregate central model with the new contribution ─────────
    agg_result = None
    agg_error = None
    try:
        agg_result = aggregate()
        current_app.logger.info(
            "FedAvg complete: %d file(s) aggregated, %d skipped",
            agg_result["aggregated"],
            agg_result["skipped"],
        )
    except Exception as exc:
        agg_error = str(exc)
        current_app.logger.warning("FedAvg failed after upload: %s", exc)

    # ── Evaluate: score the new central model against held-out test data ─────
    eval_result = None
    eval_error = None
    if agg_result:
        try:
            eval_result = evaluate()
            current_app.logger.info(
                "Evaluation complete: accuracy=%.4f  f1=%.4f",
                eval_result["accuracy"],
                eval_result["f1"],
            )
        except Exception as exc:
            eval_error = str(exc)
            current_app.logger.warning("Evaluation failed: %s", exc)

    response = {
        "message": f'"{filename}" uploaded successfully.',
        "saved_as": saved_as,
        "size_kb": size_kb,
        "sha256": checksum,
        "uploaded_at": timestamp,
        "round_id": key["round_id"],
        "slot": key["slot"],
    }

    if agg_result:
        response["aggregation"] = {
            "aggregated": agg_result["aggregated"],
            "skipped": agg_result["skipped"],
        }
    elif agg_error:
        response["aggregation_warning"] = agg_error
        response["contributors_required"] = MIN_CONTRIBUTORS

    if eval_result:
        response["metrics"] = eval_result
    elif eval_error:
        response["metrics_warning"] = eval_error

    return jsonify(response), 201


# ── GET /api/weights/list ─────────────────────────────────────────────────────


@weights_bp.route("/list", methods=["GET"])
def list_weights():
    try:
        entries = db_list_weights()
    except Exception as exc:
        return jsonify({"detail": f"Database error: {exc}"}), 500

    return jsonify({"weights": entries, "count": len(entries)}), 200
