import hashlib
import os
from datetime import datetime, timezone

from aggregate import aggregate
from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

weights_bp = Blueprint("weights", __name__)

ALLOWED_EXTENSIONS = {".pt", ".pth", ".pkl"}


def _allowed(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def _upload_dir() -> str:
    path = current_app.config["WEIGHTS_UPLOAD_DIR"]
    os.makedirs(path, exist_ok=True)
    return path


def _sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── POST /api/weights/upload ────────────────────────────────────────────────


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

    # Prefix with a UTC timestamp so filenames never collide
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    saved_name = f"{timestamp}_{filename}"
    save_path = os.path.join(_upload_dir(), saved_name)

    file.save(save_path)

    checksum = _sha256(save_path)
    size_kb = round(os.path.getsize(save_path) / 1024, 2)

    current_app.logger.info(
        "Weights uploaded: %s  size=%.1f KB  sha256=%s", saved_name, size_kb, checksum
    )

    # ── FedAvg: re-aggregate central model with the new contribution ─────────
    agg_result = None
    agg_error = None
    try:
        agg_result = aggregate(upload_dir=_upload_dir())
        current_app.logger.info(
            "FedAvg complete: %d file(s) aggregated, %d skipped",
            agg_result["aggregated"],
            agg_result["skipped"],
        )
    except Exception as exc:
        agg_error = str(exc)
        current_app.logger.warning("FedAvg failed after upload: %s", exc)

    response = {
        "message": f'"{filename}" uploaded successfully.',
        "saved_as": saved_name,
        "size_kb": size_kb,
        "sha256": checksum,
        "uploaded_at": timestamp,
    }

    if agg_result:
        response["aggregation"] = {
            "aggregated": agg_result["aggregated"],
            "skipped": agg_result["skipped"],
        }
    elif agg_error:
        response["aggregation_warning"] = agg_error

    return jsonify(response), 201


# ── GET /api/weights/list ───────────────────────────────────────────────────


@weights_bp.route("/list", methods=["GET"])
def list_weights():
    upload_dir = _upload_dir()
    entries = []

    for fname in sorted(os.listdir(upload_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue

        fpath = os.path.join(upload_dir, fname)
        stat = os.stat(fpath)
        entries.append(
            {
                "filename": fname,
                "size_kb": round(stat.st_size / 1024, 2),
                "uploaded_at": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    return jsonify({"weights": entries, "count": len(entries)}), 200
