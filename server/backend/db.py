"""
Database layer for CARP server backend.

Uses PostgreSQL with psycopg2. Connection string is read from DATABASE_URL in .env.

Schema
------
uploaded_weights
    id          SERIAL PRIMARY KEY
    filename    TEXT NOT NULL          -- original sanitised filename
    saved_as    TEXT NOT NULL UNIQUE   -- timestamp-prefixed unique name
    sha256      TEXT NOT NULL
    size_kb     FLOAT NOT NULL
    uploaded_at TIMESTAMPTZ DEFAULT now()
    is_valid    BOOLEAN DEFAULT true   -- set false to exclude from FedAvg
    weights     BYTEA NOT NULL         -- raw masked .pt file bytes
    round_id    INTEGER                -- which aggregation round this upload belongs to
    slot        INTEGER                -- slot within the round (1, 2, or 3)

metrics_history
    id             SERIAL PRIMARY KEY
    recorded_at    TIMESTAMPTZ DEFAULT now()
    accuracy       FLOAT NOT NULL
    precision      FLOAT NOT NULL
    recall         FLOAT NOT NULL
    f1             FLOAT NOT NULL
"""

import io
import logging
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.pool

logger = logging.getLogger(__name__)

_pool: psycopg2.pool.SimpleConnectionPool | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploaded_weights (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    saved_as    TEXT NOT NULL UNIQUE,
    sha256      TEXT NOT NULL,
    size_kb     FLOAT NOT NULL,
    uploaded_at TIMESTAMPTZ DEFAULT now(),
    is_valid    BOOLEAN DEFAULT true,
    weights     BYTEA NOT NULL,
    round_id    INTEGER,
    slot        INTEGER
);

CREATE TABLE IF NOT EXISTS metrics_history (
    id          SERIAL PRIMARY KEY,
    recorded_at TIMESTAMPTZ DEFAULT now(),
    accuracy    FLOAT NOT NULL,
    precision   FLOAT NOT NULL,
    recall      FLOAT NOT NULL,
    f1          FLOAT NOT NULL
);
"""

# Migration: add round_id and slot to existing uploaded_weights tables
_MIGRATE_ROUNDS = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'uploaded_weights' AND column_name = 'round_id'
    ) THEN
        ALTER TABLE uploaded_weights ADD COLUMN round_id INTEGER;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'uploaded_weights' AND column_name = 'slot'
    ) THEN
        ALTER TABLE uploaded_weights ADD COLUMN slot INTEGER;
    END IF;
END
$$;
"""


# ── Connection pool ───────────────────────────────────────────────────────────


def init_db() -> None:
    """
    Initialise the connection pool and create the schema if it doesn't exist.
    Call once at app startup.
    """
    global _pool

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    _pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        dsn=database_url,
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
            cur.execute(_MIGRATE_ROUNDS)
        conn.commit()

    logger.info("Database initialised.")


@contextmanager
def get_conn():
    """Context manager that checks out a connection and returns it to the pool."""
    if _pool is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")

    conn = _pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


# ── CRUD ──────────────────────────────────────────────────────────────────────


def insert_weights(
    filename: str,
    saved_as: str,
    sha256: str,
    size_kb: float,
    data: bytes,
    round_id: int | None = None,
    slot: int | None = None,
) -> int:
    """
    Insert a new weights record. Returns the new row id.

    Parameters
    ----------
    filename : original sanitised filename (e.g. dp_weights.pt)
    saved_as : unique timestamped name (e.g. 20260222T025610Z_dp_weights.pt)
    sha256   : hex SHA-256 of the original (pre-mask) file bytes
    size_kb  : file size in kilobytes
    data     : masked .pt bytes to store
    round_id : aggregation round this upload belongs to
    slot     : slot within the round (1, 2, or 3)
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO uploaded_weights
                    (filename, saved_as, sha256, size_kb, weights, round_id, slot)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    filename,
                    saved_as,
                    sha256,
                    size_kb,
                    psycopg2.Binary(data),
                    round_id,
                    slot,
                ),
            )
            row_id = cur.fetchone()[0]
        conn.commit()

    logger.info(
        "Inserted weights row id=%d  saved_as=%s  round=%s  slot=%s",
        row_id,
        saved_as,
        round_id,
        slot,
    )
    return row_id


def fetch_all_valid_weights() -> list[dict]:
    """
    Return all rows where is_valid=true as a list of dicts:
        { id, filename, saved_as, sha256, size_kb, uploaded_at, weights_bytes, round_id, slot }
    Ordered by round_id then slot so complete rounds are grouped together.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, saved_as, sha256, size_kb, uploaded_at, weights, round_id, slot
                FROM uploaded_weights
                WHERE is_valid = true
                ORDER BY round_id ASC NULLS LAST, slot ASC, uploaded_at ASC
                """
            )
            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "filename": r[1],
            "saved_as": r[2],
            "sha256": r[3],
            "size_kb": r[4],
            "uploaded_at": r[5].isoformat() if r[5] else None,
            "weights_bytes": bytes(r[6]),
            "round_id": r[7],
            "slot": r[8],
        }
        for r in rows
    ]


def list_weights() -> list[dict]:
    """
    Return metadata only (no weights bytes) for all valid rows.
    Used by GET /api/weights/list.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, saved_as, sha256, size_kb, uploaded_at
                FROM uploaded_weights
                WHERE is_valid = true
                ORDER BY uploaded_at DESC
                """
            )
            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "filename": r[1],
            "saved_as": r[2],
            "sha256": r[3],
            "size_kb": r[4],
            "uploaded_at": r[5].isoformat() if r[5] else None,
        }
        for r in rows
    ]


def mark_invalid(row_id: int) -> None:
    """Soft-delete a weights row so it is excluded from future FedAvg runs."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE uploaded_weights SET is_valid = false WHERE id = %s",
                (row_id,),
            )
        conn.commit()
    logger.info("Marked weights row id=%d as invalid.", row_id)


def weights_to_buffer(weights_bytes: bytes) -> io.BytesIO:
    """Wrap raw bytes in a BytesIO buffer ready for torch.load."""
    buf = io.BytesIO(weights_bytes)
    buf.seek(0)
    return buf


def get_max_round_id() -> int | None:
    """
    Return the highest round_id present in the uploaded_weights table, or
    None if no rounds have been recorded yet.

    Used at startup to resume round numbering after a server restart so
    round IDs are never reused across sessions.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MAX(round_id) FROM uploaded_weights WHERE round_id IS NOT NULL"
            )
            row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else None


def insert_metrics_history(accuracy: float, precision: float, recall: float, f1: float) -> None:
    """Append one row after a successful aggregation + evaluate(). Used for upload log graph."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO metrics_history (accuracy, precision, recall, f1)
                VALUES (%s, %s, %s, %s)
                """,
                (accuracy, precision, recall, f1),
            )
        conn.commit()


def get_metrics_history(limit: int = 3) -> list[dict]:
    """Return the most recent metrics rows (newest first). Used for the upload log graph."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, recorded_at, accuracy, precision, recall, f1
                FROM metrics_history
                ORDER BY recorded_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "recorded_at": r[1].isoformat() if r[1] else None,
            "accuracy": r[2],
            "precision": r[3],
            "recall": r[4],
            "f1": r[5],
        }
        for r in rows
    ]
