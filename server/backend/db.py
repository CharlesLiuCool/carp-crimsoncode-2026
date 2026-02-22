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
    weights     BYTEA NOT NULL         -- raw .pt file bytes
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
    weights     BYTEA NOT NULL
);
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
) -> int:
    """
    Insert a new weights record. Returns the new row id.

    Parameters
    ----------
    filename : original sanitised filename (e.g. dp_weights.pt)
    saved_as : unique timestamped name (e.g. 20260222T025610Z_dp_weights.pt)
    sha256   : hex SHA-256 of the file bytes
    size_kb  : file size in kilobytes
    data     : raw bytes of the .pt file
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO uploaded_weights
                    (filename, saved_as, sha256, size_kb, weights)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (filename, saved_as, sha256, size_kb, psycopg2.Binary(data)),
            )
            row_id = cur.fetchone()[0]
        conn.commit()

    logger.info("Inserted weights row id=%d  saved_as=%s", row_id, saved_as)
    return row_id


def fetch_all_valid_weights() -> list[dict]:
    """
    Return all rows where is_valid=true as a list of dicts:
        { id, filename, saved_as, sha256, size_kb, uploaded_at, weights_bytes }
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, saved_as, sha256, size_kb, uploaded_at, weights
                FROM uploaded_weights
                WHERE is_valid = true
                ORDER BY uploaded_at ASC
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
