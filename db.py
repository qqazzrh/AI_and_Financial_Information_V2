"""Database layer for the stateful tiered pipeline.

Manages two embedded databases:
- SQLite: users, document run cache (the "cache ledger"), and query history
- LanceDB: document chunks and Voyage AI embedding vectors

All functions are synchronous since the pipeline itself is synchronous.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger("biotech_disclosure_pipeline")

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    subscription_tier TEXT NOT NULL DEFAULT 'standard'
        CHECK(subscription_tier IN ('standard', 'premium')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS document_runs (
    run_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    document_type TEXT NOT NULL,
    release_date TEXT,
    analysis_tier TEXT NOT NULL
        CHECK(analysis_tier IN ('text_only', 'vector_graph')),
    worker_payload TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_runs_cache
    ON document_runs(ticker, document_type, release_date, analysis_tier);

CREATE TABLE IF NOT EXISTS query_history (
    query_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    final_arbiter_response TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (run_id) REFERENCES document_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_query_history_user
    ON query_history(user_id);
"""


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_sqlite(db_dir: Path) -> sqlite3.Connection:
    """Create or open the SQLite database and ensure the schema exists."""
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "pipeline.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SQLITE_SCHEMA)
    conn.commit()
    logger.info("SQLite database ready at %s", db_path)
    return conn


def init_lancedb(db_dir: Path) -> Any:
    """Create or open the LanceDB database directory.

    Returns a lancedb.DBConnection (imported lazily so the module can load
    even if lancedb is not yet installed).
    """
    import lancedb

    lance_dir = db_dir / "lancedb"
    lance_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lance_dir))
    logger.info("LanceDB ready at %s", lance_dir)
    return db


def init_databases(db_dir: Path | None = None) -> tuple[sqlite3.Connection, Any]:
    """Initialize both databases. Returns (sqlite_conn, lance_db)."""
    if db_dir is None:
        env_dir = os.environ.get("DATABASE_DIR", "data/db")
        db_dir = Path(env_dir)
    return init_sqlite(db_dir), init_lancedb(db_dir)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_or_create_user(
    conn: sqlite3.Connection,
    user_id: str,
    tier: str = "standard",
) -> dict[str, Any]:
    """Return existing user or insert a new one."""
    row = conn.execute(
        "SELECT user_id, subscription_tier, created_at FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    if row:
        return dict(row)
    conn.execute(
        "INSERT INTO users (user_id, subscription_tier) VALUES (?, ?)",
        (user_id, tier),
    )
    conn.commit()
    return {"user_id": user_id, "subscription_tier": tier, "created_at": _now_iso()}


# ---------------------------------------------------------------------------
# Document run cache (the "cache ledger")
# ---------------------------------------------------------------------------

def find_cached_run(
    conn: sqlite3.Connection,
    ticker: str,
    document_type: str,
    release_date: str | None,
    analysis_tier: str,
) -> dict[str, Any] | None:
    """Look up a cached document run. Returns None on miss."""
    row = conn.execute(
        """
        SELECT run_id, ticker, document_type, release_date,
               analysis_tier, worker_payload, created_at
        FROM document_runs
        WHERE ticker = ? AND document_type = ? AND release_date IS ? AND analysis_tier = ?
        """,
        (ticker.upper(), document_type, release_date, analysis_tier),
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    if result.get("worker_payload"):
        result["worker_payload"] = json.loads(result["worker_payload"])
    return result


def find_any_cached_run(
    conn: sqlite3.Connection,
    ticker: str,
    document_type: str,
    release_date: str | None,
) -> dict[str, Any] | None:
    """Find any cached run for this document (any tier). Prefers vector_graph."""
    row = conn.execute(
        """
        SELECT run_id, ticker, document_type, release_date,
               analysis_tier, worker_payload, created_at
        FROM document_runs
        WHERE ticker = ? AND document_type = ? AND release_date IS ?
        ORDER BY CASE analysis_tier WHEN 'vector_graph' THEN 0 ELSE 1 END
        LIMIT 1
        """,
        (ticker.upper(), document_type, release_date),
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    if result.get("worker_payload"):
        result["worker_payload"] = json.loads(result["worker_payload"])
    return result


def save_document_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    ticker: str,
    document_type: str,
    release_date: str | None,
    analysis_tier: str,
    worker_payload: dict[str, Any] | None = None,
) -> None:
    """Insert or replace a document run in the cache ledger."""
    conn.execute(
        """
        INSERT OR REPLACE INTO document_runs
            (run_id, ticker, document_type, release_date, analysis_tier, worker_payload)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            ticker.upper(),
            document_type,
            release_date,
            analysis_tier,
            json.dumps(worker_payload, default=str) if worker_payload else None,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Query history
# ---------------------------------------------------------------------------

def record_query(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    run_id: str,
    ticker: str,
    arbiter_response: dict[str, Any] | None = None,
) -> str:
    """Record a user query, linking to the document run. Returns query_id."""
    query_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO query_history (query_id, user_id, run_id, ticker, final_arbiter_response)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            query_id,
            user_id,
            run_id,
            ticker.upper(),
            json.dumps(arbiter_response, default=str) if arbiter_response else None,
        ),
    )
    conn.commit()
    return query_id


def get_user_history(
    conn: sqlite3.Connection,
    user_id: str,
) -> list[dict[str, Any]]:
    """Return all past queries for a user, newest first."""
    rows = conn.execute(
        """
        SELECT qh.query_id, qh.user_id, qh.run_id, qh.ticker,
               qh.timestamp, qh.final_arbiter_response,
               dr.analysis_tier, dr.document_type, dr.release_date
        FROM query_history qh
        JOIN document_runs dr ON qh.run_id = dr.run_id
        WHERE qh.user_id = ?
        ORDER BY qh.timestamp DESC
        """,
        (user_id,),
    ).fetchall()
    results = []
    for row in rows:
        entry = dict(row)
        if entry.get("final_arbiter_response"):
            entry["final_arbiter_response"] = json.loads(entry["final_arbiter_response"])
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# LanceDB chunk / embedding storage
# ---------------------------------------------------------------------------

def save_chunks_to_lancedb(
    lance_db: Any,
    *,
    document_id: str,
    chunks: Sequence[dict[str, Any]],
    embeddings: Sequence[Sequence[float]] | None = None,
) -> None:
    """Save document chunks (and optionally their embeddings) to LanceDB.

    Each record contains: document_id, chunk_id, chunk_index, text,
    section_id, section_title, vector (if embeddings provided),
    and graph edge metadata (previous_chunk_id, next_chunk_id,
    cross_reference_targets).
    """
    if not chunks:
        return

    records = []
    for i, chunk in enumerate(chunks):
        record: dict[str, Any] = {
            "document_id": document_id,
            "chunk_id": chunk.get("chunk_id", f"{document_id}_chunk_{i}"),
            "chunk_index": chunk.get("chunk_index", i),
            "text": chunk.get("text", ""),
            "section_id": chunk.get("parent_section_id"),
            "section_title": chunk.get("parent_section_title"),
            "previous_chunk_id": chunk.get("previous_chunk_id"),
            "next_chunk_id": chunk.get("next_chunk_id"),
            "cross_reference_targets": json.dumps(
                chunk.get("cross_reference_targets", [])
            ),
        }
        if embeddings and i < len(embeddings):
            record["vector"] = embeddings[i]
        records.append(record)

    table_name = "document_chunks"
    existing_tables = lance_db.table_names()
    if table_name in existing_tables:
        try:
            table = lance_db.open_table(table_name)
            # Delete existing chunks for this document before re-inserting
            table.delete(f'document_id = "{document_id}"')
            table.add(records)
        except Exception:
            # Schema mismatch — drop and recreate
            lance_db.drop_table(table_name)
            lance_db.create_table(table_name, records)
    else:
        lance_db.create_table(table_name, records)

    logger.info(
        "Saved %d chunks for document %s to LanceDB (embeddings=%s)",
        len(records),
        document_id,
        embeddings is not None,
    )


def get_chunks_from_lancedb(
    lance_db: Any,
    document_id: str,
) -> list[dict[str, Any]]:
    """Retrieve all chunks for a document from LanceDB."""
    table_name = "document_chunks"
    try:
        table = lance_db.open_table(table_name)
        results = (
            table.search()
            .where(f'document_id = "{document_id}"')
            .limit(10000)
            .to_list()
        )
        return results
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
