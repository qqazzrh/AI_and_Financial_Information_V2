import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("penrs_mcp")

PENRS_CACHE_DIR = Path(os.getenv("PENRS_CACHE_DIR", ".penrs_cache")).resolve()
PENRS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_META_FIELDS = {"_cached_at", "_api", "_ticker", "_doc_type", "_date"}


def cache_key(api: str, ticker: str, doc_type: str, date: str | None = None) -> str:
    """Build a deterministic SHA-256 key for cache lookups."""
    material = f"{api}|{ticker}|{doc_type}|{date or ''}"
    key = hashlib.sha256(material.encode("utf-8")).hexdigest()
    logger.info(
        "cache_key generated for api=%s ticker=%s doc_type=%s date=%s",
        api,
        ticker,
        doc_type,
        date,
    )
    return key


def _cache_path(api: str, ticker: str, doc_type: str, date: str | None = None) -> Path:
    key = cache_key(api=api, ticker=ticker, doc_type=doc_type, date=date)
    return PENRS_CACHE_DIR / f"{key}.json"


def cache_set(
    api: str,
    ticker: str,
    doc_type: str,
    date: str | None,
    payload: dict[str, Any],
) -> Path:
    """Persist a cache payload with metadata and timestamp."""
    path = _cache_path(api=api, ticker=ticker, doc_type=doc_type, date=date)
    record = {
        "_cached_at": datetime.now(timezone.utc).isoformat(),
        "_api": api,
        "_ticker": ticker,
        "_doc_type": doc_type,
        "_date": date,
        "payload": payload,
    }
    path.write_text(json.dumps(record, ensure_ascii=True), encoding="utf-8")
    logger.info("cache_set wrote %s", path)
    return path


def cache_get(
    api: str,
    ticker: str,
    doc_type: str,
    date: str | None,
    max_age_hours: float,
) -> dict[str, Any] | None:
    """Return cached payload if present and not expired."""
    path = _cache_path(api=api, ticker=ticker, doc_type=doc_type, date=date)
    if not path.exists():
        logger.info("cache_get miss (missing file): %s", path)
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("cache_get miss (read/parse error): %s (%s)", path, exc)
        return None

    cached_at_raw = raw.get("_cached_at")
    if not cached_at_raw:
        logger.warning("cache_get miss (missing _cached_at): %s", path)
        return None

    try:
        cached_at = datetime.fromisoformat(cached_at_raw)
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)
    except ValueError:
        logger.warning("cache_get miss (invalid _cached_at): %s", path)
        return None

    age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600.0
    if age_hours >= max_age_hours:
        logger.info(
            "cache_get expired: %s age_hours=%.3f max_age_hours=%.3f",
            path,
            age_hours,
            max_age_hours,
        )
        return None

    payload = raw.get("payload")
    if payload is not None:
        logger.info("cache_get hit: %s", path)
        return payload

    # Backward-compatible fallback: return non-metadata fields as payload.
    inferred_payload = {k: v for k, v in raw.items() if k not in _META_FIELDS}
    logger.info("cache_get hit (inferred payload): %s", path)
    return inferred_payload if inferred_payload else None
