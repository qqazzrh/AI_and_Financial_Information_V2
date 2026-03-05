"""
utils.py — Shared utilities for PENRS notebooks.

Single import point for all three notebooks (worker_nodes, orchestrator, tests).
Contains: Cache, HTTP, Rate Limit, Router.
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

import httpx

logger = logging.getLogger("penrs_mcp")


## ─── Cache ───

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


## ─── HTTP ───

_RETRY_STATUSES = {429, 503}
_MAX_RETRIES = 3
_DEFAULT_TIMEOUT = 30.0


async def _api_request(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    api_name: str = "unknown",
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict:
    """Shared async HTTP client with retry and structured error responses."""
    params = params or {}
    headers = headers or {}

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, _MAX_RETRIES + 2):  # 1 initial + 3 retries
            try:
                response = await client.get(url, params=params, headers=headers)
            except httpx.TimeoutException:
                logger.error("[%s] Request timed out: %s", api_name, url)
                return {"error": "Request timed out", "detail": f"URL: {url}"}
            except httpx.RequestError as exc:
                logger.error("[%s] Request error: %s", api_name, exc)
                return {"error": "Request failed", "detail": str(exc)}

            if response.status_code in _RETRY_STATUSES and attempt <= _MAX_RETRIES:
                wait = 2 ** (attempt - 1)  # exponential backoff: 1s, 2s, 4s
                logger.warning(
                    "[%s] HTTP %s, retry %d/%d in %ds",
                    api_name, response.status_code, attempt, _MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
                continue

            if response.is_error:
                logger.error(
                    "[%s] HTTP error %s: %s", api_name, response.status_code, url
                )
                return {
                    "error": f"HTTP {response.status_code}",
                    "detail": response.text[:500],
                }

            try:
                return response.json()
            except Exception:
                return {"text": response.text}

    # Exhausted retries
    logger.error("[%s] Exhausted retries for %s", api_name, url)
    return {"error": "Max retries exceeded", "detail": f"URL: {url}"}


## ─── Rate Limit ───

_ALPHA_DAILY_LIMIT = 25
_ALPHA_MINUTE_LIMIT = 5
_ALPHA_SLEEP_SECONDS = 12
_SEC_MINUTE_LIMIT = 10
_DEFAULT_RPM_LIMIT = int(os.getenv("PENRS_DEFAULT_RPM_LIMIT", "60"))

_RATE_LIMIT_STATE: dict[str, dict[str, Any]] = {}
_RATE_LIMIT_LOCK = threading.Lock()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_api_name(api_name: str) -> str:
    return api_name.strip().lower().replace(" ", "_").replace("-", "_")


def _limits_for_api(api_name: str, rpm_limit: int | None) -> tuple[int | None, int | None]:
    normalized = _normalize_api_name(api_name)
    if normalized in {"alpha", "alpha_vantage", "alphavantage"}:
        return _ALPHA_MINUTE_LIMIT, _ALPHA_DAILY_LIMIT
    if normalized in {"sec", "sec_edgar", "edgar"}:
        return _SEC_MINUTE_LIMIT, None
    return (rpm_limit if rpm_limit is not None else _DEFAULT_RPM_LIMIT), None


def _warn_if_approaching_or_hit(
    *,
    api_name: str,
    window: str,
    count: int,
    limit: int | None,
    blocked: bool = False,
) -> None:
    if limit is None:
        return

    if blocked:
        logger.warning("[%s] %s rate limit reached (%d/%d)", api_name, window, count, limit)
        return

    if count == limit - 1:
        logger.warning(
            "[%s] %s rate limit approaching (%d/%d)",
            api_name,
            window,
            count,
            limit,
        )
    elif count == limit:
        logger.warning("[%s] %s rate limit reached (%d/%d)", api_name, window, count, limit)


def _reset_rate_limit_state() -> None:
    with _RATE_LIMIT_LOCK:
        _RATE_LIMIT_STATE.clear()


def _check_rate_limit(api_name: str, rpm_limit: int | None = None) -> bool:
    """Check and update API-specific daily/minute request counters."""
    normalized = _normalize_api_name(api_name)
    minute_limit, daily_limit = _limits_for_api(normalized, rpm_limit)
    if minute_limit is not None and minute_limit <= 0:
        raise ValueError("rpm_limit must be greater than zero")

    with _RATE_LIMIT_LOCK:
        now = _now_utc()
        day_key = now.date().isoformat()
        minute_key = now.replace(second=0, microsecond=0).isoformat()

        state = _RATE_LIMIT_STATE.setdefault(
            normalized,
            {
                "day_key": day_key,
                "daily_count": 0,
                "minute_key": minute_key,
                "minute_count": 0,
            },
        )

        if state["day_key"] != day_key:
            state["day_key"] = day_key
            state["daily_count"] = 0

        if state["minute_key"] != minute_key:
            state["minute_key"] = minute_key
            state["minute_count"] = 0

        if daily_limit is not None and state["daily_count"] >= daily_limit:
            _warn_if_approaching_or_hit(
                api_name=normalized,
                window="daily",
                count=state["daily_count"],
                limit=daily_limit,
                blocked=True,
            )
            return False

        if minute_limit is not None and state["minute_count"] >= minute_limit:
            if normalized in {"alpha", "alpha_vantage", "alphavantage"}:
                logger.warning(
                    "[%s] minute rate limit reached (%d/%d), sleeping %ss",
                    normalized,
                    state["minute_count"],
                    minute_limit,
                    _ALPHA_SLEEP_SECONDS,
                )
                time.sleep(_ALPHA_SLEEP_SECONDS)
                state["minute_key"] = _now_utc().replace(second=0, microsecond=0).isoformat()
                state["minute_count"] = 0
            else:
                _warn_if_approaching_or_hit(
                    api_name=normalized,
                    window="minute",
                    count=state["minute_count"],
                    limit=minute_limit,
                    blocked=True,
                )
                return False

        state["minute_count"] += 1
        state["daily_count"] += 1

        _warn_if_approaching_or_hit(
            api_name=normalized,
            window="minute",
            count=state["minute_count"],
            limit=minute_limit,
        )
        _warn_if_approaching_or_hit(
            api_name=normalized,
            window="daily",
            count=state["daily_count"],
            limit=daily_limit,
        )
        return True


## ─── Router ───

class DocumentType(str, Enum):
    EARNINGS_CALL = "earnings_call"
    FORM_4 = "form_4"
    NEWS_SENTIMENT = "news_sentiment"
    PRICE_HISTORY = "price_history"
    SEC_10K = "sec_10k"
    SEC_10Q = "sec_10q"
    SEC_8K = "sec_8k"
    CLINICAL_TRIALS = "clinical_trials"
    BIOMEDICAL_EVIDENCE = "biomedical_evidence"


DateRange = tuple[str | None, str | None] | dict[str, str | None] | None
Fetcher = Callable[[str, tuple[str | None, str | None], DocumentType], Awaitable[Any]]

DOCUMENT_API_ROUTING: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.EARNINGS_CALL: ("alpha_vantage",),
    DocumentType.FORM_4: ("alpha_vantage",),
    DocumentType.NEWS_SENTIMENT: ("alpha_vantage",),
    DocumentType.PRICE_HISTORY: ("alpha_vantage",),
    DocumentType.SEC_10K: ("sec_edgar",),
    DocumentType.SEC_10Q: ("sec_edgar",),
    DocumentType.SEC_8K: ("sec_edgar",),
    DocumentType.CLINICAL_TRIALS: ("clinicaltrials_gov",),
    DocumentType.BIOMEDICAL_EVIDENCE: ("openfda", "pubmed"),
}


def _normalize_date_range(date_range: DateRange) -> tuple[str | None, str | None]:
    if date_range is None:
        return (None, None)
    if isinstance(date_range, tuple):
        if len(date_range) != 2:
            raise ValueError("date_range tuple must be (date_from, date_to)")
        return date_range
    if isinstance(date_range, dict):
        return (date_range.get("from"), date_range.get("to"))
    raise TypeError("date_range must be a tuple, dict, or None")


async def _missing_fetcher(
    api_name: str,
    _ticker: str,
    _date_range: tuple[str | None, str | None],
    _document_type: DocumentType,
) -> dict[str, str]:
    return {"error": f"No fetcher configured for API '{api_name}'"}


def _is_usable_data(payload: Any) -> bool:
    if payload is None:
        return False
    if isinstance(payload, str):
        return bool(payload.strip())
    if isinstance(payload, (list, tuple, set, dict)):
        return len(payload) > 0
    return True


def _extract_source_data(result: Any) -> Any | None:
    if isinstance(result, dict):
        status = result.get("status")
        if status == "available":
            return result.get("data")
        if status == "not_released":
            return None
        if "error" in result:
            return None
        return result
    return result


async def penrs_fetch_document(
    ticker: str,
    document_type: DocumentType,
    date_range: DateRange = None,
    fetchers: Mapping[str, Fetcher] | None = None,
) -> dict[str, Any]:
    if not isinstance(document_type, DocumentType):
        raise TypeError("document_type must be an instance of DocumentType")

    normalized_date_range = _normalize_date_range(date_range)
    apis = DOCUMENT_API_ROUTING[document_type]
    fetchers = fetchers or {}

    coroutines = []
    for api in apis:
        fetcher = fetchers.get(api)
        if fetcher is None:
            coroutines.append(_missing_fetcher(api, ticker, normalized_date_range, document_type))
        else:
            coroutines.append(fetcher(ticker, normalized_date_range, document_type))

    raw_results = await asyncio.gather(*coroutines, return_exceptions=True)
    sources: list[dict[str, Any]] = []
    partial_failures: list[dict[str, str]] = []

    for api, raw_result in zip(apis, raw_results):
        if isinstance(raw_result, Exception):
            partial_failures.append({"api": api, "error": str(raw_result)})
            continue

        if isinstance(raw_result, dict) and "error" in raw_result:
            partial_failures.append({"api": api, "error": str(raw_result["error"])})
            continue

        source_data = _extract_source_data(raw_result)
        if _is_usable_data(source_data):
            sources.append({"api": api, "data": source_data})

    data: dict[str, Any] = {
        "ticker": ticker,
        "document_type": document_type.value,
        "date_range": {
            "from": normalized_date_range[0],
            "to": normalized_date_range[1],
        },
        "apis_attempted": list(apis),
    }

    if sources:
        data["sources"] = sources
        if partial_failures:
            data["partial_failures"] = partial_failures
        return {"status": "available", "data": data}

    if partial_failures:
        data["errors"] = partial_failures

    return {"status": "not_released", "data": data}


__all__ = [
    # Cache
    "PENRS_CACHE_DIR",
    "cache_get",
    "cache_key",
    "cache_set",
    # HTTP
    "_api_request",
    # Rate limiting
    "_check_rate_limit",
    "_reset_rate_limit_state",
    # Router
    "DOCUMENT_API_ROUTING",
    "DocumentType",
    "penrs_fetch_document",
]
