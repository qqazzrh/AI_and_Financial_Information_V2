import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("penrs_mcp")

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
