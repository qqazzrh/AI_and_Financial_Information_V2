import importlib
from datetime import datetime, timedelta, timezone
from unittest.mock import patch


def _reload_rate_limit_module():
    import penrs_rate_limit

    importlib.reload(penrs_rate_limit)
    penrs_rate_limit._reset_rate_limit_state()
    return penrs_rate_limit


def test_alpha_vantage_blocks_after_25_daily_calls(monkeypatch):
    mod = _reload_rate_limit_module()
    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    sleeps = []

    monkeypatch.setattr(mod, "_now_utc", lambda: now)
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    for _ in range(25):
        assert mod._check_rate_limit("alpha_vantage") is True
    assert mod._check_rate_limit("alpha_vantage") is False

    assert all(seconds == 12 for seconds in sleeps)


def test_alpha_vantage_sleeps_12_seconds_after_5_calls_same_minute(monkeypatch):
    mod = _reload_rate_limit_module()
    now = datetime(2026, 3, 2, 9, 30, tzinfo=timezone.utc)
    sleeps = []

    monkeypatch.setattr(mod, "_now_utc", lambda: now)
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    for _ in range(5):
        assert mod._check_rate_limit("alpha_vantage") is True
    assert mod._check_rate_limit("alpha_vantage") is True

    assert sleeps == [12]


def test_daily_counter_resets_at_midnight_boundary(monkeypatch):
    mod = _reload_rate_limit_module()
    previous_minute = datetime(2026, 3, 1, 23, 59, tzinfo=timezone.utc)
    midnight = datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)

    mod._RATE_LIMIT_STATE["alpha_vantage"] = {
        "day_key": previous_minute.date().isoformat(),
        "daily_count": 25,
        "minute_key": previous_minute.replace(second=0, microsecond=0).isoformat(),
        "minute_count": 5,
    }

    monkeypatch.setattr(mod, "_now_utc", lambda: midnight)
    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)

    assert mod._check_rate_limit("alpha_vantage") is True
    assert mod._RATE_LIMIT_STATE["alpha_vantage"]["daily_count"] == 1


def test_minute_counter_resets_on_new_minute_for_sec_edgar(monkeypatch):
    mod = _reload_rate_limit_module()
    now = datetime(2026, 3, 2, 11, 15, tzinfo=timezone.utc)
    next_minute = now + timedelta(minutes=1)

    monkeypatch.setattr(mod, "_now_utc", lambda: now)

    for _ in range(10):
        assert mod._check_rate_limit("sec_edgar") is True
    assert mod._check_rate_limit("sec_edgar") is False

    monkeypatch.setattr(mod, "_now_utc", lambda: next_minute)
    assert mod._check_rate_limit("sec_edgar") is True


def test_other_apis_use_configurable_rpm_limit(monkeypatch):
    mod = _reload_rate_limit_module()
    now = datetime(2026, 3, 2, 8, 45, tzinfo=timezone.utc)
    monkeypatch.setattr(mod, "_now_utc", lambda: now)

    assert mod._check_rate_limit("pubmed", rpm_limit=2) is True
    assert mod._check_rate_limit("pubmed", rpm_limit=2) is True
    assert mod._check_rate_limit("pubmed", rpm_limit=2) is False


def test_warning_logged_when_limits_approached_or_hit(monkeypatch):
    mod = _reload_rate_limit_module()
    now = datetime(2026, 3, 2, 8, 50, tzinfo=timezone.utc)
    monkeypatch.setattr(mod, "_now_utc", lambda: now)

    with patch.object(mod.logger, "warning") as log_warning:
        assert mod._check_rate_limit("pubmed", rpm_limit=2) is True
        assert mod._check_rate_limit("pubmed", rpm_limit=2) is True
        assert mod._check_rate_limit("pubmed", rpm_limit=2) is False

    assert log_warning.call_count >= 2
