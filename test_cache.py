import importlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import penrs_cache

TEST_FILES_DIR = (Path.cwd() / "Test_files").resolve()


def local_tmp_dir() -> Path:
    TEST_FILES_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = TEST_FILES_DIR / f"test_tmp_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir.resolve()


def _reload_cache_module(monkeypatch, cache_dir: Path):
    monkeypatch.setenv("PENRS_CACHE_DIR", str(cache_dir))
    importlib.reload(penrs_cache)
    return penrs_cache


def test_cache_key_is_deterministic(monkeypatch):
    mod = _reload_cache_module(monkeypatch, local_tmp_dir() / "cache")

    key1 = mod.cache_key("alpha", "MRNA", "earnings_call", "2025-01-10")
    key2 = mod.cache_key("alpha", "MRNA", "earnings_call", "2025-01-10")
    key3 = mod.cache_key("alpha", "MRNA", "earnings_call", "2025-01-11")

    assert key1 == key2
    assert key1 != key3
    assert len(key1) == 64


def test_cache_set_writes_json_with_metadata(monkeypatch):
    mod = _reload_cache_module(monkeypatch, local_tmp_dir() / "cache")
    payload = {"headline": "Sample", "score": 0.4}

    path = mod.cache_set(
        api="alpha",
        ticker="MRNA",
        doc_type="earnings_call",
        date="2025-01-10",
        payload=payload,
    )

    assert path.is_file()
    written = json.loads(path.read_text(encoding="utf-8"))

    assert "_cached_at" in written
    assert written["_api"] == "alpha"
    assert written["_ticker"] == "MRNA"
    assert written["_doc_type"] == "earnings_call"
    assert written["payload"] == payload


def test_cache_get_returns_payload_when_fresh(monkeypatch):
    mod = _reload_cache_module(monkeypatch, local_tmp_dir() / "cache")
    payload = {"k": "v", "n": 2}
    mod.cache_set(
        api="sec",
        ticker="BIIB",
        doc_type="10q",
        date="2025-02-01",
        payload=payload,
    )

    cached = mod.cache_get(
        api="sec",
        ticker="BIIB",
        doc_type="10q",
        date="2025-02-01",
        max_age_hours=12,
    )

    assert cached == payload


def test_cache_get_returns_none_for_missing_file(monkeypatch):
    mod = _reload_cache_module(monkeypatch, local_tmp_dir() / "cache")

    with patch.object(mod.logger, "info") as log_info:
        cached = mod.cache_get(
            api="ctgov",
            ticker="SAVA",
            doc_type="clinical_trial",
            date="2025-02-01",
            max_age_hours=24,
        )

    assert cached is None
    assert log_info.call_count >= 1


def test_cache_get_returns_none_when_expired(monkeypatch):
    mod = _reload_cache_module(monkeypatch, local_tmp_dir() / "cache")
    payload = {"x": 1}
    path = mod.cache_set(
        api="openfda",
        ticker="MRNA",
        doc_type="adverse_events",
        date="2025-02-01",
        payload=payload,
    )

    expired = json.loads(path.read_text(encoding="utf-8"))
    expired["_cached_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=5)
    ).isoformat()
    path.write_text(json.dumps(expired, ensure_ascii=True), encoding="utf-8")

    with patch.object(mod.logger, "info") as log_info:
        cached = mod.cache_get(
            api="openfda",
            ticker="MRNA",
            doc_type="adverse_events",
            date="2025-02-01",
            max_age_hours=1,
        )

    assert cached is None
    assert log_info.call_count >= 1


def test_cache_operations_are_logged(monkeypatch):
    mod = _reload_cache_module(monkeypatch, local_tmp_dir() / f"cache-{uuid.uuid4().hex}")

    with patch.object(mod.logger, "info") as log_info:
        mod.cache_set(
            api="pubmed",
            ticker="SRPT",
            doc_type="publication",
            date="2025-03-02",
            payload={"paper_count": 3},
        )
        result = mod.cache_get(
            api="pubmed",
            ticker="SRPT",
            doc_type="publication",
            date="2025-03-02",
            max_age_hours=1,
        )

    assert result == {"paper_count": 3}
    assert log_info.call_count >= 3
