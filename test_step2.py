import asyncio
from unittest.mock import patch

import httpx

import penrs_http


class DummyResponse:
    def __init__(
        self,
        status_code: int,
        *,
        json_data=None,
        text: str = "",
        json_error: Exception | None = None,
    ):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self._json_error = json_error

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._json_data


def _patch_async_client(monkeypatch, planned_results):
    instances = []

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout
            self.calls = []
            instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None, headers=None):
            self.calls.append({"url": url, "params": params, "headers": headers})
            if not planned_results:
                raise AssertionError("No planned response left for FakeAsyncClient.get()")

            next_result = planned_results.pop(0)
            if isinstance(next_result, Exception):
                raise next_result
            return next_result

    monkeypatch.setattr(penrs_http.httpx, "AsyncClient", FakeAsyncClient)
    return instances


def test_api_request_success_json_uses_default_timeout(monkeypatch):
    planned = [DummyResponse(200, json_data={"ok": True})]
    clients = _patch_async_client(monkeypatch, planned)

    result = asyncio.run(
        penrs_http._api_request(
            "https://example.test/data",
            params={"ticker": "MRNA"},
            headers={"X-Test": "1"},
            api_name="alpha_vantage",
        )
    )

    assert result == {"ok": True}
    assert len(clients) == 1
    assert clients[0].timeout == 30.0
    assert clients[0].calls == [
        {
            "url": "https://example.test/data",
            "params": {"ticker": "MRNA"},
            "headers": {"X-Test": "1"},
        }
    ]


def test_api_request_returns_text_when_json_parse_fails(monkeypatch):
    planned = [DummyResponse(200, text="raw text body", json_error=ValueError("bad json"))]
    _patch_async_client(monkeypatch, planned)

    result = asyncio.run(penrs_http._api_request("https://example.test/text"))

    assert result == {"text": "raw text body"}


def test_api_request_returns_structured_http_error_and_logs(monkeypatch):
    planned = [DummyResponse(500, text="server exploded")]
    _patch_async_client(monkeypatch, planned)

    with patch.object(penrs_http.logger, "error") as log_error:
        result = asyncio.run(penrs_http._api_request("https://example.test/fail", api_name="sec"))

    assert result == {"error": "HTTP 500", "detail": "server exploded"}
    assert log_error.call_count >= 1


def test_api_request_retries_429_503_with_exponential_backoff(monkeypatch):
    planned = [
        DummyResponse(429, text="rate limited"),
        DummyResponse(503, text="service unavailable"),
        DummyResponse(200, json_data={"status": "ok"}),
    ]
    clients = _patch_async_client(monkeypatch, planned)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(penrs_http.asyncio, "sleep", fake_sleep)

    with patch.object(penrs_http.logger, "warning") as log_warning:
        result = asyncio.run(penrs_http._api_request("https://example.test/retry", api_name="ctgov"))

    assert result == {"status": "ok"}
    assert sleeps == [1, 2]
    assert len(clients[0].calls) == 3
    assert log_warning.call_count == 2


def test_api_request_stops_after_max_retries_on_429(monkeypatch):
    planned = [
        DummyResponse(429, text="limited"),
        DummyResponse(429, text="limited"),
        DummyResponse(429, text="limited"),
        DummyResponse(429, text="still limited"),
    ]
    clients = _patch_async_client(monkeypatch, planned)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(penrs_http.asyncio, "sleep", fake_sleep)

    result = asyncio.run(penrs_http._api_request("https://example.test/limit", api_name="alpha"))

    assert result == {"error": "HTTP 429", "detail": "still limited"}
    assert len(clients[0].calls) == 4
    assert sleeps == [1, 2, 4]


def test_api_request_timeout_is_user_friendly_and_logged(monkeypatch):
    planned = [httpx.TimeoutException("took too long")]
    _patch_async_client(monkeypatch, planned)

    with patch.object(penrs_http.logger, "error") as log_error:
        result = asyncio.run(penrs_http._api_request("https://example.test/timeout", api_name="openfda"))

    assert result["error"] == "Request timed out"
    assert "https://example.test/timeout" in result["detail"]
    assert log_error.call_count >= 1


def test_api_request_request_error_is_user_friendly_and_logged(monkeypatch):
    request = httpx.Request("GET", "https://example.test/network")
    planned = [httpx.RequestError("network unreachable", request=request)]
    _patch_async_client(monkeypatch, planned)

    with patch.object(penrs_http.logger, "error") as log_error:
        result = asyncio.run(penrs_http._api_request("https://example.test/network", api_name="pubmed"))

    assert result["error"] == "Request failed"
    assert "network unreachable" in result["detail"]
    assert log_error.call_count >= 1


def test_api_request_respects_custom_timeout(monkeypatch):
    planned = [DummyResponse(200, json_data={"ok": True})]
    clients = _patch_async_client(monkeypatch, planned)

    result = asyncio.run(
        penrs_http._api_request(
            "https://example.test/custom-timeout",
            timeout=5.5,
            api_name="alpha_vantage",
        )
    )

    assert result == {"ok": True}
    assert clients[0].timeout == 5.5
