import asyncio
import logging
import httpx

logger = logging.getLogger("penrs_mcp")

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
