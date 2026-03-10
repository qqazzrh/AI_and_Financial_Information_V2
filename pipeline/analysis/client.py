"""Analysis client abstraction and implementations (Moonshot/Kimi, Callable)."""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pipeline.config import MOONSHOT_API_KEY, logger
from pipeline.enums import ProcessingStatus
from pipeline.models import (
    AnalysisClientRequest,
    AnalysisClientResponse,
    AnalysisWarning,
    PipelineConfig,
)
from pipeline.analysis.rubrics import make_analysis_warning

__all__ = [
    "BaseAnalysisClient",
    "CallableAnalysisClient",
    "MoonshotAnalysisClient",
    "build_analysis_client",
]


# ---------------------------------------------------------------------------
# Base analysis client
# ---------------------------------------------------------------------------


class BaseAnalysisClient(ABC):
    """Base interface for local, mock, or wrapped analysis clients."""

    client_name: ClassVar[str] = "base_analysis_client"
    model_name: ClassVar[str] = "configurable_placeholder"

    @abstractmethod
    def run_analysis(self, request: AnalysisClientRequest) -> AnalysisClientResponse:
        """Run one analysis request and return a normalized response object."""


# ---------------------------------------------------------------------------
# Callable wrapper client
# ---------------------------------------------------------------------------


class CallableAnalysisClient(BaseAnalysisClient):
    """Wrap a callable or `.invoke(...)` object without assuming any provider-specific payload shape."""

    client_name = "callable_analysis_client"

    def __init__(
        self,
        invoke_target: Callable[[dict[str, Any]], Any] | Any,
        *,
        client_name: str = "callable_analysis_client",
        model_name: str = "callable_placeholder",
    ) -> None:
        self.invoke_target = invoke_target
        self.client_name = client_name
        self.model_name = model_name

    def run_analysis(self, request: AnalysisClientRequest) -> AnalysisClientResponse:
        request_payload = request.model_dump(mode="json")
        try:
            if hasattr(self.invoke_target, "invoke"):
                raw_result = self.invoke_target.invoke(request_payload)
            elif callable(self.invoke_target):
                raw_result = self.invoke_target(request_payload)
            else:
                raise TypeError("invoke_target must be callable or expose an invoke method.")
        except Exception as exc:
            return AnalysisClientResponse(
                client_name=self.client_name,
                model_name=self.model_name,
                status=ProcessingStatus.ANALYSIS_FAILED,
                raw_text=str(exc),
                warnings=[
                    make_analysis_warning(
                        "analysis_client_invocation_failed",
                        f"Callable analysis client failed: {exc}",
                    )
                ],
                is_mock=False,
            )

        if isinstance(raw_result, AnalysisClientResponse):
            return raw_result
        if isinstance(raw_result, dict):
            return AnalysisClientResponse(
                client_name=self.client_name,
                model_name=self.model_name,
                status=ProcessingStatus.SUCCESS,
                raw_output=raw_result,
            )
        return AnalysisClientResponse(
            client_name=self.client_name,
            model_name=self.model_name,
            status=ProcessingStatus.SUCCESS,
            raw_text=str(raw_result),
        )


# ---------------------------------------------------------------------------
# Moonshot AI (Kimi) client
# ---------------------------------------------------------------------------


class _MoonshotRateLimitError(Exception):
    """Raised on 429 to trigger tenacity retry; not raised for other HTTP errors."""


class MoonshotAnalysisClient(BaseAnalysisClient):
    """Live analysis client using the Moonshot AI (Kimi) OpenAI-compatible API."""

    client_name = "moonshot_analysis_client"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str = "moonshot-v1-128k",
        max_tokens: int = 4096,
        temperature: float = 0.2,
        base_url: str = "https://api.moonshot.ai/v1",
    ):
        self._api_key = api_key or MOONSHOT_API_KEY or os.environ.get("MOONSHOT_API_KEY", "")
        if not self._api_key:
            raise ValueError("MOONSHOT_API_KEY must be set for Moonshot analysis.")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._base_url = base_url
        self._http = httpx.Client(timeout=180.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(_MoonshotRateLimitError),
    )
    def _call_api(self, system_prompt: str, user_message: str) -> str:
        """Call Moonshot chat completions API with retry only on 429 rate limits."""
        response = self._http.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        if response.status_code == 429:
            logger.warning("Moonshot rate limit hit (429), will retry...")
            raise _MoonshotRateLimitError(f"Rate limited (429): {response.text[:200]}")
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _extract_json(self, raw_text: str) -> dict[str, Any] | None:
        """Parse JSON from LLM response -- same 3-strategy parser."""
        text = raw_text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass
        return None

    def run_analysis(self, request: AnalysisClientRequest) -> AnalysisClientResponse:
        """Run one analysis request against Moonshot AI and parse the structured response."""
        system_prompt = request.prompt_context.get("analysis_instructions", "")
        if not system_prompt:
            system_prompt = "You are a biotech disclosure analysis worker. Analyze the provided document and return structured JSON output."

        user_message = request.prompt_text or json.dumps(
            request.prompt_context, default=str, indent=2
        )

        try:
            raw_text = self._call_api(system_prompt, user_message)
        except (httpx.HTTPStatusError, _MoonshotRateLimitError, Exception) as exc:
            logger.error("Moonshot API error: %s", exc)
            return AnalysisClientResponse(
                client_name=self.client_name,
                model_name=self.model_name,
                status=ProcessingStatus.ANALYSIS_FAILED,
                raw_text=str(exc),
                warnings=[
                    make_analysis_warning(
                        "moonshot_api_error",
                        f"Moonshot API call failed: {exc}",
                    )
                ],
                is_mock=False,
            )

        parsed = self._extract_json(raw_text)
        if parsed is None:
            logger.warning("Could not parse JSON from Moonshot response. Returning raw text.")
            return AnalysisClientResponse(
                client_name=self.client_name,
                model_name=self.model_name,
                status=ProcessingStatus.PARTIAL,
                raw_output=None,
                raw_text=raw_text,
                warnings=[
                    make_analysis_warning(
                        "json_parse_failed",
                        "LLM response could not be parsed as JSON. Raw text preserved.",
                    )
                ],
                is_mock=False,
            )

        return AnalysisClientResponse(
            client_name=self.client_name,
            model_name=self.model_name,
            status=ProcessingStatus.SUCCESS,
            raw_output=parsed,
            raw_text=raw_text,
            is_mock=False,
        )


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def build_analysis_client(
    config: PipelineConfig,
    preferred_client: BaseAnalysisClient | None = None,
) -> BaseAnalysisClient:
    """Build the default shared analysis client for the current notebook configuration."""
    if preferred_client is not None:
        return preferred_client
    return MoonshotAnalysisClient(model_name=config.worker_model_name or "moonshot-v1-128k")
