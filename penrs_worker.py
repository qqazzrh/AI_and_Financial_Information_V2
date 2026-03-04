import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from penrs_router import DocumentType, penrs_fetch_document

RubricFetcher = Callable[[str], Any | Awaitable[Any]]
DocumentFetcher = Callable[[str, DocumentType, dict[str, str] | None], Any | Awaitable[Any]]
LLMInvoker = Callable[[str], Any | Awaitable[Any]]

_DEFAULT_RUBRICS_PATH = Path("rubrics.json")
_TRUNCATION_MARKER_TEMPLATE = "\n...[truncated {count} chars]...\n"


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _load_rubric_from_json(rubric_id: str) -> dict[str, Any]:
    if not _DEFAULT_RUBRICS_PATH.exists():
        return {"rubric_id": rubric_id, "error": "rubrics.json not found"}

    try:
        payload = json.loads(_DEFAULT_RUBRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"rubric_id": rubric_id, "error": f"Unable to read rubrics.json: {exc}"}

    if isinstance(payload, dict):
        rubric = payload.get(rubric_id)
        if isinstance(rubric, dict):
            return rubric
        return {"rubric_id": rubric_id, "error": f"Rubric '{rubric_id}' not found"}

    return {"rubric_id": rubric_id, "error": "rubrics.json must contain an object"}


def _coerce_text(document_payload: Any) -> str:
    if isinstance(document_payload, str):
        return document_payload
    if isinstance(document_payload, bytes):
        return document_payload.decode("utf-8", errors="replace")
    if document_payload is None:
        return ""
    if isinstance(document_payload, (dict, list, tuple)):
        return json.dumps(document_payload, ensure_ascii=True, sort_keys=True)
    return str(document_payload)


def truncate_for_context(text: str, max_chars: int = 12000) -> str:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if len(text) <= max_chars:
        return text

    truncated_count = len(text) - max_chars
    marker = _TRUNCATION_MARKER_TEMPLATE.format(count=truncated_count)
    if len(marker) >= max_chars:
        return text[:max_chars]

    remaining = max_chars - len(marker)
    head_len = remaining // 2
    tail_len = remaining - head_len
    return f"{text[:head_len]}{marker}{text[-tail_len:]}"


def _extract_json_from_text(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[{\[]", text):
        start = match.start()
        try:
            parsed, _end = decoder.raw_decode(text[start:])
            return parsed
        except json.JSONDecodeError:
            continue
    return None


class PENRSWorker:
    def __init__(
        self,
        *,
        name: str,
        weight: float,
        signal_density: float,
        rubric_id: str,
        document_type: DocumentType,
        rubric_fetcher: RubricFetcher | None = None,
        document_fetcher: DocumentFetcher | None = None,
        llm_invoker: LLMInvoker | None = None,
        max_context_chars: int = 12000,
    ) -> None:
        self.name = name
        self.weight = weight
        self.signal_density = signal_density
        self.rubric_id = rubric_id
        self.document_type = document_type
        self.rubric_fetcher = rubric_fetcher or _load_rubric_from_json
        self.document_fetcher = document_fetcher or penrs_fetch_document
        self.llm_invoker = llm_invoker or (lambda _prompt: "{}")
        self.max_context_chars = max_context_chars

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "signal_density": self.signal_density,
        }

    def parse_json_response(self, response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return response
        if isinstance(response, list):
            return {"items": response}
        if response is None:
            return {"parse_error": "empty_response", "raw_response": ""}

        text = str(response).strip()
        if not text:
            return {"parse_error": "empty_response", "raw_response": ""}

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return {"items": parsed}
        except json.JSONDecodeError:
            pass

        fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if fenced_match:
            fenced_payload = fenced_match.group(1).strip()
            try:
                parsed = json.loads(fenced_payload)
                if isinstance(parsed, dict):
                    return parsed
                return {"items": parsed}
            except json.JSONDecodeError:
                pass

        extracted = _extract_json_from_text(text)
        if extracted is not None:
            if isinstance(extracted, dict):
                return extracted
            return {"items": extracted}

        return {"parse_error": "unable_to_parse_json", "raw_response": text}

    def build_prompt(
        self,
        *,
        ticker: str,
        date_from: str,
        date_to: str,
        rubric: dict[str, Any],
        document_excerpt: str,
    ) -> str:
        return (
            f"Worker: {self.name}\n"
            f"Ticker: {ticker}\n"
            f"Date range: {date_from} -> {date_to}\n"
            f"Rubric JSON: {json.dumps(rubric, ensure_ascii=True)}\n"
            f"Document excerpt:\n{document_excerpt}\n"
            "Return only JSON."
        )

    async def run(self, ticker: str, date_from: str, date_to: str) -> dict[str, Any]:
        rubric_raw = await _maybe_await(self.rubric_fetcher(self.rubric_id))
        rubric = rubric_raw if isinstance(rubric_raw, dict) else {"rubric": rubric_raw}

        doc_result_raw = await _maybe_await(
            self.document_fetcher(
                ticker,
                self.document_type,
                {"from": date_from, "to": date_to},
            )
        )
        doc_result = doc_result_raw if isinstance(doc_result_raw, dict) else {"status": "not_released", "data": {}}

        if doc_result.get("status") == "not_released":
            data = doc_result.get("data", {})
            return {
                "status": "not_released",
                "worker": self.metadata,
                "ticker": ticker,
                "date_from": date_from,
                "date_to": date_to,
                "document_type": self.document_type.value,
                "apis_attempted": list(data.get("apis_attempted", [])),
                "detail": data,
            }

        document_data = doc_result.get("data", {})
        excerpt = truncate_for_context(_coerce_text(document_data), max_chars=self.max_context_chars)
        prompt = self.build_prompt(
            ticker=ticker,
            date_from=date_from,
            date_to=date_to,
            rubric=rubric,
            document_excerpt=excerpt,
        )

        llm_raw = await _maybe_await(self.llm_invoker(prompt))
        parsed = self.parse_json_response(llm_raw)
        return {
            "status": "available",
            "worker": self.metadata,
            "ticker": ticker,
            "date_from": date_from,
            "date_to": date_to,
            "document_type": self.document_type.value,
            "rubric": rubric,
            "result": parsed,
        }
