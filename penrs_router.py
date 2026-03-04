import asyncio
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping


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
