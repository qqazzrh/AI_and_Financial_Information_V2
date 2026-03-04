import asyncio

from penrs_router import DocumentType
from penrs_worker import PENRSWorker, truncate_for_context


def test_truncate_for_context_preserves_start_and_end():
    source = "A" * 80 + "MIDDLE" + "Z" * 80
    truncated = truncate_for_context(source, max_chars=60)

    assert len(truncated) == 60
    assert truncated.startswith("A")
    assert truncated.endswith("Z")
    assert "[truncated" in truncated


def test_parse_json_response_handles_markdown_block():
    worker = PENRSWorker(
        name="TestWorker",
        weight=1.0,
        signal_density=0.5,
        rubric_id="worker_test",
        document_type=DocumentType.SEC_10Q,
    )
    response = "analysis:\n```json\n{\"score\": 0.7, \"summary\": \"ok\"}\n```\nextra"

    parsed = worker.parse_json_response(response)

    assert parsed == {"score": 0.7, "summary": "ok"}


def test_parse_json_response_handles_embedded_prose_and_invalid_json_fallback():
    worker = PENRSWorker(
        name="TestWorker",
        weight=1.0,
        signal_density=0.5,
        rubric_id="worker_test",
        document_type=DocumentType.SEC_10Q,
    )
    prose_response = "Before note. {\"signal\": \"bearish\", \"confidence\": 0.81} After note."
    invalid_response = "I could not determine anything with confidence."

    parsed_from_prose = worker.parse_json_response(prose_response)
    parsed_invalid = worker.parse_json_response(invalid_response)

    assert parsed_from_prose == {"signal": "bearish", "confidence": 0.81}
    assert parsed_invalid["parse_error"] == "unable_to_parse_json"
    assert parsed_invalid["raw_response"] == invalid_response


def test_run_fetches_rubric_and_document_and_enriches_metadata():
    calls = {"rubric": 0, "document": 0, "prompt": None}

    async def fake_rubric_fetcher(rubric_id):
        calls["rubric"] += 1
        assert rubric_id == "worker_earnings"
        return {"name": "Earnings rubric", "threshold": 0.6}

    async def fake_document_fetcher(ticker, document_type, date_range):
        calls["document"] += 1
        assert ticker == "MRNA"
        assert document_type is DocumentType.EARNINGS_CALL
        assert date_range == {"from": "2026-01-01", "to": "2026-02-01"}
        return {
            "status": "available",
            "data": {
                "ticker": ticker,
                "sources": [{"api": "alpha_vantage", "data": {"transcript": "A" * 300}}],
            },
        }

    async def fake_llm(prompt):
        calls["prompt"] = prompt
        return "```json\n{\"score\": 0.42, \"thesis\": \"mixed\"}\n```"

    worker = PENRSWorker(
        name="EarningsWorker",
        weight=1.2,
        signal_density=0.75,
        rubric_id="worker_earnings",
        document_type=DocumentType.EARNINGS_CALL,
        rubric_fetcher=fake_rubric_fetcher,
        document_fetcher=fake_document_fetcher,
        llm_invoker=fake_llm,
        max_context_chars=120,
    )

    result = asyncio.run(worker.run("MRNA", "2026-01-01", "2026-02-01"))

    assert calls["rubric"] == 1
    assert calls["document"] == 1
    assert "Document excerpt:" in calls["prompt"]
    assert result["status"] == "available"
    assert result["worker"] == {
        "name": "EarningsWorker",
        "weight": 1.2,
        "signal_density": 0.75,
    }
    assert result["result"] == {"score": 0.42, "thesis": "mixed"}


def test_run_handles_not_released_without_llm_call():
    calls = {"llm": 0}

    async def fake_document_fetcher(_ticker, _document_type, _date_range):
        return {
            "status": "not_released",
            "data": {
                "apis_attempted": ["sec_edgar"],
                "reason": "Filing not yet published",
            },
        }

    async def fake_llm(_prompt):
        calls["llm"] += 1
        return {"score": 0.0}

    worker = PENRSWorker(
        name="SECFilingWorker",
        weight=1.0,
        signal_density=0.4,
        rubric_id="worker_sec",
        document_type=DocumentType.SEC_10Q,
        rubric_fetcher=lambda _rubric_id: {"stub": True},
        document_fetcher=fake_document_fetcher,
        llm_invoker=fake_llm,
    )

    result = asyncio.run(worker.run("BIIB", "2026-01-01", "2026-02-01"))

    assert result["status"] == "not_released"
    assert result["apis_attempted"] == ["sec_edgar"]
    assert result["worker"]["name"] == "SECFilingWorker"
    assert calls["llm"] == 0
