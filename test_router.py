import asyncio
from enum import Enum

import pytest

from penrs_router import DOCUMENT_API_ROUTING, DocumentType, penrs_fetch_document


def test_document_type_is_strict_enum_with_nine_values():
    assert issubclass(DocumentType, Enum)
    assert len(DocumentType) == 9
    assert set(DOCUMENT_API_ROUTING.keys()) == set(DocumentType)
    assert all(isinstance(api, str) and api for apis in DOCUMENT_API_ROUTING.values() for api in apis)


def test_penrs_fetch_document_requires_document_type_enum():
    with pytest.raises(TypeError):
        asyncio.run(
            penrs_fetch_document(
                ticker="MRNA",
                document_type="sec_10q",  # type: ignore[arg-type]
                date_range=("2026-01-01", "2026-02-01"),
                fetchers={},
            )
        )


def test_penrs_fetch_document_aggregates_multi_source_results():
    async def openfda_fetcher(ticker, date_range, document_type):
        assert ticker == "MRNA"
        assert date_range == ("2026-01-01", "2026-02-01")
        assert document_type is DocumentType.BIOMEDICAL_EVIDENCE
        return {"status": "available", "data": {"events": 3}}

    async def pubmed_fetcher(_ticker, _date_range, _document_type):
        return {"status": "available", "data": {"papers": 7}}

    result = asyncio.run(
        penrs_fetch_document(
            ticker="MRNA",
            document_type=DocumentType.BIOMEDICAL_EVIDENCE,
            date_range=("2026-01-01", "2026-02-01"),
            fetchers={
                "openfda": openfda_fetcher,
                "pubmed": pubmed_fetcher,
            },
        )
    )

    assert result["status"] == "available"
    assert result["data"]["apis_attempted"] == ["openfda", "pubmed"]
    assert result["data"]["sources"] == [
        {"api": "openfda", "data": {"events": 3}},
        {"api": "pubmed", "data": {"papers": 7}},
    ]


def test_penrs_fetch_document_returns_not_released_with_attempted_apis():
    async def sec_fetcher(_ticker, _date_range, _document_type):
        return {"status": "not_released"}

    result = asyncio.run(
        penrs_fetch_document(
            ticker="BIIB",
            document_type=DocumentType.SEC_10Q,
            date_range=("2026-01-01", "2026-02-01"),
            fetchers={"sec_edgar": sec_fetcher},
        )
    )

    assert result["status"] == "not_released"
    assert result["data"]["apis_attempted"] == ["sec_edgar"]


def test_penrs_fetch_document_handles_partial_failures():
    async def openfda_fetcher(_ticker, _date_range, _document_type):
        return {"status": "available", "data": {"events": 1}}

    async def pubmed_fetcher(_ticker, _date_range, _document_type):
        raise RuntimeError("pubmed outage")

    result = asyncio.run(
        penrs_fetch_document(
            ticker="SAVA",
            document_type=DocumentType.BIOMEDICAL_EVIDENCE,
            date_range=("2026-01-01", "2026-02-01"),
            fetchers={
                "openfda": openfda_fetcher,
                "pubmed": pubmed_fetcher,
            },
        )
    )

    assert result["status"] == "available"
    assert result["data"]["sources"] == [{"api": "openfda", "data": {"events": 1}}]
    assert result["data"]["partial_failures"] == [{"api": "pubmed", "error": "pubmed outage"}]
