from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.test_support import import_penrs_server


class FetcherUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_spec_1_1_required_imports_and_mcp_tool_registration_present(self):
        server = import_penrs_server(force_reload=True)

        self.assertTrue(hasattr(server, "os"))
        self.assertTrue(hasattr(server, "logging"))
        self.assertTrue(hasattr(server, "FastMCP"))
        self.assertTrue(hasattr(server, "_api_request"))
        self.assertTrue(hasattr(server, "cache_set"))
        self.assertTrue(hasattr(server, "PENRS_CACHE_DIR"))
        self.assertEqual(server.mcp.name, "penrs_mcp")

        for tool_name in (
            "fetch_alpha_vantage",
            "fetch_sec_edgar",
            "fetch_openfda",
            "fetch_pubmed",
        ):
            self.assertTrue(getattr(server, tool_name)._is_mcp_tool)

    async def test_spec_1_2_alpha_vantage_success_routes_and_caches(self):
        server = import_penrs_server(force_reload=True)
        result_payload = {"ok": True}

        with patch.object(server, "_api_request", AsyncMock(return_value=result_payload)) as api_mock, patch.object(
            server, "cache_set", Mock()
        ) as cache_mock:
            result = await server.fetch_alpha_vantage("MRNA", "TIME_SERIES_DAILY", "2026-01-01")

        self.assertEqual(result, result_payload)
        api_mock.assert_awaited_once_with(
            "https://www.alphavantage.co/query",
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": "MRNA",
                "apikey": server.ALPHA_VANTAGE_API_KEY,
            },
            api_name="alpha_vantage",
        )
        cache_mock.assert_called_once_with(
            api="alpha_vantage",
            ticker="MRNA",
            doc_type="TIME_SERIES_DAILY",
            date="2026-01-01",
            payload=result_payload,
        )

    async def test_spec_1_2_alpha_vantage_error_skips_cache(self):
        server = import_penrs_server(force_reload=True)
        error_payload = {"error": "upstream failure"}

        with patch.object(server, "_api_request", AsyncMock(return_value=error_payload)) as api_mock, patch.object(
            server, "cache_set", Mock()
        ) as cache_mock:
            result = await server.fetch_alpha_vantage("MRNA", "TIME_SERIES_DAILY")

        self.assertEqual(result, error_payload)
        api_mock.assert_awaited_once()
        cache_mock.assert_not_called()

    async def test_spec_1_3_sec_edgar_success_routes_headers_and_caches(self):
        server = import_penrs_server(force_reload=True)
        result_payload = {"filing": "raw text"}

        with patch.object(server, "_api_request", AsyncMock(return_value=result_payload)) as api_mock, patch.object(
            server, "cache_set", Mock()
        ) as cache_mock:
            result = await server.fetch_sec_edgar("MRNA", "0000123456", "10q.htm")

        self.assertEqual(result, result_payload)
        api_mock.assert_awaited_once_with(
            "https://www.sec.gov/Archives/edgar/data/MRNA/0000123456/10q.htm",
            headers={"User-Agent": server.SEC_USER_AGENT},
            api_name="sec_edgar",
        )
        cache_mock.assert_called_once_with(
            api="sec_edgar",
            ticker="MRNA",
            doc_type="filing",
            date=None,
            payload=result_payload,
        )

    async def test_spec_1_3_sec_edgar_error_skips_cache(self):
        server = import_penrs_server(force_reload=True)
        error_payload = {"error": "forbidden"}

        with patch.object(server, "_api_request", AsyncMock(return_value=error_payload)), patch.object(
            server, "cache_set", Mock()
        ) as cache_mock:
            result = await server.fetch_sec_edgar("MRNA", "0000123456", "10q.htm")

        self.assertEqual(result, error_payload)
        cache_mock.assert_not_called()

    async def test_spec_1_4_openfda_without_api_key_routes_and_caches(self):
        server = import_penrs_server(force_reload=True)
        result_payload = {"results": []}

        with patch.object(server, "OPENFDA_API_KEY", None), patch.object(
            server, "_api_request", AsyncMock(return_value=result_payload)
        ) as api_mock, patch.object(server, "cache_set", Mock()) as cache_mock:
            result = await server.fetch_openfda("MRNA", limit=7)

        self.assertEqual(result, result_payload)
        api_mock.assert_awaited_once_with(
            "https://api.fda.gov/drug/event.json",
            params={"search": "patient.drug.medicinalproduct:MRNA", "limit": 7},
            api_name="openfda",
        )
        cache_mock.assert_called_once_with(
            api="openfda",
            ticker="MRNA",
            doc_type="adverse_events",
            date=None,
            payload=result_payload,
        )

    async def test_spec_1_4_openfda_includes_api_key_when_present(self):
        server = import_penrs_server(force_reload=True)
        result_payload = {"results": []}

        with patch.object(server, "OPENFDA_API_KEY", "openfda-key"), patch.object(
            server, "_api_request", AsyncMock(return_value=result_payload)
        ) as api_mock:
            await server.fetch_openfda("BIIB")

        self.assertEqual(api_mock.await_args.kwargs["params"]["api_key"], "openfda-key")
        self.assertEqual(api_mock.await_args.kwargs["params"]["limit"], 10)

    async def test_spec_1_4_openfda_error_skips_cache(self):
        server = import_penrs_server(force_reload=True)
        error_payload = {"error": "upstream"}

        with patch.object(server, "_api_request", AsyncMock(return_value=error_payload)), patch.object(
            server, "cache_set", Mock()
        ) as cache_mock:
            result = await server.fetch_openfda("MRNA")

        self.assertEqual(result, error_payload)
        cache_mock.assert_not_called()

    async def test_spec_1_5_pubmed_without_api_key_routes_and_caches(self):
        server = import_penrs_server(force_reload=True)
        result_payload = {"esearchresult": {"idlist": ["1", "2"]}}

        with patch.object(server, "NCBI_API_KEY", None), patch.object(
            server, "_api_request", AsyncMock(return_value=result_payload)
        ) as api_mock, patch.object(server, "cache_set", Mock()) as cache_mock:
            result = await server.fetch_pubmed("multiple sclerosis", retmax=2)

        self.assertEqual(result, result_payload)
        api_mock.assert_awaited_once_with(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": "multiple sclerosis", "retmode": "json", "retmax": 2},
            api_name="pubmed",
        )
        cache_mock.assert_called_once_with(
            api="pubmed",
            ticker="multiple_sclerosis",
            doc_type="publications",
            date=None,
            payload=result_payload,
        )

    async def test_spec_1_5_pubmed_includes_api_key_when_present(self):
        server = import_penrs_server(force_reload=True)
        result_payload = {"esearchresult": {"idlist": []}}

        with patch.object(server, "NCBI_API_KEY", "ncbi-key"), patch.object(
            server, "_api_request", AsyncMock(return_value=result_payload)
        ) as api_mock:
            await server.fetch_pubmed("glioblastoma")

        self.assertEqual(api_mock.await_args.kwargs["params"]["api_key"], "ncbi-key")
        self.assertEqual(api_mock.await_args.kwargs["params"]["retmax"], 5)

    async def test_spec_1_5_pubmed_error_skips_cache(self):
        server = import_penrs_server(force_reload=True)
        error_payload = {"error": "failure"}

        with patch.object(server, "_api_request", AsyncMock(return_value=error_payload)), patch.object(
            server, "cache_set", Mock()
        ) as cache_mock:
            result = await server.fetch_pubmed("glioblastoma")

        self.assertEqual(result, error_payload)
        cache_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
