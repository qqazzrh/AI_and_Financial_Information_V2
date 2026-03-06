from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

from tests.test_support import import_penrs_server


class FetcherCacheIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="penrs_fetcher_integration_")).resolve()
        self.cache_dir = self.tmpdir / "cache"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _load_cached_record(
        self,
        *,
        server,
        api: str,
        ticker: str,
        doc_type: str,
        date: str | None,
    ) -> dict:
        material = f"{api}|{ticker}|{doc_type}|{date or ''}"
        cache_key = hashlib.sha256(material.encode("utf-8")).hexdigest()
        path = server.PENRS_CACHE_DIR / f"{cache_key}.json"
        self.assertTrue(path.exists(), f"Expected cache file to exist: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    async def test_spec_1_2_alpha_vantage_success_writes_immutable_ground_truth_to_disk(self):
        with patch.dict(os.environ, {"PENRS_CACHE_DIR": str(self.cache_dir)}, clear=False):
            server = import_penrs_server(force_reload=True)
        payload = {"series": [{"close": 100.0}]}

        with patch.object(server, "_api_request", AsyncMock(return_value=payload)):
            result = await server.fetch_alpha_vantage("MRNA", "TIME_SERIES_DAILY", "2026-01-01")

        self.assertEqual(result, payload)
        record = self._load_cached_record(
            server=server,
            api="alpha_vantage",
            ticker="MRNA",
            doc_type="TIME_SERIES_DAILY",
            date="2026-01-01",
        )
        self.assertEqual(record["_api"], "alpha_vantage")
        self.assertEqual(record["_ticker"], "MRNA")
        self.assertEqual(record["_doc_type"], "TIME_SERIES_DAILY")
        self.assertEqual(record["_date"], "2026-01-01")
        self.assertEqual(record["payload"], payload)
        self.assertIn("_cached_at", record)

    async def test_spec_1_3_sec_edgar_success_writes_expected_cache_record(self):
        with patch.dict(os.environ, {"PENRS_CACHE_DIR": str(self.cache_dir)}, clear=False):
            server = import_penrs_server(force_reload=True)
        payload = {"filing": "<html>10Q</html>"}

        with patch.object(server, "_api_request", AsyncMock(return_value=payload)):
            result = await server.fetch_sec_edgar("MRNA", "0000123456", "10q.htm")

        self.assertEqual(result, payload)
        record = self._load_cached_record(
            server=server,
            api="sec_edgar",
            ticker="MRNA",
            doc_type="filing",
            date=None,
        )
        self.assertEqual(record["_api"], "sec_edgar")
        self.assertEqual(record["_ticker"], "MRNA")
        self.assertEqual(record["_doc_type"], "filing")
        self.assertIsNone(record["_date"])
        self.assertEqual(record["payload"], payload)

    async def test_spec_1_4_openfda_success_writes_expected_cache_record(self):
        with patch.dict(os.environ, {"PENRS_CACHE_DIR": str(self.cache_dir)}, clear=False):
            server = import_penrs_server(force_reload=True)
        payload = {"results": [{"id": "abc"}]}

        with patch.object(server, "_api_request", AsyncMock(return_value=payload)):
            result = await server.fetch_openfda("BIIB", limit=3)

        self.assertEqual(result, payload)
        record = self._load_cached_record(
            server=server,
            api="openfda",
            ticker="BIIB",
            doc_type="adverse_events",
            date=None,
        )
        self.assertEqual(record["_api"], "openfda")
        self.assertEqual(record["_ticker"], "BIIB")
        self.assertEqual(record["_doc_type"], "adverse_events")
        self.assertEqual(record["payload"], payload)

    async def test_spec_1_5_pubmed_success_writes_normalized_cache_record(self):
        with patch.dict(os.environ, {"PENRS_CACHE_DIR": str(self.cache_dir)}, clear=False):
            server = import_penrs_server(force_reload=True)
        payload = {"esearchresult": {"idlist": ["1"]}}

        with patch.object(server, "_api_request", AsyncMock(return_value=payload)):
            result = await server.fetch_pubmed("multiple sclerosis", retmax=1)

        self.assertEqual(result, payload)
        record = self._load_cached_record(
            server=server,
            api="pubmed",
            ticker="multiple_sclerosis",
            doc_type="publications",
            date=None,
        )
        self.assertEqual(record["_api"], "pubmed")
        self.assertEqual(record["_ticker"], "multiple_sclerosis")
        self.assertEqual(record["_doc_type"], "publications")
        self.assertEqual(record["payload"], payload)

    async def test_error_responses_do_not_write_cache_files(self):
        with patch.dict(os.environ, {"PENRS_CACHE_DIR": str(self.cache_dir)}, clear=False):
            server = import_penrs_server(force_reload=True)

        with patch.object(server, "_api_request", AsyncMock(return_value={"error": "upstream"})):
            result = await server.fetch_openfda("MRNA", limit=2)

        self.assertEqual(result, {"error": "upstream"})
        self.assertFalse(any(server.PENRS_CACHE_DIR.glob("*.json")))


if __name__ == "__main__":
    unittest.main()
