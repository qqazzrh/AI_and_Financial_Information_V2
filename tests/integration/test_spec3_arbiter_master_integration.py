from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import utils
from tests.test_support import import_orchestrator, import_worker_nodes
from utils import DocumentType


class Spec3MasterIntegrationTests(unittest.TestCase):
    def test_master_synthesize_maps_evidence_with_derived_cache_keys(self):
        mod = import_orchestrator(force_reload=True)
        master = mod.MasterAgent()

        worker_results = [
            {
                "status": "available",
                "worker": {"name": "Clinical Signals", "weight": 1.0, "signal_density": 0.8},
                "document_type": "earnings_call",
                "result": {
                    "score": 0.3,
                    "evidence_nodes": [
                        {"verbatim_quote": "trial enrollment improved", "reasoning": "quoted"},
                        "invalid-node",
                    ],
                },
            },
            {
                "status": "available",
                "worker": {"name": "   ", "weight": 1.0, "signal_density": 0.8},
                "result": {
                    "score": -0.1,
                    "evidence_nodes": [{"verbatim_quote": "cash burn narrowed", "reasoning": "quoted"}],
                },
            },
            {
                "status": "error",
                "worker": {"name": "Should Skip", "weight": 1.0, "signal_density": 0.8},
                "document_type": "press_release",
                "result": {
                    "score": 0.8,
                    "evidence_nodes": [{"verbatim_quote": "must be ignored", "reasoning": "not available"}],
                },
            },
        ]

        synthesized = master.synthesize(
            ticker="MRNA",
            date_from="2026-01-01",
            date_to="2026-02-01",
            worker_results=worker_results,
            arbiter_result={"weighted_score": 0.42},
        )

        evidence = synthesized["evidence"]
        self.assertEqual(len(evidence), 2)

        self.assertEqual(
            evidence[0]["cache_key"],
            utils.cache_key(api="clinical_signals", ticker="MRNA", doc_type="earnings_call"),
        )
        self.assertEqual(
            evidence[1]["cache_key"],
            utils.cache_key(api="unknown_worker", ticker="MRNA", doc_type="unknown_doc_type"),
        )
        self.assertEqual(synthesized["available_worker_count"], 2)
        self.assertEqual(synthesized["total_worker_count"], 3)


class Spec3RunPenrsIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_penrs_hoists_master_evidence_to_top_level_and_saved_report(self):
        mod = import_orchestrator(force_reload=True)

        class StubWorker:
            name = "Evidence Worker"
            weight = 1.0
            signal_density = 0.9

            async def run(self, ticker: str, date_from: str, date_to: str):
                return {
                    "status": "available",
                    "ticker": ticker,
                    "date_from": date_from,
                    "date_to": date_to,
                    "worker": {"name": self.name, "weight": self.weight, "signal_density": self.signal_density},
                    "document_type": "news_sentiment",
                    "result": {
                        "score": 0.4,
                        "thesis": "Supported by evidence",
                        "evidence_nodes": [
                            {"verbatim_quote": "management reaffirmed guidance", "reasoning": "explicit statement"}
                        ],
                    },
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            report = await mod.run_penrs(
                "BIIB",
                "2026-01-01",
                "2026-02-01",
                workers=[StubWorker()],
                report_dir=tmp_dir,
            )

            self.assertIn("evidence", report)
            self.assertEqual(report["evidence"], report["master"]["evidence"])
            self.assertEqual(len(report["evidence"]), 1)

            report_path = Path(report["report_path"])
            self.assertTrue(report_path.exists())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["evidence"], payload["master"]["evidence"])
            self.assertEqual(payload["evidence"], report["evidence"])

    async def test_run_penrs_sets_arbiter_error_when_worker_has_non_neutral_score_without_evidence(self):
        mod = import_orchestrator(force_reload=True)

        class InvalidWorker:
            name = "Invalid Worker"
            weight = 1.0
            signal_density = 0.7

            async def run(self, ticker: str, date_from: str, date_to: str):
                return {
                    "status": "available",
                    "ticker": ticker,
                    "date_from": date_from,
                    "date_to": date_to,
                    "worker": {"name": self.name, "weight": self.weight, "signal_density": self.signal_density},
                    "document_type": "news_sentiment",
                    "result": {"score": 0.55, "thesis": "No evidence attached."},
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            report = await mod.run_penrs(
                "BIIB",
                "2026-01-01",
                "2026-02-01",
                workers=[InvalidWorker()],
                report_dir=tmp_dir,
            )

            self.assertEqual(report["arbiter"]["status"], "error")
            self.assertIn("Non-neutral worker score requires validated evidence_nodes", report["arbiter"]["error"])
            self.assertEqual(report["arbiter"]["weighted_score"], 0.0)
            self.assertEqual(report["master"]["final_score"], 0.0)
            self.assertEqual(report["master"]["evidence"], [])
            self.assertEqual(report["evidence"], [])

    async def test_run_penrs_with_real_worker_neutralizes_hallucinated_quotes_before_arbiter(self):
        orchestrator_mod = import_orchestrator(force_reload=True)
        worker_mod = import_worker_nodes(force_reload=True)

        def fake_rubric_fetcher(_rubric_id):
            return {"criteria": "integration-hallucination-destruction"}

        async def fake_document_fetcher(_ticker, _doc_type, _date_range):
            return {"status": "available", "data": "Validated excerpt has no fabricated quote."}

        def fake_llm_invoker(_prompt, *, system):
            _ = system
            return json.dumps(
                {
                    "score": -0.6,
                    "thesis": "Strong downside with bad citation.",
                    "evidence_nodes": [
                        {"verbatim_quote": "fabricated quote text", "reasoning": "hallucinated"},
                    ],
                }
            )

        worker = worker_mod.PENRSWorker(
            name="Quote Validator",
            weight=1.0,
            signal_density=0.9,
            rubric_id="r1",
            document_type=DocumentType.NEWS_SENTIMENT,
            rubric_fetcher=fake_rubric_fetcher,
            document_fetcher=fake_document_fetcher,
            llm_invoker=fake_llm_invoker,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            report = await orchestrator_mod.run_penrs(
                "MRNA",
                "2026-01-01",
                "2026-02-01",
                workers=[worker],
                report_dir=tmp_dir,
            )

            worker_result = report["worker_results"][0]["result"]
            self.assertEqual(worker_result["score"], 0.0)
            self.assertEqual(worker_result["evidence_nodes"], [])
            self.assertIn("Score neutralized due to hallucinated evidence.", worker_result["thesis"])
            self.assertEqual(report["arbiter"]["status"], "available")
            self.assertEqual(report["arbiter"]["weighted_score"], 0.0)
            self.assertEqual(report["master"]["evidence"], [])


if __name__ == "__main__":
    unittest.main()
