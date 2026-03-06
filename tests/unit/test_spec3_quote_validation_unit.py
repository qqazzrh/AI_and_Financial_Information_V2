from __future__ import annotations

import json
import unittest

from tests.test_support import import_orchestrator, import_worker_nodes
from utils import DocumentType


SYSTEM_NOTE = "[SYSTEM NOTE: Score neutralized due to hallucinated evidence.]"


class Spec3WorkerHallucinationUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_prunes_hallucinated_nodes_and_keeps_valid_quotes(self):
        mod = import_worker_nodes(force_reload=True)

        def fake_rubric_fetcher(_rubric_id):
            return {"criteria": "quote-validation"}

        async def fake_document_fetcher(_ticker, _doc_type, _date_range):
            return {
                "status": "available",
                "data": "Revenue grew sequentially and guidance was reaffirmed for FY26.",
            }

        llm_payload = {
            "score": 0.6,
            "thesis": "Mostly constructive update.",
            "evidence_nodes": [
                {"verbatim_quote": "guidance was reaffirmed", "reasoning": "Directly in excerpt."},
                {"verbatim_quote": "pipeline discontinued", "reasoning": "Hallucinated."},
                {"reasoning": "Missing quote key."},
                "not-a-dict",
            ],
        }

        def fake_llm_invoker(_prompt, *, system):
            _ = system
            return json.dumps(llm_payload)

        worker = mod.PENRSWorker(
            name="Validation Worker",
            weight=1.0,
            signal_density=1.0,
            rubric_id="test_rubric",
            document_type=DocumentType.NEWS_SENTIMENT,
            rubric_fetcher=fake_rubric_fetcher,
            document_fetcher=fake_document_fetcher,
            llm_invoker=fake_llm_invoker,
        )

        result = await worker.run("TEST", "2026-01-01", "2026-02-01")

        self.assertEqual(result["result"]["score"], 0.6)
        self.assertEqual(
            result["result"]["evidence_nodes"],
            [{"verbatim_quote": "guidance was reaffirmed", "reasoning": "Directly in excerpt."}],
        )
        self.assertNotIn(SYSTEM_NOTE, result["result"]["thesis"])

    async def test_neutralizes_score_and_appends_system_note_when_all_nodes_are_hallucinated(self):
        mod = import_worker_nodes(force_reload=True)

        def fake_rubric_fetcher(_rubric_id):
            return {"criteria": "neutralize"}

        async def fake_document_fetcher(_ticker, _doc_type, _date_range):
            return {"status": "available", "data": "Only this sentence is real."}

        def fake_llm_invoker(_prompt, *, system):
            _ = system
            return json.dumps(
                {
                    "score": -0.8,
                    "thesis": "High conviction downside.",
                    "evidence_nodes": [
                        {"verbatim_quote": "fabricated phrase", "reasoning": "Not present."},
                    ],
                }
            )

        worker = mod.PENRSWorker(
            name="Neutralize Worker",
            weight=1.0,
            signal_density=1.0,
            rubric_id="test_rubric",
            document_type=DocumentType.NEWS_SENTIMENT,
            rubric_fetcher=fake_rubric_fetcher,
            document_fetcher=fake_document_fetcher,
            llm_invoker=fake_llm_invoker,
        )

        result = await worker.run("TEST", "2026-01-01", "2026-02-01")

        self.assertEqual(result["result"]["evidence_nodes"], [])
        self.assertEqual(result["result"]["score"], 0.0)
        self.assertIn(SYSTEM_NOTE, result["result"]["thesis"])

    async def test_system_note_is_not_duplicated_when_already_present(self):
        mod = import_worker_nodes(force_reload=True)

        def fake_rubric_fetcher(_rubric_id):
            return {"criteria": "dedupe-note"}

        async def fake_document_fetcher(_ticker, _doc_type, _date_range):
            return {"status": "available", "data": "Ground-truth excerpt."}

        def fake_llm_invoker(_prompt, *, system):
            _ = system
            return json.dumps(
                {
                    "score": 0.4,
                    "thesis": f"Original thesis. {SYSTEM_NOTE}",
                    "evidence_nodes": [{"verbatim_quote": "not in excerpt", "reasoning": "Bad quote."}],
                }
            )

        worker = mod.PENRSWorker(
            name="No Duplicate Note",
            weight=1.0,
            signal_density=1.0,
            rubric_id="test_rubric",
            document_type=DocumentType.NEWS_SENTIMENT,
            rubric_fetcher=fake_rubric_fetcher,
            document_fetcher=fake_document_fetcher,
            llm_invoker=fake_llm_invoker,
        )

        result = await worker.run("TEST", "2026-01-01", "2026-02-01")
        thesis = result["result"]["thesis"]

        self.assertEqual(result["result"]["score"], 0.0)
        self.assertEqual(result["result"]["evidence_nodes"], [])
        self.assertEqual(thesis.count(SYSTEM_NOTE), 1)

    async def test_neutralizes_non_neutral_score_when_llm_returns_empty_evidence_list(self):
        mod = import_worker_nodes(force_reload=True)

        def fake_rubric_fetcher(_rubric_id):
            return {"criteria": "empty-evidence-list"}

        async def fake_document_fetcher(_ticker, _doc_type, _date_range):
            return {"status": "available", "data": "Verified excerpt text only."}

        def fake_llm_invoker(_prompt, *, system):
            _ = system
            return json.dumps(
                {
                    "score": 0.9,
                    "thesis": "Strongly positive.",
                    "evidence_nodes": [],
                }
            )

        worker = mod.PENRSWorker(
            name="Empty Evidence Neutralizer",
            weight=1.0,
            signal_density=1.0,
            rubric_id="test_rubric",
            document_type=DocumentType.NEWS_SENTIMENT,
            rubric_fetcher=fake_rubric_fetcher,
            document_fetcher=fake_document_fetcher,
            llm_invoker=fake_llm_invoker,
        )

        result = await worker.run("TEST", "2026-01-01", "2026-02-01")

        self.assertEqual(result["result"]["evidence_nodes"], [])
        self.assertEqual(result["result"]["score"], 0.0)
        self.assertIn(SYSTEM_NOTE, result["result"]["thesis"])

    async def test_quote_validation_uses_truncated_excerpt_not_full_document(self):
        mod = import_worker_nodes(force_reload=True)

        def fake_rubric_fetcher(_rubric_id):
            return {"criteria": "truncate-context"}

        # Quote appears only after the first 40 chars.
        long_text = "A" * 40 + " QUOTE_ONLY_IN_TAIL " + "B" * 40

        async def fake_document_fetcher(_ticker, _doc_type, _date_range):
            return {"status": "available", "data": long_text}

        def fake_llm_invoker(_prompt, *, system):
            _ = system
            return json.dumps(
                {
                    "score": 0.25,
                    "thesis": "Tail quote support.",
                    "evidence_nodes": [
                        {"verbatim_quote": "QUOTE_ONLY_IN_TAIL", "reasoning": "Present in full document only."}
                    ],
                }
            )

        worker = mod.PENRSWorker(
            name="Truncation Guard",
            weight=1.0,
            signal_density=1.0,
            rubric_id="test_rubric",
            document_type=DocumentType.NEWS_SENTIMENT,
            rubric_fetcher=fake_rubric_fetcher,
            document_fetcher=fake_document_fetcher,
            llm_invoker=fake_llm_invoker,
            max_context_chars=40,
        )

        result = await worker.run("TEST", "2026-01-01", "2026-02-01")

        self.assertEqual(result["result"]["evidence_nodes"], [])
        self.assertEqual(result["result"]["score"], 0.0)
        self.assertIn(SYSTEM_NOTE, result["result"]["thesis"])


class Spec3ArbiterAndReportUnitTests(unittest.TestCase):
    def setUp(self):
        self.mod = import_orchestrator(force_reload=True)
        self.arbiter = self.mod.ArbiterAgent()

    def _base_worker_result(self):
        return {
            "status": "available",
            "worker": {"name": "w1", "weight": 1.0, "signal_density": 0.7},
            "result": {"score": 0.0, "thesis": "neutral"},
        }

    def test_arbiter_allows_neutral_score_without_evidence_nodes(self):
        worker_result = self._base_worker_result()
        self.arbiter._validate_worker_result(worker_result)

    def test_arbiter_rejects_non_neutral_score_without_evidence_nodes(self):
        worker_result = self._base_worker_result()
        worker_result["result"]["score"] = 0.2

        with self.assertRaisesRegex(ValueError, "Non-neutral worker score requires validated evidence_nodes"):
            self.arbiter._validate_worker_result(worker_result)

    def test_arbiter_rejects_non_neutral_score_with_empty_evidence_nodes(self):
        worker_result = self._base_worker_result()
        worker_result["result"]["score"] = -0.3
        worker_result["result"]["evidence_nodes"] = []

        with self.assertRaisesRegex(ValueError, "Non-neutral worker score requires validated evidence_nodes"):
            self.arbiter._validate_worker_result(worker_result)

    def test_arbiter_accepts_non_neutral_score_with_validated_evidence_nodes(self):
        worker_result = self._base_worker_result()
        worker_result["result"]["score"] = 0.5
        worker_result["result"]["evidence_nodes"] = [{"verbatim_quote": "in excerpt", "reasoning": "validated"}]

        self.arbiter._validate_worker_result(worker_result)

    def test_arbiter_rejects_non_neutral_string_score_without_evidence_nodes(self):
        worker_result = self._base_worker_result()
        worker_result["result"]["score"] = "0.2"

        with self.assertRaisesRegex(ValueError, "Non-neutral worker score requires validated evidence_nodes"):
            self.arbiter._validate_worker_result(worker_result)

    def test_penrs_report_model_exposes_evidence_field_with_default_factory(self):
        report = self.mod.PENRSReport.model_validate(
            {
                "ticker": "MRNA",
                "date_from": "2026-01-01",
                "date_to": "2026-02-01",
                "generated_at": "2026-03-06T00:00:00+00:00",
                "worker_results": [],
                "arbiter": {"status": "available"},
                "master": {"status": "available"},
                "report_path": "/tmp/r.json",
            }
        )

        self.assertEqual(report.evidence, [])

    def test_penrs_report_evidence_default_factory_not_shared_between_instances(self):
        payload = {
            "ticker": "MRNA",
            "date_from": "2026-01-01",
            "date_to": "2026-02-01",
            "generated_at": "2026-03-06T00:00:00+00:00",
            "worker_results": [],
            "arbiter": {"status": "available"},
            "master": {"status": "available"},
            "report_path": "/tmp/r.json",
        }
        report_a = self.mod.PENRSReport.model_validate(payload)
        report_b = self.mod.PENRSReport.model_validate(payload)

        report_a.evidence.append({"verbatim_quote": "q"})
        self.assertEqual(report_b.evidence, [])


if __name__ == "__main__":
    unittest.main()
