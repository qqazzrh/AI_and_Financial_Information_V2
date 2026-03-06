from __future__ import annotations

import json
import unittest

from tests.test_support import import_worker_nodes
from utils import DocumentType


class WorkerSpec2IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_spec_2_3_run_passes_anthropic_system_prompt_and_parses_response(self):
        mod = import_worker_nodes(force_reload=True)
        seen = {"prompt": None, "system": None}
        llm_payload = {
            "score": -0.9,
            "thesis": "Management is bailing out while pushing dilute and delay tactics.",
            "evidence_nodes": [
                {
                    "verbatim_quote": "we are severely delaying our phase 3 trials",
                    "reasoning": "Direct delay admission.",
                }
            ],
        }

        async def fake_document_fetcher(_ticker, _doc_type, _date_range):
            return {
                "status": "available",
                "data": "Q3 earnings were solid, however, we are severely delaying our phase 3 trials due to enrollment issues.",
            }

        def fake_rubric_fetcher(_rubric_id):
            return {"criteria": "test"}

        def fake_llm_invoker(prompt, *, system):
            seen["prompt"] = prompt
            seen["system"] = system
            return json.dumps(llm_payload)

        worker = mod.PENRSWorker(
            name="Dilute and Delay",
            weight=1.0,
            signal_density=1.0,
            rubric_id="test_rubric",
            document_type=DocumentType.NEWS_SENTIMENT,
            rubric_fetcher=fake_rubric_fetcher,
            document_fetcher=fake_document_fetcher,
            llm_invoker=fake_llm_invoker,
        )

        result = await worker.run("TEST", "2026-01-01", "2026-02-01")

        self.assertEqual(result["status"], "available")
        self.assertEqual(seen["system"], "Respond only in valid JSON.")
        self.assertIn("You MUST return a strictly valid JSON object adhering to this exact schema:", seen["prompt"])
        self.assertEqual(result["result"], llm_payload)

    async def test_spec_2_3_run_falls_back_when_invoker_rejects_system_kwarg(self):
        mod = import_worker_nodes(force_reload=True)
        calls = {"count": 0}

        def fake_rubric_fetcher(_rubric_id):
            return {"criteria": "fallback-test"}

        async def fake_document_fetcher(_ticker, _doc_type, _date_range):
            return {"status": "available", "data": "Simple excerpt"}

        def positional_only_llm(prompt):
            calls["count"] += 1
            return '{"score": 0.1, "thesis": "ok", "evidence_nodes": []}'

        worker = mod.PENRSWorker(
            name="Fallback Worker",
            weight=1.0,
            signal_density=1.0,
            rubric_id="test_rubric",
            document_type=DocumentType.NEWS_SENTIMENT,
            rubric_fetcher=fake_rubric_fetcher,
            document_fetcher=fake_document_fetcher,
            llm_invoker=positional_only_llm,
        )

        result = await worker.run("TEST", "2026-01-01", "2026-02-01")

        self.assertEqual(calls["count"], 1)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["result"]["score"], 0.1)
        self.assertEqual(result["result"]["thesis"], "ok")
        self.assertEqual(result["result"]["evidence_nodes"], [])

    async def test_spec_2_3_default_llm_invoker_accepts_system_and_returns_schema_safe_payload(self):
        mod = import_worker_nodes(force_reload=True)

        def fake_rubric_fetcher(_rubric_id):
            return {"criteria": "default-llm-test"}

        async def fake_document_fetcher(_ticker, _doc_type, _date_range):
            return {"status": "available", "data": "Document text"}

        worker = mod.PENRSWorker(
            name="Default Invoker",
            weight=1.0,
            signal_density=1.0,
            rubric_id="test_rubric",
            document_type=DocumentType.NEWS_SENTIMENT,
            rubric_fetcher=fake_rubric_fetcher,
            document_fetcher=fake_document_fetcher,
            llm_invoker=None,
        )

        result = await worker.run("TEST", "2026-01-01", "2026-02-01")

        self.assertEqual(result["status"], "available")
        self.assertEqual(
            result["result"],
            {"score": 0.0, "thesis": "Parse failure", "evidence_nodes": []},
        )

    async def test_spec_2_3_run_passes_system_prompt_to_async_kwargs_invoker(self):
        mod = import_worker_nodes(force_reload=True)
        seen = {"system": None}

        def fake_rubric_fetcher(_rubric_id):
            return {"criteria": "kwargs-test"}

        async def fake_document_fetcher(_ticker, _doc_type, _date_range):
            return {"status": "available", "data": "Document excerpt for kwargs invoker."}

        async def async_kwargs_llm(prompt, **kwargs):
            _ = prompt
            seen["system"] = kwargs.get("system")
            return '{"score": -0.2, "thesis": "ok", "evidence_nodes": []}'

        worker = mod.PENRSWorker(
            name="Kwargs Worker",
            weight=1.0,
            signal_density=1.0,
            rubric_id="test_rubric",
            document_type=DocumentType.NEWS_SENTIMENT,
            rubric_fetcher=fake_rubric_fetcher,
            document_fetcher=fake_document_fetcher,
            llm_invoker=async_kwargs_llm,
        )

        result = await worker.run("TEST", "2026-01-01", "2026-02-01")

        self.assertEqual(seen["system"], "Respond only in valid JSON.")
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["result"]["score"], -0.2)


if __name__ == "__main__":
    unittest.main()
