from __future__ import annotations

import json
import unittest

from tests.test_support import import_worker_nodes
from utils import DocumentType


class WorkerSpec2UnitTests(unittest.TestCase):
    def _make_worker(self):
        mod = import_worker_nodes(force_reload=True)
        return mod, mod.PENRSWorker(
            name="Clinical Signals",
            weight=1.0,
            signal_density=0.9,
            rubric_id="test_rubric",
            document_type=DocumentType.NEWS_SENTIMENT,
        )

    def test_spec_2_1_build_prompt_injects_strict_schema_and_verbatim_constraint(self):
        _mod, worker = self._make_worker()

        prompt = worker.build_prompt(
            ticker="MRNA",
            date_from="2026-01-01",
            date_to="2026-02-01",
            rubric={"criteria": "test"},
            document_excerpt="Phase 3 enrollment delay was disclosed.",
        )

        self.assertIn("Return only JSON.", prompt)
        self.assertIn("You MUST return a strictly valid JSON object adhering to this exact schema:", prompt)
        self.assertIn('"score": <float between -1.0 and 1.0>', prompt)
        self.assertIn('"thesis": <string>', prompt)
        self.assertIn('"evidence_nodes": [', prompt)
        self.assertIn(
            '{"verbatim_quote": <exact substring from document>, "reasoning": <string>}',
            prompt,
        )
        self.assertIn(
            "The 'verbatim_quote' MUST be an exact, character-for-character substring of the provided Document excerpt.",
            prompt,
        )

    def test_spec_2_2_parse_json_response_defaults_when_keys_missing(self):
        _mod, worker = self._make_worker()

        parsed = worker.parse_json_response("{}")

        self.assertEqual(
            parsed,
            {"score": 0.0, "thesis": "Parse failure", "evidence_nodes": []},
        )

    def test_spec_2_2_parse_json_response_defaults_when_any_required_key_missing(self):
        _mod, worker = self._make_worker()
        default_payload = {"score": 0.0, "thesis": "Parse failure", "evidence_nodes": []}

        missing_score = worker.parse_json_response(json.dumps({"thesis": "x", "evidence_nodes": []}))
        missing_thesis = worker.parse_json_response(json.dumps({"score": 0.2, "evidence_nodes": []}))
        missing_evidence = worker.parse_json_response(json.dumps({"score": 0.2, "thesis": "x"}))

        self.assertEqual(missing_score, default_payload)
        self.assertEqual(missing_thesis, default_payload)
        self.assertEqual(missing_evidence, default_payload)

    def test_spec_2_2_parse_json_response_enforces_types_and_clamps_score(self):
        _mod, worker = self._make_worker()

        parsed = worker.parse_json_response(
            json.dumps(
                {
                    "score": "7.5",
                    "thesis": 999,
                    "evidence_nodes": {"invalid": True},
                }
            )
        )

        self.assertEqual(parsed["score"], 1.0)
        self.assertEqual(parsed["thesis"], "Parse failure")
        self.assertEqual(parsed["evidence_nodes"], [])

    def test_spec_2_2_parse_json_response_coerces_score_to_float_type(self):
        _mod, worker = self._make_worker()

        parsed = worker.parse_json_response(
            json.dumps(
                {
                    "score": 1,
                    "thesis": "typed value",
                    "evidence_nodes": [],
                }
            )
        )

        self.assertIsInstance(parsed["score"], float)
        self.assertEqual(parsed["score"], 1.0)

    def test_spec_2_2_parse_json_response_supports_embedded_and_fenced_json(self):
        _mod, worker = self._make_worker()
        expected = {
            "score": -0.25,
            "thesis": "Delay risk increasing.",
            "evidence_nodes": [{"verbatim_quote": "delaying our phase 3 trials", "reasoning": "Direct disclosure."}],
        }

        parsed_fenced = worker.parse_json_response(f"```json\n{json.dumps(expected)}\n```")
        parsed_embedded = worker.parse_json_response(f"Model output: {json.dumps(expected)} trailing notes")

        self.assertEqual(parsed_fenced, expected)
        self.assertEqual(parsed_embedded, expected)

    def test_spec_2_2_parse_json_response_returns_default_for_non_json(self):
        _mod, worker = self._make_worker()

        parsed = worker.parse_json_response("not-json-at-all")

        self.assertEqual(
            parsed,
            {"score": 0.0, "thesis": "Parse failure", "evidence_nodes": []},
        )

    def test_spec_2_2_parse_json_response_handles_dict_and_none_inputs(self):
        _mod, worker = self._make_worker()

        parsed_dict = worker.parse_json_response(
            {"score": -0.4, "thesis": "dict-input", "evidence_nodes": []}
        )
        parsed_none = worker.parse_json_response(None)

        self.assertEqual(
            parsed_dict,
            {"score": -0.4, "thesis": "dict-input", "evidence_nodes": []},
        )
        self.assertEqual(
            parsed_none,
            {"score": 0.0, "thesis": "Parse failure", "evidence_nodes": []},
        )


if __name__ == "__main__":
    unittest.main()
