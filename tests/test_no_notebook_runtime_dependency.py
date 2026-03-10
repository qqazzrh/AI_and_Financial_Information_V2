from pathlib import Path
import unittest
from unittest.mock import patch

from pipeline.arbiter.models import EXPECTED_DOCUMENT_TYPES
from pipeline.enums import ProcessingStatus
from pipeline.models import (
    ArbiterOutput,
    FinalUIPayload,
    RetrievalResult,
    WorkerInput,
    WorkerOutput,
)
from pipeline.orchestration import TieredPipelineRequest, run_full_pipeline, run_tiered_pipeline, set_database_handles


class _FakeWorker:
    def __init__(self, analysis_client=None):
        self.analysis_client = analysis_client

    def analyze(self, worker_input: WorkerInput) -> WorkerOutput:
        return WorkerOutput(
            worker_name="fake_worker",
            document_type=worker_input.document_type,
            status=ProcessingStatus.SUCCESS,
            summary=f"ok:{worker_input.document_type.value}",
            confidence=0.8,
        )


class _FakeArbiter:
    def arbitrate(self, arbiter_input):
        return ArbiterOutput(
            arbiter_id="arbiter-1",
            arbiter_name="fake_arbiter",
            status=ProcessingStatus.SUCCESS,
            covered_document_types=[w.document_type for w in arbiter_input.worker_outputs],
            missing_document_types=[],
        )


class _FakeMaster:
    def build_payload(self, master_input):
        return FinalUIPayload(
            ticker=master_input.ticker,
            status=ProcessingStatus.SUCCESS,
            disclosures=[],
        )


def _fake_retrieval_result_map(requests):
    results = {}
    for request in requests:
        results[request.document_type] = RetrievalResult(
            request=request,
            adapter_name="fake_adapter",
            status=ProcessingStatus.RETRIEVAL_FAILED,
        )
    return results


class PipelineNoNotebookDependencyTests(unittest.TestCase):
    def test_run_full_pipeline_from_python_modules(self):
        fake_registry = {dt: _FakeWorker for dt in EXPECTED_DOCUMENT_TYPES}

        with (
            patch("pipeline.orchestration.resolve_company_from_ticker", return_value=("DemoCo", ["DemoCo", "DEMO"], "1234")),
            patch("pipeline.orchestration.run_retrieval", side_effect=_fake_retrieval_result_map),
            patch("pipeline.orchestration.build_analysis_client", return_value=type("C", (), {"client_name": "fake_client"})()),
            patch("pipeline.orchestration.WORKER_REGISTRY", fake_registry),
            patch("pipeline.orchestration.CrossDocumentArbiter", return_value=_FakeArbiter()),
            patch("pipeline.orchestration.IntegratedMasterNode", return_value=_FakeMaster()),
        ):
            result = run_full_pipeline("DEMO")

        self.assertIn("final_payload", result)
        self.assertEqual(result["final_payload"].status, ProcessingStatus.SUCCESS)
        self.assertEqual(set(result["worker_outputs"].keys()), set(EXPECTED_DOCUMENT_TYPES))


    def test_runtime_entrypoints_do_not_reference_notebooks(self):
        checked_files = [
            Path("pipeline/__init__.py"),
            Path("pipeline/orchestration.py"),
            Path("run_pipeline.py"),
            Path("api_server.py"),
        ]
        for file_path in checked_files:
            text = file_path.read_text().lower()
            self.assertNotIn(".ipynb", text, f"{file_path} should not reference notebooks")
            self.assertNotIn("exec(", text, f"{file_path} should not execute notebook code")

    def test_run_tiered_pipeline_from_python_modules(self):
        fake_registry = {dt: _FakeWorker for dt in EXPECTED_DOCUMENT_TYPES}
        set_database_handles(object(), object())

        with (
            patch("pipeline.orchestration.resolve_company_from_ticker", return_value=("DemoCo", ["DemoCo", "DEMO"], "1234")),
            patch("pipeline.orchestration.run_retrieval", side_effect=_fake_retrieval_result_map),
            patch("pipeline.orchestration.build_analysis_client", return_value=type("C", (), {"client_name": "fake_client"})()),
            patch("pipeline.orchestration.WORKER_REGISTRY", fake_registry),
            patch("pipeline.orchestration.CrossDocumentArbiter", return_value=_FakeArbiter()),
            patch("pipeline.orchestration.IntegratedMasterNode", return_value=_FakeMaster()),
            patch("pipeline.orchestration.get_or_create_user", return_value={"user_id": "u1"}),
            patch("pipeline.orchestration.find_cached_run", return_value=None),
            patch("pipeline.orchestration.save_document_run", return_value=None),
            patch("pipeline.orchestration.record_query", return_value="q1"),
            patch("pipeline.orchestration.save_chunks_to_lancedb", return_value=None),
        ):
            req = TieredPipelineRequest(ticker="DEMO", user_id="u1", enable_graph_context=False)
            result = run_tiered_pipeline(req)

        self.assertEqual(result["analysis_tier"], "text_only")
        self.assertEqual(result["final_payload"].status, ProcessingStatus.SUCCESS)
        self.assertEqual(result["cache_misses"], len(EXPECTED_DOCUMENT_TYPES))


if __name__ == "__main__":
    unittest.main()
