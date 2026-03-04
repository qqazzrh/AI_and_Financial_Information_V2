import asyncio
import json
import os
from pathlib import Path
import uuid

import penrs_pipeline
from penrs_pipeline import run_all_workers, run_penrs

TEST_FILES_DIR = (Path.cwd() / "Test_files").resolve()


def local_tmp_dir() -> Path:
    TEST_FILES_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = TEST_FILES_DIR / f"test_tmp_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir.resolve()


class StubWorker:
    def __init__(
        self,
        *,
        name: str,
        weight: float,
        signal_density: float,
        score: float,
        fail: bool = False,
    ) -> None:
        self.name = name
        self.weight = weight
        self.signal_density = signal_density
        self.score = score
        self.fail = fail

    async def run(self, ticker: str, date_from: str, date_to: str) -> dict:
        if self.fail:
            raise RuntimeError(f"{self.name} exploded")
        return {
            "status": "available",
            "ticker": ticker,
            "date_from": date_from,
            "date_to": date_to,
            "worker": {
                "name": self.name,
                "weight": self.weight,
                "signal_density": self.signal_density,
            },
            "result": {
                "score": self.score,
                "summary": f"{self.name} summary",
            },
        }


class StubArbiter:
    def __init__(self) -> None:
        self.received: list[dict] | None = None

    def evaluate(self, worker_results: list[dict]) -> dict:
        self.received = list(worker_results)
        return {
            "status": "available",
            "worker_scores": [],
            "weighted_score": 0.33,
            "contradictions": [],
        }


class StubMaster:
    def __init__(self) -> None:
        self.received: dict | None = None

    async def synthesize(
        self,
        *,
        ticker: str,
        date_from: str,
        date_to: str,
        worker_results: list[dict],
        arbiter_result: dict,
    ) -> dict:
        self.received = {
            "ticker": ticker,
            "date_from": date_from,
            "date_to": date_to,
            "worker_results": worker_results,
            "arbiter_result": arbiter_result,
        }
        return {
            "status": "available",
            "final_score": arbiter_result.get("weighted_score", 0.0),
        }


def test_run_penrs_executes_pipeline_and_saves_report():
    working_dir = local_tmp_dir()
    original_cwd = Path.cwd()
    os.chdir(working_dir)
    workers = [
        StubWorker(name="W1", weight=1.0, signal_density=0.7, score=0.4),
        StubWorker(name="W2", weight=1.3, signal_density=0.8, score=0.2),
    ]
    arbiter = StubArbiter()
    master = StubMaster()

    try:
        report = asyncio.run(
            run_penrs(
                "MRNA",
                "2026-01-01",
                "2026-02-01",
                workers=workers,
                arbiter=arbiter,
                master=master,
            )
        )
    finally:
        os.chdir(original_cwd)

    assert arbiter.received is not None
    assert len(arbiter.received) == 2
    assert master.received is not None
    assert master.received["arbiter_result"]["weighted_score"] == 0.33

    report_path = Path(report["report_path"])
    assert report_path.exists()
    assert report_path.parent.name == "penrs_reports"

    saved_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_payload["ticker"] == "MRNA"
    assert saved_payload["master"]["final_score"] == 0.33
    assert saved_payload["arbiter"]["status"] == "available"


def test_run_penrs_worker_failure_is_isolated():
    output_dir = local_tmp_dir()
    workers = [
        StubWorker(name="FailWorker", weight=1.0, signal_density=0.6, score=0.0, fail=True),
        StubWorker(name="GoodWorker", weight=1.0, signal_density=0.6, score=0.55),
    ]
    arbiter = StubArbiter()

    report = asyncio.run(
        run_penrs(
            "BIIB",
            "2026-01-01",
            "2026-02-01",
            workers=workers,
            arbiter=arbiter,
            report_dir=output_dir,
        )
    )

    statuses = {entry["worker"]["name"]: entry["status"] for entry in report["worker_results"]}
    assert statuses["FailWorker"] == "error"
    assert statuses["GoodWorker"] == "available"
    assert arbiter.received is not None
    assert len(arbiter.received) == 1
    assert arbiter.received[0]["worker"]["name"] == "GoodWorker"
    assert Path(report["report_path"]).exists()


def test_run_all_workers_uses_asyncio_gather_return_exceptions(monkeypatch):
    captured: dict[str, bool] = {}

    async def fake_gather(*coroutines, return_exceptions=False):
        captured["return_exceptions"] = return_exceptions
        results = []
        for coroutine in coroutines:
            try:
                results.append(await coroutine)
            except Exception as exc:
                if return_exceptions:
                    results.append(exc)
                else:
                    raise
        return results

    monkeypatch.setattr(penrs_pipeline.asyncio, "gather", fake_gather)
    workers = [
        StubWorker(name="Broken", weight=1.0, signal_density=0.5, score=0.0, fail=True),
        StubWorker(name="Healthy", weight=1.0, signal_density=0.5, score=0.1),
    ]

    results = asyncio.run(run_all_workers(workers, "SRPT", "2026-01-01", "2026-02-01"))

    assert captured["return_exceptions"] is True
    statuses = {entry["worker"]["name"]: entry["status"] for entry in results}
    assert statuses["Broken"] == "error"
    assert statuses["Healthy"] == "available"
