from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from penrs_arbiter import ArbiterAgent


class PENRSReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    ticker: str
    date_from: str
    date_to: str
    generated_at: str
    worker_results: list[dict[str, Any]] = Field(default_factory=list)
    arbiter: dict[str, Any]
    master: dict[str, Any]
    report_path: str


class MasterAgent:
    def synthesize(
        self,
        *,
        ticker: str,
        date_from: str,
        date_to: str,
        worker_results: list[dict[str, Any]],
        arbiter_result: dict[str, Any],
    ) -> dict[str, Any]:
        available_workers = sum(1 for result in worker_results if result.get("status") == "available")
        weighted_score = arbiter_result.get("weighted_score", 0.0)
        if not isinstance(weighted_score, (int, float)):
            weighted_score = 0.0

        return {
            "status": "available",
            "model": "programmatic_master_v1",
            "ticker": ticker,
            "date_range": {"from": date_from, "to": date_to},
            "final_score": float(weighted_score),
            "available_worker_count": available_workers,
            "total_worker_count": len(worker_results),
        }


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _coerce_float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _worker_identity(worker: Any) -> dict[str, Any]:
    return {
        "name": str(getattr(worker, "name", worker.__class__.__name__)),
        "weight": _coerce_float_or_zero(getattr(worker, "weight", 0.0)),
        "signal_density": _coerce_float_or_zero(getattr(worker, "signal_density", 0.0)),
    }


async def _invoke_worker(worker: Any, ticker: str, date_from: str, date_to: str) -> Any:
    run_callable = getattr(worker, "run", None)
    if run_callable is None or not callable(run_callable):
        raise TypeError("Worker must expose a callable run(ticker, date_from, date_to)")
    return await _maybe_await(run_callable(ticker, date_from, date_to))


async def run_all_workers(
    workers: Sequence[Any],
    ticker: str,
    date_from: str,
    date_to: str,
) -> list[dict[str, Any]]:
    worker_list = list(workers)
    if not worker_list:
        return []

    coroutines = [_invoke_worker(worker, ticker, date_from, date_to) for worker in worker_list]
    raw_results = await asyncio.gather(*coroutines, return_exceptions=True)

    worker_results: list[dict[str, Any]] = []
    for worker, raw_result in zip(worker_list, raw_results):
        if isinstance(raw_result, Exception):
            worker_results.append(
                {
                    "status": "error",
                    "worker": _worker_identity(worker),
                    "ticker": ticker,
                    "date_from": date_from,
                    "date_to": date_to,
                    "error": str(raw_result),
                }
            )
            continue

        if not isinstance(raw_result, dict):
            worker_results.append(
                {
                    "status": "error",
                    "worker": _worker_identity(worker),
                    "ticker": ticker,
                    "date_from": date_from,
                    "date_to": date_to,
                    "error": "Worker returned non-dict result",
                }
            )
            continue

        worker_results.append(raw_result)

    return worker_results


def _build_arbiter_input(worker_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arbiter_input: list[dict[str, Any]] = []
    for worker_result in worker_results:
        if worker_result.get("status") != "available":
            continue
        if not isinstance(worker_result.get("worker"), dict):
            continue
        if not isinstance(worker_result.get("result"), dict):
            continue
        arbiter_input.append(worker_result)
    return arbiter_input


def _evaluate_with_arbiter(arbiter: ArbiterAgent, worker_results: list[dict[str, Any]]) -> dict[str, Any]:
    arbiter_input = _build_arbiter_input(worker_results)
    if not arbiter_input:
        return {
            "status": "not_available",
            "worker_scores": [],
            "weighted_score": 0.0,
            "contradictions": [],
        }

    try:
        return arbiter.evaluate(arbiter_input)
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "worker_scores": [],
            "weighted_score": 0.0,
            "contradictions": [],
        }


def _safe_filename_component(value: str) -> str:
    sanitized = "".join(char if (char.isalnum() or char in {"-", "_"}) else "_" for char in value)
    sanitized = sanitized.strip("_")
    return sanitized or "unknown"


async def run_penrs(
    ticker: str,
    date_from: str,
    date_to: str,
    *,
    workers: Sequence[Any] | None = None,
    arbiter: ArbiterAgent | None = None,
    master: MasterAgent | None = None,
    report_dir: str | Path = "penrs_reports",
    now: datetime | None = None,
) -> dict[str, Any]:
    worker_results = await run_all_workers(list(workers or []), ticker, date_from, date_to)

    arbiter_agent = arbiter or ArbiterAgent()
    arbiter_result = _evaluate_with_arbiter(arbiter_agent, worker_results)

    master_agent = master or MasterAgent()
    master_result_raw = master_agent.synthesize(
        ticker=ticker,
        date_from=date_from,
        date_to=date_to,
        worker_results=worker_results,
        arbiter_result=arbiter_result,
    )
    master_result = await _maybe_await(master_result_raw)
    if not isinstance(master_result, dict):
        master_result = {"status": "error", "error": "Master returned non-dict result"}

    generated_at = now or datetime.now(timezone.utc)
    generated_at_utc = generated_at.astimezone(timezone.utc)
    report_dir_path = Path(report_dir)
    report_dir_path.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{_safe_filename_component(ticker)}_"
        f"{_safe_filename_component(date_from)}_"
        f"{_safe_filename_component(date_to)}_"
        f"{generated_at_utc.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path = (report_dir_path / filename).resolve()

    report_payload = {
        "ticker": ticker,
        "date_from": date_from,
        "date_to": date_to,
        "generated_at": generated_at_utc.isoformat(),
        "worker_results": worker_results,
        "arbiter": arbiter_result,
        "master": master_result,
        "report_path": str(report_path),
    }
    validated_report = PENRSReport.model_validate(report_payload)
    serialized_report = validated_report.model_dump(mode="json")
    report_path.write_text(json.dumps(serialized_report, indent=2, ensure_ascii=True), encoding="utf-8")
    return serialized_report
