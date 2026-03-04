import json

import pytest

from penrs_arbiter import ArbiterAgent, ARBITER_SYSTEM_PROMPT


def _worker_result(
    *,
    name: str,
    weight: float,
    signal_density: float,
    score: float,
    star_rating: int | None = None,
    summary: str = "",
) -> dict:
    result_payload = {"score": score, "summary": summary}
    if star_rating is not None:
        result_payload["star_rating"] = star_rating
    return {
        "status": "available",
        "worker": {
            "name": name,
            "weight": weight,
            "signal_density": signal_density,
        },
        "result": result_payload,
    }


def test_system_prompt_contains_required_role_and_mandatory_contradictions():
    assert "Lead Portfolio Manager" in ARBITER_SYSTEM_PROMPT
    assert "Lipstick on a Pig" in ARBITER_SYSTEM_PROMPT
    assert "Bailing Out" in ARBITER_SYSTEM_PROMPT
    assert "Dilute and Delay" in ARBITER_SYSTEM_PROMPT


def test_arbiter_validates_required_worker_schema_fields():
    arbiter = ArbiterAgent()
    bad_worker = {
        "status": "available",
        "worker": {
            "name": "MissingWeight",
            "signal_density": 0.8,
        },
        "result": {"score": 0.3},
    }

    with pytest.raises(ValueError, match="worker.weight"):
        arbiter.evaluate([bad_worker])


def test_arbiter_clamps_scores_and_applies_star_rating_weights():
    arbiter = ArbiterAgent()
    workers = [
        _worker_result(
            name="WorkerA",
            weight=2.0,
            signal_density=0.9,
            score=1.8,
            star_rating=5,
        ),
        _worker_result(
            name="WorkerB",
            weight=1.5,
            signal_density=0.2,
            score=-2.4,
            star_rating=2,
        ),
    ]

    report = arbiter.evaluate(workers)
    worker_scores = {entry["name"]: entry for entry in report["worker_scores"]}

    assert worker_scores["WorkerA"]["normalized_score"] == 1.0
    assert worker_scores["WorkerA"]["effective_weight"] == 2.0
    assert worker_scores["WorkerA"]["weighted_score"] == 2.0

    assert worker_scores["WorkerB"]["normalized_score"] == -1.0
    assert worker_scores["WorkerB"]["effective_weight"] == 0.6
    assert worker_scores["WorkerB"]["weighted_score"] == -0.6

    assert report["weighted_score"] == pytest.approx(1.4 / 2.6, rel=1e-6)


def test_arbiter_returns_json_schema_with_contradiction_flags_and_severities():
    arbiter = ArbiterAgent()
    workers = [
        _worker_result(
            name="NarrativeWorker",
            weight=1.0,
            signal_density=0.6,
            score=0.2,
            summary="Management is putting lipstick on a pig while bailing out early holders.",
        )
    ]

    report = arbiter.evaluate(workers)
    contradictions = {entry["name"]: entry for entry in report["contradictions"]}

    assert set(contradictions) == {"Lipstick on a Pig", "Bailing Out", "Dilute and Delay"}
    assert contradictions["Lipstick on a Pig"]["flagged"] is True
    assert contradictions["Lipstick on a Pig"]["severity"] == "High"
    assert contradictions["Bailing Out"]["flagged"] is True
    assert contradictions["Bailing Out"]["severity"] == "High"
    assert contradictions["Dilute and Delay"]["severity"] == "Medium"

    json.dumps(report)

