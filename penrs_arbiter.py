from __future__ import annotations

import re
from typing import Any

ARBITER_SYSTEM_PROMPT = (
    "You are the Lead Portfolio Manager reviewing all worker outputs for consistency.\n"
    "You must flag mandatory contradictions when present:\n"
    "- Lipstick on a Pig\n"
    "- Bailing Out\n"
    "- Dilute and Delay\n"
    "Return strictly valid JSON."
)

_MANDATORY_CONTRADICTIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "Lipstick on a Pig",
        "severity": "High",
        "patterns": (r"\blipstick on a pig\b",),
    },
    {
        "name": "Bailing Out",
        "severity": "High",
        "patterns": (r"\bbailing out\b", r"\bbail[- ]?out\b"),
    },
    {
        "name": "Dilute and Delay",
        "severity": "Medium",
        "patterns": (r"\bdilute and delay\b",),
    },
)


def _require_field(payload: dict[str, Any], field_name: str, path: str) -> Any:
    if field_name not in payload:
        raise ValueError(f"Missing required field: {path}")
    return payload[field_name]


def _coerce_float(value: Any, field_path: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field '{field_path}' must be numeric") from exc


def _normalize_score(score: float) -> float:
    return max(-1.0, min(1.0, score))


def _derive_star_rating(signal_density: float) -> int:
    if signal_density >= 0.85:
        return 5
    if signal_density >= 0.65:
        return 4
    if signal_density >= 0.45:
        return 3
    if signal_density >= 0.25:
        return 2
    return 1


def _extract_star_rating(result: dict[str, Any], signal_density: float) -> int:
    rating = result.get("star_rating")
    if rating is None:
        return _derive_star_rating(signal_density)
    rating_value = _coerce_float(rating, "result.star_rating")
    return int(max(1, min(5, round(rating_value))))


def _collect_narrative_text(result: dict[str, Any]) -> str:
    fragments: list[str] = []
    for key in ("summary", "thesis", "narrative", "analysis"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            fragments.append(value.strip())
    return " ".join(fragments)


def _detect_mandatory_contradictions(narrative_text: str) -> list[dict[str, Any]]:
    normalized_text = narrative_text.lower()
    contradictions: list[dict[str, Any]] = []
    for rule in _MANDATORY_CONTRADICTIONS:
        matched_phrase = None
        for pattern in rule["patterns"]:
            match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
            if match:
                matched_phrase = match.group(0)
                break

        contradictions.append(
            {
                "name": rule["name"],
                "severity": rule["severity"],
                "flagged": matched_phrase is not None,
                "evidence": matched_phrase,
            }
        )
    return contradictions


class ArbiterAgent:
    def __init__(self, system_prompt: str = ARBITER_SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt

    def _validate_worker_result(self, worker_result: dict[str, Any]) -> None:
        if not isinstance(worker_result, dict):
            raise ValueError("Each worker result must be a JSON object")

        _require_field(worker_result, "status", "status")
        worker = _require_field(worker_result, "worker", "worker")
        result = _require_field(worker_result, "result", "result")

        if not isinstance(worker, dict):
            raise ValueError("Field 'worker' must be an object")
        if not isinstance(result, dict):
            raise ValueError("Field 'result' must be an object")

        _require_field(worker, "name", "worker.name")
        _require_field(worker, "weight", "worker.weight")
        _require_field(worker, "signal_density", "worker.signal_density")
        _require_field(result, "score", "result.score")

    def evaluate(self, worker_results: list[dict[str, Any]]) -> dict[str, Any]:
        worker_scores: list[dict[str, Any]] = []
        narrative_fragments: list[str] = []
        total_effective_weight = 0.0
        weighted_score_sum = 0.0

        for worker_result in worker_results:
            self._validate_worker_result(worker_result)
            if worker_result.get("status") != "available":
                continue

            worker = worker_result["worker"]
            result = worker_result["result"]

            name = str(worker["name"])
            base_weight = _coerce_float(worker["weight"], "worker.weight")
            signal_density = _coerce_float(worker["signal_density"], "worker.signal_density")
            raw_score = _coerce_float(result["score"], "result.score")
            normalized_score = _normalize_score(raw_score)
            star_rating = _extract_star_rating(result, signal_density)
            effective_weight = round(base_weight * (star_rating / 5.0), 10)
            weighted_score = round(normalized_score * effective_weight, 10)

            total_effective_weight += effective_weight
            weighted_score_sum += weighted_score
            narrative_fragments.append(_collect_narrative_text(result))

            worker_scores.append(
                {
                    "name": name,
                    "raw_score": raw_score,
                    "normalized_score": normalized_score,
                    "weight": base_weight,
                    "star_rating": star_rating,
                    "effective_weight": effective_weight,
                    "weighted_score": weighted_score,
                }
            )

        weighted_score = 0.0
        if total_effective_weight > 0:
            weighted_score = weighted_score_sum / total_effective_weight
            weighted_score = _normalize_score(weighted_score)
            weighted_score = round(weighted_score, 10)

        contradictions = _detect_mandatory_contradictions(" ".join(narrative_fragments))

        return {
            "status": "available",
            "arbiter_role": "Lead Portfolio Manager",
            "system_prompt": self.system_prompt,
            "worker_scores": worker_scores,
            "weighted_score": weighted_score,
            "contradictions": contradictions,
        }
