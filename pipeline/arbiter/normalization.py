"""Normalization utilities for converting worker outputs into arbiter-ready signals."""

from __future__ import annotations

from typing import Any, Sequence

from pipeline.enums import (
    NormalizedSignalDirection,
    ProcessingStatus,
    SentimentLabel,
)
from pipeline.models import (
    ArbiterIssue,
    ArbiterWarning,
    NormalizedWorkerOutput,
    WorkerOutput,
)

from pipeline.arbiter.models import (
    STRUCTURED_DOCUMENT_TYPES,
    SOFT_DOCUMENT_TYPES,
    USABLE_WORKER_STATUSES,
    convert_worker_pipeline_issue_to_arbiter_issue,
    flatten_worker_evidence_references,
    make_arbiter_warning,
)

# Re-export build_standardized_analysis_score so the arbiter __init__ can
# import it from this module under an alias.  The analysis subpackage may
# not be fully assembled yet, so we try the submodule first and fall back
# to the package-level re-export.
try:
    from pipeline.analysis.rubrics import build_standardized_analysis_score  # noqa: F401
except (ImportError, ModuleNotFoundError):
    try:
        from pipeline.analysis import build_standardized_analysis_score  # noqa: F401
    except (ImportError, ModuleNotFoundError):  # pragma: no cover

        def build_standardized_analysis_score(*args: object, **kwargs: object) -> object:  # type: ignore[misc]
            raise NotImplementedError(
                "build_standardized_analysis_score is not available because "
                "pipeline.analysis has not been fully assembled yet."
            )


def clamp_value(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def safe_mean(values: Sequence[float | None]) -> float | None:
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    return sum(present_values) / len(present_values)


def normalize_worker_sentiment_label(worker_output: WorkerOutput) -> SentimentLabel:
    if worker_output.sentiment is None:
        return SentimentLabel.INSUFFICIENT_EVIDENCE
    return worker_output.sentiment.label


def estimate_worker_evidence_density(worker_output: WorkerOutput) -> float:
    if worker_output.status == ProcessingStatus.NO_DOCUMENT:
        return 0.0

    density = 0.0
    density += min(0.70, len(worker_output.evidence) * 0.20)
    density += min(0.20, len(worker_output.key_points) * 0.04)
    density += 0.10 if worker_output.summary else 0.0
    if worker_output.status == ProcessingStatus.PARTIAL:
        density -= 0.10
    elif worker_output.status not in USABLE_WORKER_STATUSES:
        density -= 0.20
    density -= min(0.10, len(worker_output.issues) * 0.02)
    return clamp_value(density, 0.0, 1.0)


def validate_worker_output_for_arbiter(
    worker_output: WorkerOutput,
) -> tuple[list[ArbiterWarning], list[ArbiterIssue]]:
    warnings: list[ArbiterWarning] = []
    issues: list[ArbiterIssue] = [
        convert_worker_pipeline_issue_to_arbiter_issue(worker_output, issue) for issue in worker_output.issues
    ]

    if worker_output.status not in USABLE_WORKER_STATUSES:
        warnings.append(
            make_arbiter_warning(
                "worker_output_not_usable",
                f"{worker_output.worker_name} returned status {worker_output.status.value}; its signal will be treated cautiously.",
                document_types=[worker_output.document_type],
                worker_names=[worker_output.worker_name],
            )
        )
    if not worker_output.summary:
        warnings.append(make_arbiter_warning("worker_output_missing_summary", "Worker output is missing a summary field.", document_types=[worker_output.document_type], worker_names=[worker_output.worker_name]))
    if worker_output.sentiment is None:
        warnings.append(make_arbiter_warning("worker_output_missing_sentiment", "Worker output is missing sentiment data.", document_types=[worker_output.document_type], worker_names=[worker_output.worker_name]))
    if not worker_output.evidence:
        warnings.append(make_arbiter_warning("worker_output_missing_evidence", "Worker output has no cited evidence.", document_types=[worker_output.document_type], worker_names=[worker_output.worker_name]))
    if not worker_output.key_points:
        warnings.append(make_arbiter_warning("worker_output_missing_key_points", "Worker output has no key points for comparison.", document_types=[worker_output.document_type], worker_names=[worker_output.worker_name]))

    confidence_candidates = [worker_output.confidence, worker_output.sentiment.confidence if worker_output.sentiment else None, worker_output.tone.confidence if worker_output.tone else None]
    if safe_mean(confidence_candidates) is None:
        warnings.append(make_arbiter_warning("worker_output_missing_confidence", "Worker output is missing both direct and nested confidence fields.", document_types=[worker_output.document_type], worker_names=[worker_output.worker_name]))

    if worker_output.sentiment is not None and worker_output.sentiment.score is not None:
        score = worker_output.sentiment.score
        label = worker_output.sentiment.label
        incompatible_label = (
            (label == SentimentLabel.POSITIVE and score < -0.10)
            or (label == SentimentLabel.NEGATIVE and score > 0.10)
            or (label == SentimentLabel.NEUTRAL and abs(score) > 0.35)
        )
        if incompatible_label:
            warnings.append(make_arbiter_warning("sentiment_label_score_mismatch", "Worker sentiment label and score point in materially different directions.", document_types=[worker_output.document_type], worker_names=[worker_output.worker_name], metadata={"sentiment_label": label.value, "sentiment_score": score}))

    evidence_density = estimate_worker_evidence_density(worker_output)
    if evidence_density < 0.30 and worker_output.status in USABLE_WORKER_STATUSES:
        warnings.append(make_arbiter_warning("worker_output_sparse_signal", "Worker output has sparse evidence density for cross-document comparison.", document_types=[worker_output.document_type], worker_names=[worker_output.worker_name], metadata={"evidence_density": round(evidence_density, 3)}))

    return warnings, issues


def normalize_signal_direction(worker_output: WorkerOutput, *, normalized_confidence: float) -> NormalizedSignalDirection:
    if worker_output.status not in USABLE_WORKER_STATUSES:
        return NormalizedSignalDirection.UNCERTAIN
    sentiment = worker_output.sentiment
    if sentiment is None:
        return NormalizedSignalDirection.UNCERTAIN
    if sentiment.label == SentimentLabel.POSITIVE:
        return NormalizedSignalDirection.POSITIVE
    if sentiment.label == SentimentLabel.NEGATIVE:
        return NormalizedSignalDirection.NEGATIVE
    if sentiment.label == SentimentLabel.MIXED:
        return NormalizedSignalDirection.MIXED
    if sentiment.label == SentimentLabel.NEUTRAL:
        return NormalizedSignalDirection.NEUTRAL if normalized_confidence >= 0.55 else NormalizedSignalDirection.UNCERTAIN
    return NormalizedSignalDirection.UNCERTAIN


def normalize_worker_output_for_arbitration(worker_output: WorkerOutput) -> NormalizedWorkerOutput:
    warnings, issues = validate_worker_output_for_arbiter(worker_output)
    sentiment_label = normalize_worker_sentiment_label(worker_output)
    sentiment_score = worker_output.sentiment.score if worker_output.sentiment else None
    evidence_density = estimate_worker_evidence_density(worker_output)
    evidence_references = flatten_worker_evidence_references(worker_output)
    confidence_candidates = [worker_output.confidence, worker_output.sentiment.confidence if worker_output.sentiment else None, worker_output.tone.confidence if worker_output.tone else None]
    worker_confidence_raw = safe_mean(confidence_candidates)
    normalized_confidence = worker_confidence_raw if worker_confidence_raw is not None else 0.35
    normalization_notes: list[str] = []

    if worker_output.status == ProcessingStatus.PARTIAL:
        normalized_confidence = clamp_value(normalized_confidence - 0.08, 0.0, 1.0)
        normalization_notes.append("Reduced confidence slightly because the worker status is partial.")
    elif worker_output.status not in USABLE_WORKER_STATUSES:
        normalized_confidence = min(normalized_confidence, 0.25)
        normalization_notes.append("Capped confidence because the worker did not return a usable status.")
    if evidence_density < 0.30:
        normalized_confidence = clamp_value(normalized_confidence - 0.07, 0.0, 1.0)
        normalization_notes.append("Reduced confidence because evidence density is sparse.")

    direction = normalize_signal_direction(worker_output, normalized_confidence=normalized_confidence)
    if normalized_confidence < 0.45:
        warnings.append(make_arbiter_warning("worker_output_low_confidence", "Worker output carries low confidence after arbiter normalization.", document_types=[worker_output.document_type], worker_names=[worker_output.worker_name], metadata={"normalized_confidence": round(normalized_confidence, 3)}))

    return NormalizedWorkerOutput(
        worker_name=worker_output.worker_name,
        document_type=worker_output.document_type,
        status=worker_output.status,
        summary=worker_output.summary,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        worker_confidence_raw=worker_confidence_raw,
        normalized_confidence=normalized_confidence,
        evidence_density=evidence_density,
        key_points=list(worker_output.key_points),
        caveats=list(worker_output.caveats),
        evidence_count=len(worker_output.evidence),
        key_point_count=len(worker_output.key_points),
        caveat_count=len(worker_output.caveats),
        issue_count=len(worker_output.issues),
        direction=direction,
        fogging_score=worker_output.tone.fogging_score if worker_output.tone else None,
        hedging_score=worker_output.tone.hedging_score if worker_output.tone else None,
        promotional_score=worker_output.tone.promotional_score if worker_output.tone else None,
        is_structured_document=worker_output.document_type in STRUCTURED_DOCUMENT_TYPES,
        is_soft_document=worker_output.document_type in SOFT_DOCUMENT_TYPES,
        warnings=warnings,
        issues=issues,
        evidence_references=evidence_references,
        normalization_notes=normalization_notes,
    )
