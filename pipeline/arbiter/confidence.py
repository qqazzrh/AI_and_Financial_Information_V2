"""Confidence assessment, arbiter warnings, and missing-coverage notes."""

from __future__ import annotations

from typing import Any, Sequence

from pipeline.enums import (
    DocumentType,
    ProcessingStatus,
)
from pipeline.models import (
    ArbiterConfidenceAdjustment,
    ArbiterConfidenceAssessment,
    ArbiterConflict,
    ArbiterWarning,
    NormalizedWorkerOutput,
)

from pipeline.arbiter.models import (
    STRUCTURED_DOCUMENT_TYPES,
    USABLE_WORKER_STATUSES,
    document_type_label,
    make_arbiter_warning,
    stable_unique,
)
from pipeline.arbiter.normalization import (
    clamp_value,
    safe_mean,
)

__all__ = [
    "make_confidence_adjustment",
    "adjust_confidence_for_evidence_density",
    "adjust_confidence_for_conflict",
    "adjust_confidence_for_missing_coverage",
    "adjust_confidence_for_structured_support",
    "assess_cross_document_confidence",
    "build_missing_coverage_notes",
    "build_arbiter_warnings",
]


def make_confidence_adjustment(factor_name: str, adjustment: float, rationale: str) -> ArbiterConfidenceAdjustment:
    return ArbiterConfidenceAdjustment(factor_name=factor_name, adjustment=adjustment, rationale=rationale)


def adjust_confidence_for_evidence_density(normalized_outputs: Sequence[NormalizedWorkerOutput]) -> ArbiterConfidenceAdjustment:
    evidence_density_average = safe_mean([output.evidence_density for output in normalized_outputs]) or 0.0
    adjustment = clamp_value((evidence_density_average - 0.50) * 0.30, -0.15, 0.15)
    return make_confidence_adjustment("evidence_density", adjustment, f"Average evidence density was {evidence_density_average:.2f}.")


def adjust_confidence_for_conflict(conflicts: Sequence[ArbiterConflict]) -> ArbiterConfidenceAdjustment:
    high_confidence_conflict_count = sum(1 for conflict in conflicts if conflict.high_confidence_conflict)
    adjustment = -min(0.25, (len(conflicts) * 0.03) + (high_confidence_conflict_count * 0.05))
    return make_confidence_adjustment("explicit_conflicts", adjustment, f"Detected {len(conflicts)} explicit conflicts, including {high_confidence_conflict_count} high-confidence conflicts.")


def adjust_confidence_for_missing_coverage(missing_document_types: Sequence[DocumentType]) -> ArbiterConfidenceAdjustment:
    adjustment = -min(0.10, len(missing_document_types) * 0.02)
    return make_confidence_adjustment("missing_coverage", adjustment, f"Missing usable coverage for {len(missing_document_types)} expected document types.")


def adjust_confidence_for_structured_support(normalized_outputs: Sequence[NormalizedWorkerOutput]) -> tuple[ArbiterConfidenceAdjustment, float]:
    usable_outputs = [output for output in normalized_outputs if output.status in USABLE_WORKER_STATUSES]
    if not usable_outputs:
        return make_confidence_adjustment("structured_support", -0.10, "No usable worker outputs were available."), 0.0
    weighted_supports = [output.normalized_confidence * (0.50 + (0.50 * output.evidence_density)) for output in usable_outputs]
    total_support = sum(weighted_supports)
    structured_support = sum(output.normalized_confidence * (0.50 + (0.50 * output.evidence_density)) for output in usable_outputs if output.is_structured_document)
    structured_support_ratio = structured_support / total_support if total_support > 0.0 else 0.0
    adjustment = clamp_value((structured_support_ratio - 0.50) * 0.20, -0.10, 0.10)
    return make_confidence_adjustment("structured_support", adjustment, f"Structured-document support ratio was {structured_support_ratio:.2f}."), structured_support_ratio


def assess_cross_document_confidence(
    normalized_outputs: Sequence[NormalizedWorkerOutput],
    conflicts: Sequence[ArbiterConflict],
    missing_document_types: Sequence[DocumentType],
) -> ArbiterConfidenceAssessment:
    usable_outputs = [output for output in normalized_outputs if output.status in USABLE_WORKER_STATUSES]
    worker_confidence_average = safe_mean([output.normalized_confidence for output in usable_outputs]) or 0.25
    evidence_density_average = safe_mean([output.evidence_density for output in usable_outputs]) or 0.0
    base_confidence = clamp_value(worker_confidence_average, 0.0, 1.0)
    adjustments = [
        adjust_confidence_for_evidence_density(usable_outputs),
        adjust_confidence_for_conflict(conflicts),
        adjust_confidence_for_missing_coverage(missing_document_types),
    ]
    structured_support_adjustment, structured_support_ratio = adjust_confidence_for_structured_support(usable_outputs)
    adjustments.append(structured_support_adjustment)
    final_confidence = clamp_value(base_confidence + sum(adjustment.adjustment for adjustment in adjustments), 0.0, 1.0)
    return ArbiterConfidenceAssessment(
        base_confidence=base_confidence,
        final_confidence=final_confidence,
        worker_confidence_average=worker_confidence_average,
        evidence_density_average=evidence_density_average,
        structured_support_ratio=structured_support_ratio,
        missing_document_types=list(missing_document_types),
        high_confidence_conflict_count=sum(1 for conflict in conflicts if conflict.high_confidence_conflict),
        adjustments=adjustments,
        reasoning_notes=[
            f"base_confidence={base_confidence:.2f}",
            f"worker_confidence_average={worker_confidence_average:.2f}",
            f"evidence_density_average={evidence_density_average:.2f}",
            f"structured_support_ratio={structured_support_ratio:.2f}",
        ],
    )


def build_missing_coverage_notes(
    normalized_outputs: Sequence[NormalizedWorkerOutput],
    missing_document_types: Sequence[DocumentType],
) -> list[str]:
    notes: list[str] = []
    if missing_document_types:
        notes.append("Missing or unusable worker coverage for: " + ", ".join(document_type_label(document_type) for document_type in missing_document_types) + ".")
    structured_coverage_present = any(output.is_structured_document and output.status in USABLE_WORKER_STATUSES for output in normalized_outputs)
    if not structured_coverage_present:
        notes.append("No successful structured-document worker output is available; current arbitration is dominated by softer narrative evidence.")
    usable_outputs = [output for output in normalized_outputs if output.status in USABLE_WORKER_STATUSES]
    if usable_outputs and all(output.is_soft_document for output in usable_outputs):
        notes.append("All usable support currently comes from softer narrative documents rather than harder disclosures.")
    return notes


def build_arbiter_warnings(
    normalized_outputs: Sequence[NormalizedWorkerOutput],
    conflicts: Sequence[ArbiterConflict],
    missing_document_types: Sequence[DocumentType],
    confidence_assessment: ArbiterConfidenceAssessment,
) -> list[ArbiterWarning]:
    warnings: list[ArbiterWarning] = [warning for normalized_output in normalized_outputs for warning in normalized_output.warnings]
    usable_outputs = [output for output in normalized_outputs if output.status in USABLE_WORKER_STATUSES]
    if len(usable_outputs) < 2:
        warnings.append(make_arbiter_warning("too_few_worker_outputs", "Too few usable worker outputs are available for a strong cross-document judgment.", document_types=[output.document_type for output in usable_outputs], worker_names=[output.worker_name for output in usable_outputs]))
    if any(conflict.high_confidence_conflict for conflict in conflicts):
        warnings.append(make_arbiter_warning("high_confidence_worker_conflict", "At least two higher-confidence worker outputs materially disagree.", document_types=stable_unique([document_type for conflict in conflicts for document_type in conflict.positive_document_types + conflict.negative_document_types]), worker_names=stable_unique([worker_name for conflict in conflicts for worker_name in conflict.worker_names])))
    if any(document_type in STRUCTURED_DOCUMENT_TYPES for document_type in missing_document_types):
        warnings.append(make_arbiter_warning("missing_hard_document_coverage", "One or more structured disclosure types are missing or unusable for arbitration.", document_types=[document_type for document_type in missing_document_types if document_type in STRUCTURED_DOCUMENT_TYPES]))
    if usable_outputs and all(output.is_soft_document for output in usable_outputs):
        warnings.append(make_arbiter_warning("soft_document_only_support", "All usable evidence currently comes from softer narrative documents.", document_types=[output.document_type for output in usable_outputs], worker_names=[output.worker_name for output in usable_outputs]))
    evidence_density_average = safe_mean([output.evidence_density for output in usable_outputs]) or 0.0
    if usable_outputs and evidence_density_average < 0.30:
        warnings.append(make_arbiter_warning("low_evidence_density", "Average evidence density across available worker outputs is low.", document_types=[output.document_type for output in usable_outputs], worker_names=[output.worker_name for output in usable_outputs], metadata={"evidence_density_average": round(evidence_density_average, 3)}))
    if confidence_assessment.final_confidence < 0.35:
        warnings.append(make_arbiter_warning("low_arbiter_confidence", "Cross-document confidence is low after reconciliation.", metadata={"final_confidence": round(confidence_assessment.final_confidence, 3)}))
    deduplicated_warnings: dict[tuple[Any, ...], ArbiterWarning] = {}
    for warning in warnings:
        warning_key = (warning.issue_code, warning.message, tuple(document_type.value for document_type in warning.document_types), tuple(warning.worker_names))
        if warning_key not in deduplicated_warnings:
            deduplicated_warnings[warning_key] = warning
    return list(deduplicated_warnings.values())
