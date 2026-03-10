"""Arbiter-specific constants, helpers, and factory functions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Hashable, Sequence

from pipeline.config import DISCLOSURE_TYPE_LABELS
from pipeline.enums import (
    AnalysisIssueSeverity,
    ArbiterSignalCategory,
    CrossDocumentTheme,
    DocumentType,
    EvidenceInterpretation,
    NormalizedSignalDirection,
    ProcessingStatus,
)
from pipeline.models import (
    ArbiterEvidenceReference,
    ArbiterIssue,
    ArbiterWarning,
    EvidenceSnippet,
    PipelineError,
    WorkerOutput,
)


EXPECTED_DOCUMENT_TYPES: list[DocumentType] = list(DocumentType)

USABLE_WORKER_STATUSES = {ProcessingStatus.SUCCESS, ProcessingStatus.PARTIAL}

STRUCTURED_DOCUMENT_TYPES = {
    DocumentType.MATERIAL_EVENT,
    DocumentType.CLINICAL_TRIAL_UPDATE,
    DocumentType.FDA_REVIEW,
    DocumentType.FINANCING_DILUTION,
}

SOFT_DOCUMENT_TYPES = {DocumentType.INVESTOR_COMMUNICATION}

THEME_LABELS: dict[CrossDocumentTheme, str] = {
    CrossDocumentTheme.CLINICAL_EXECUTION: "Clinical execution",
    CrossDocumentTheme.REGULATORY_POSTURE: "Regulatory posture",
    CrossDocumentTheme.FINANCING_AND_RUNWAY: "Financing and runway",
    CrossDocumentTheme.OPERATIONAL_EXECUTION: "Operational execution",
    CrossDocumentTheme.NARRATIVE_CREDIBILITY: "Narrative credibility",
}

THEME_DOCUMENT_RELEVANCE: dict[CrossDocumentTheme, dict[DocumentType, float]] = {
    CrossDocumentTheme.CLINICAL_EXECUTION: {
        DocumentType.CLINICAL_TRIAL_UPDATE: 1.00,
        DocumentType.MATERIAL_EVENT: 0.75,
        DocumentType.FDA_REVIEW: 0.65,
        DocumentType.INVESTOR_COMMUNICATION: 0.45,
    },
    CrossDocumentTheme.REGULATORY_POSTURE: {
        DocumentType.FDA_REVIEW: 1.00,
        DocumentType.MATERIAL_EVENT: 0.75,
        DocumentType.CLINICAL_TRIAL_UPDATE: 0.55,
        DocumentType.INVESTOR_COMMUNICATION: 0.40,
    },
    CrossDocumentTheme.FINANCING_AND_RUNWAY: {
        DocumentType.FINANCING_DILUTION: 1.00,
        DocumentType.MATERIAL_EVENT: 0.70,
        DocumentType.INVESTOR_COMMUNICATION: 0.50,
    },
    CrossDocumentTheme.OPERATIONAL_EXECUTION: {
        DocumentType.MATERIAL_EVENT: 0.85,
        DocumentType.CLINICAL_TRIAL_UPDATE: 0.90,
        DocumentType.FDA_REVIEW: 0.70,
        DocumentType.FINANCING_DILUTION: 0.55,
        DocumentType.INVESTOR_COMMUNICATION: 0.45,
    },
    CrossDocumentTheme.NARRATIVE_CREDIBILITY: {
        DocumentType.INVESTOR_COMMUNICATION: 1.00,
        DocumentType.MATERIAL_EVENT: 0.80,
        DocumentType.CLINICAL_TRIAL_UPDATE: 0.80,
        DocumentType.FDA_REVIEW: 0.85,
        DocumentType.FINANCING_DILUTION: 0.80,
    },
}


def stable_unique(items: Sequence[Hashable]) -> list[Any]:
    seen: set[Hashable] = set()
    ordered: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def document_type_label(document_type: DocumentType) -> str:
    return DISCLOSURE_TYPE_LABELS.get(document_type.value, document_type.value.replace("_", " "))


def make_arbiter_issue(
    issue_code: str,
    message: str,
    *,
    severity: AnalysisIssueSeverity = AnalysisIssueSeverity.WARNING,
    document_types: Sequence[DocumentType] | None = None,
    worker_names: Sequence[str] | None = None,
    recoverable: bool = True,
    metadata: dict[str, Any] | None = None,
) -> ArbiterIssue:
    return ArbiterIssue(
        issue_code=issue_code,
        message=message,
        severity=severity,
        document_types=list(document_types or []),
        worker_names=list(worker_names or []),
        recoverable=recoverable,
        metadata=metadata or {},
    )


def make_arbiter_warning(
    issue_code: str,
    message: str,
    *,
    document_types: Sequence[DocumentType] | None = None,
    worker_names: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArbiterWarning:
    return ArbiterWarning(
        issue_code=issue_code,
        message=message,
        document_types=list(document_types or []),
        worker_names=list(worker_names or []),
        metadata=metadata or {},
    )


def worker_status_rank(status: ProcessingStatus) -> int:
    ranking = {
        ProcessingStatus.SUCCESS: 5,
        ProcessingStatus.PARTIAL: 4,
        ProcessingStatus.PENDING: 3,
        ProcessingStatus.NO_DOCUMENT: 2,
        ProcessingStatus.RETRIEVAL_FAILED: 1,
        ProcessingStatus.SELECTION_FAILED: 1,
        ProcessingStatus.EXTRACTION_FAILED: 1,
        ProcessingStatus.ANALYSIS_FAILED: 1,
    }
    return ranking.get(status, 0)


def worker_output_quality_key(worker_output: WorkerOutput) -> tuple[float, float, int, int]:
    sentiment_confidence = worker_output.sentiment.confidence if worker_output.sentiment else None
    confidence = worker_output.confidence if worker_output.confidence is not None else sentiment_confidence or 0.0
    return (float(worker_status_rank(worker_output.status)), float(confidence), len(worker_output.evidence), len(worker_output.key_points))


def select_best_worker_output_per_type(
    worker_outputs: Sequence[WorkerOutput],
) -> tuple[list[WorkerOutput], list[ArbiterWarning]]:
    grouped: dict[DocumentType, list[WorkerOutput]] = defaultdict(list)
    for worker_output in worker_outputs:
        grouped[worker_output.document_type].append(worker_output)

    selected_outputs: list[WorkerOutput] = []
    warnings: list[ArbiterWarning] = []
    for document_type in EXPECTED_DOCUMENT_TYPES:
        group = grouped.get(document_type, [])
        if not group:
            continue
        if len(group) > 1:
            ordered_group = sorted(group, key=worker_output_quality_key, reverse=True)
            selected_outputs.append(ordered_group[0])
            warnings.append(
                make_arbiter_warning(
                    "duplicate_worker_outputs",
                    f"Multiple worker outputs were provided for {document_type_label(document_type)}; the strongest one was kept.",
                    document_types=[document_type],
                    worker_names=[output.worker_name for output in group],
                    metadata={"kept_worker": ordered_group[0].worker_name},
                )
            )
            continue
        selected_outputs.append(group[0])

    return selected_outputs, warnings


def flatten_worker_evidence_references(worker_output: WorkerOutput) -> list[ArbiterEvidenceReference]:
    return [
        ArbiterEvidenceReference(
            worker_name=worker_output.worker_name,
            document_type=worker_output.document_type,
            evidence_id=evidence.evidence_id,
            document_id=evidence.document_id,
            source_url=evidence.source_url,
            source_chunk_id=evidence.source_chunk_id,
            source_section_id=evidence.source_section_id,
            section_title=evidence.section_title,
            interpretation=evidence.interpretation,
            snippet_text=evidence.snippet_text,
            rationale=evidence.rationale,
        )
        for evidence in worker_output.evidence
    ]


def deduplicate_arbiter_evidence(
    evidence_references: Sequence[ArbiterEvidenceReference],
) -> list[ArbiterEvidenceReference]:
    deduplicated: dict[tuple[Any, ...], ArbiterEvidenceReference] = {}
    for reference in evidence_references:
        reference_key = (
            reference.worker_name,
            reference.document_type.value,
            reference.document_id,
            reference.evidence_id,
            reference.source_chunk_id,
            reference.snippet_text,
        )
        if reference_key not in deduplicated:
            deduplicated[reference_key] = reference
    return list(deduplicated.values())


def deduplicate_evidence_snippets(evidence_snippets: Sequence[EvidenceSnippet]) -> list[EvidenceSnippet]:
    deduplicated: dict[tuple[Any, ...], EvidenceSnippet] = {}
    for snippet in evidence_snippets:
        snippet_key = (
            snippet.document_id,
            snippet.evidence_id,
            snippet.source_chunk_id,
            snippet.snippet_text,
            snippet.rationale,
        )
        if snippet_key not in deduplicated:
            deduplicated[snippet_key] = snippet
    return list(deduplicated.values())


def convert_worker_pipeline_issue_to_arbiter_issue(
    worker_output: WorkerOutput,
    issue: PipelineError,
) -> ArbiterIssue:
    severity_value = str(issue.details.get("severity", AnalysisIssueSeverity.WARNING.value))
    severity = AnalysisIssueSeverity(severity_value) if severity_value in AnalysisIssueSeverity._value2member_map_ else AnalysisIssueSeverity.WARNING
    return make_arbiter_issue(
        issue.error_code,
        issue.message,
        severity=severity,
        document_types=[worker_output.document_type] if worker_output.document_type else [],
        worker_names=[worker_output.worker_name],
        recoverable=issue.recoverable,
        metadata={"stage": issue.stage, **issue.details},
    )


def extract_worker_signal_profile(normalized_output: "NormalizedWorkerOutput") -> dict[str, Any]:
    from pipeline.models import NormalizedWorkerOutput as _NWO  # noqa: F401

    return {
        "worker_name": normalized_output.worker_name,
        "document_type": normalized_output.document_type.value,
        "status": normalized_output.status.value,
        "direction": normalized_output.direction.value,
        "sentiment_label": normalized_output.sentiment_label.value,
        "sentiment_score": normalized_output.sentiment_score,
        "normalized_confidence": round(normalized_output.normalized_confidence, 3),
        "evidence_density": round(normalized_output.evidence_density, 3),
        "evidence_count": normalized_output.evidence_count,
        "key_point_count": normalized_output.key_point_count,
        "fogging_score": normalized_output.fogging_score,
        "hedging_score": normalized_output.hedging_score,
        "promotional_score": normalized_output.promotional_score,
    }
