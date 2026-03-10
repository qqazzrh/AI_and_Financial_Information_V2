"""
Master node: assembles final UI-facing payload from worker + arbiter outputs.

Contains the IntegratedMasterNode and all helper functions for building
the master summary, aggregating sentiment/tone, and extracting signals.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pipeline.models import (
    MasterNode,
    MasterInput,
    MasterOutput,
    FinalUIPayload,
    DisclosureCardPayload,
    WorkerOutput,
    ArbiterOutput,
    ArbiterIssue,
    PipelineError,
    ProvenanceRecord,
    SentimentAssessment,
    ToneAssessment,
    DocumentMetadata,
    RetrievalResult,
)
from pipeline.enums import (
    DocumentType,
    ProcessingStatus,
    SentimentLabel,
    NormalizedSignalDirection,
)
from pipeline.config import now_utc
from pipeline.retrieval.base import (
    build_provenance_record,
    build_document_metadata_from_candidate,
)
from pipeline.arbiter.models import (
    USABLE_WORKER_STATUSES,
    EXPECTED_DOCUMENT_TYPES,
    stable_unique,
)
from pipeline.arbiter.normalization import safe_mean
from pipeline.analysis.rubrics import normalize_sentiment_label
from pipeline.processing.sections import selected_document_from_retrieval_result


__all__ = [
    "arbiter_issue_to_pipeline_error",
    "build_disclosure_card_payload",
    "extract_positive_signals",
    "extract_negative_signals",
    "extract_uncertainties",
    "extract_story_substance_flags",
    "build_master_summary",
    "aggregate_master_sentiment",
    "aggregate_master_tone",
    "IntegratedMasterNode",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def arbiter_issue_to_pipeline_error(arbiter_issue: ArbiterIssue) -> PipelineError:
    return PipelineError(
        error_code=arbiter_issue.issue_code,
        message=arbiter_issue.message,
        stage="arbiter",
        document_type=arbiter_issue.document_types[0] if arbiter_issue.document_types else None,
        recoverable=arbiter_issue.recoverable,
        details={
            "severity": arbiter_issue.severity.value,
            "document_types": [document_type.value for document_type in arbiter_issue.document_types],
            "worker_names": list(arbiter_issue.worker_names),
            **arbiter_issue.metadata,
        },
    )


def build_disclosure_card_payload(
    worker_output: WorkerOutput,
    retrieval_result: RetrievalResult | None = None,
) -> DisclosureCardPayload:
    metadata = worker_output.document_metadata
    if metadata is None and retrieval_result is not None and retrieval_result.selected_candidate is not None:
        metadata = build_document_metadata_from_candidate(retrieval_result.selected_candidate)

    return DisclosureCardPayload(
        document_type=worker_output.document_type,
        worker_name=worker_output.worker_name,
        status=worker_output.status,
        title=metadata.title if metadata else None,
        source_name=metadata.source_name if metadata else None,
        source_url=metadata.source_url if metadata else None,
        summary=worker_output.summary,
        sentiment_label=worker_output.sentiment.label if worker_output.sentiment else None,
        sentiment_score=worker_output.sentiment.score if worker_output.sentiment else None,
        confidence=worker_output.confidence,
        fogging_score=worker_output.tone.fogging_score if worker_output.tone else None,
        hedging_score=worker_output.tone.hedging_score if worker_output.tone else None,
        promotional_score=worker_output.tone.promotional_score if worker_output.tone else None,
        key_points=list(worker_output.key_points),
        caveats=list(worker_output.caveats),
        warnings=[warning.message for warning in worker_output.warnings],
        evidence=list(worker_output.evidence),
        provenance=list(worker_output.provenance),
    )


def extract_positive_signals(arbiter_output: ArbiterOutput | None, worker_outputs: Sequence[WorkerOutput]) -> list[str]:
    signals: list[str] = []
    if arbiter_output is not None:
        signals.extend(group.description for group in arbiter_output.positive_signal_groups)
        signals.extend(judgment.summary for judgment in arbiter_output.cross_document_judgments if judgment.direction == NormalizedSignalDirection.POSITIVE)
    if not signals:
        signals.extend(output.key_points[0] for output in worker_outputs if output.key_points and output.sentiment and output.sentiment.label == SentimentLabel.POSITIVE)
    return stable_unique([signal for signal in signals if signal])


def extract_negative_signals(arbiter_output: ArbiterOutput | None, worker_outputs: Sequence[WorkerOutput]) -> list[str]:
    signals: list[str] = []
    if arbiter_output is not None:
        signals.extend(group.description for group in arbiter_output.negative_signal_groups)
        signals.extend(conflict.description for conflict in arbiter_output.conflicting_signals)
        signals.extend(judgment.summary for judgment in arbiter_output.cross_document_judgments if judgment.direction == NormalizedSignalDirection.NEGATIVE)
    if not signals:
        signals.extend(output.key_points[0] for output in worker_outputs if output.key_points and output.sentiment and output.sentiment.label == SentimentLabel.NEGATIVE)
    return stable_unique([signal for signal in signals if signal])


def extract_uncertainties(arbiter_output: ArbiterOutput | None, worker_outputs: Sequence[WorkerOutput]) -> list[str]:
    signals: list[str] = []
    if arbiter_output is not None:
        signals.extend(finding.summary for finding in arbiter_output.unresolved_uncertainties)
        signals.extend(arbiter_output.missing_coverage_notes)
    if not signals:
        signals.extend(output.caveats[0] for output in worker_outputs if output.caveats)
    return stable_unique([signal for signal in signals if signal])


def extract_story_substance_flags(arbiter_output: ArbiterOutput | None, worker_outputs: Sequence[WorkerOutput]) -> list[str]:
    flags: list[str] = []
    if arbiter_output is not None:
        flags.extend(finding.summary for finding in arbiter_output.fogging_or_story_substance_flags)
    if not flags:
        flags.extend(output.caveats[0] for output in worker_outputs if output.caveats)
    return stable_unique([flag for flag in flags if flag])


def build_master_summary(
    arbiter_output: ArbiterOutput | None,
    worker_outputs: Sequence[WorkerOutput],
    *,
    positive_signals: Sequence[str],
    negative_signals: Sequence[str],
    uncertainties: Sequence[str],
) -> str:
    summary_parts: list[str] = []
    if arbiter_output is not None and arbiter_output.summary:
        summary_parts.append(arbiter_output.summary)
    else:
        usable_outputs = [output for output in worker_outputs if output.status in USABLE_WORKER_STATUSES]
        summary_parts.append(f"Integrated review covered {len(usable_outputs)} disclosure(s) with explicit worker, arbiter, and master handoff.")
    if positive_signals:
        summary_parts.append(f"Positive emphasis: {positive_signals[0]}")
    if negative_signals:
        summary_parts.append(f"Negative emphasis: {negative_signals[0]}")
    if uncertainties:
        summary_parts.append(f"Primary uncertainty: {uncertainties[0]}")
    return " ".join(summary_parts).strip()


def _label_to_fallback_score(label: SentimentLabel) -> float | None:
    """Map a sentiment label to a synthetic score when no numeric score is available."""
    mapping = {
        SentimentLabel.POSITIVE: 0.40,
        SentimentLabel.NEGATIVE: -0.40,
        SentimentLabel.NEUTRAL: 0.0,
        SentimentLabel.MIXED: 0.06,
        SentimentLabel.INSUFFICIENT_EVIDENCE: None,
    }
    return mapping.get(label)


def aggregate_master_sentiment(
    arbiter_output: ArbiterOutput | None,
    worker_outputs: Sequence[WorkerOutput],
) -> SentimentAssessment | None:
    # Use arbiter sentiment UNLESS it says insufficient_evidence while workers have data
    # Collect scores, falling back to label-derived scores when numeric scores are missing
    usable_scores: list[float] = []
    for output in worker_outputs:
        if output.sentiment:
            if output.sentiment.score is not None:
                usable_scores.append(output.sentiment.score)
            elif output.sentiment.label not in (None, SentimentLabel.INSUFFICIENT_EVIDENCE):
                fallback = _label_to_fallback_score(output.sentiment.label)
                if fallback is not None:
                    usable_scores.append(fallback)
    if arbiter_output is not None and arbiter_output.sentiment is not None:
        if arbiter_output.sentiment.label != SentimentLabel.INSUFFICIENT_EVIDENCE:
            return arbiter_output.sentiment
        # Arbiter said insufficient_evidence - fall through to worker aggregation if workers have data
        if not usable_scores:
            return arbiter_output.sentiment
    if not usable_scores:
        return None
    # Weight non-neutral scores 2x to make overall sentiment more responsive
    weights = []
    for output in worker_outputs:
        if output.sentiment:
            if output.sentiment.score is not None:
                w = 2.0 if output.sentiment.label not in (SentimentLabel.NEUTRAL, SentimentLabel.INSUFFICIENT_EVIDENCE, None) else 1.0
                weights.append((output.sentiment.score, w))
            elif output.sentiment.label not in (None, SentimentLabel.INSUFFICIENT_EVIDENCE):
                fallback = _label_to_fallback_score(output.sentiment.label)
                if fallback is not None:
                    w = 2.0 if output.sentiment.label not in (SentimentLabel.NEUTRAL,) else 1.0
                    weights.append((fallback, w))
    if weights:
        total_weight = sum(w for _, w in weights)
        aggregate_score = sum(s * w for s, w in weights) / total_weight if total_weight > 0 else 0.0
    else:
        aggregate_score = safe_mean(usable_scores)
    has_positive = any(output.sentiment and output.sentiment.label == SentimentLabel.POSITIVE for output in worker_outputs)
    has_negative = any(output.sentiment and output.sentiment.label == SentimentLabel.NEGATIVE for output in worker_outputs)
    # Use worker label directions to boost aggregate when scores are near thresholds
    if has_positive and has_negative:
        inferred_label = SentimentLabel.MIXED
    elif has_positive and aggregate_score > 0:
        inferred_label = SentimentLabel.POSITIVE
    elif has_negative and aggregate_score < 0:
        inferred_label = SentimentLabel.NEGATIVE
    else:
        inferred_label = normalize_sentiment_label(None, aggregate_score)[0]
    return SentimentAssessment(
        label=inferred_label,
        score=aggregate_score,
        confidence=safe_mean([output.confidence for output in worker_outputs]),
        rationale="Master sentiment aggregated from worker outputs with label-aware direction boosting.",
    )


def aggregate_master_tone(
    arbiter_output: ArbiterOutput | None,
    worker_outputs: Sequence[WorkerOutput],
) -> ToneAssessment | None:
    if arbiter_output is not None and arbiter_output.tone is not None:
        return arbiter_output.tone
    tone_values = [output.tone for output in worker_outputs if output.tone is not None]
    if not tone_values:
        return None
    return ToneAssessment(
        fogging_score=safe_mean([tone.fogging_score for tone in tone_values]),
        hedging_score=safe_mean([tone.hedging_score for tone in tone_values]),
        promotional_score=safe_mean([tone.promotional_score for tone in tone_values]),
        confidence=safe_mean([tone.confidence for tone in tone_values]),
        rationale="Fallback master tone aggregated directly from worker outputs.",
    )


# ---------------------------------------------------------------------------
# IntegratedMasterNode
# ---------------------------------------------------------------------------

class IntegratedMasterNode(MasterNode):
    """Concrete master node that emits the final UI-facing payload."""

    def build_master_output(self, master_input: MasterInput) -> MasterOutput:
        retrieval_result_map = {result.request.document_type: result for result in master_input.retrieval_results}
        worker_outputs = sorted(master_input.worker_outputs, key=lambda output: EXPECTED_DOCUMENT_TYPES.index(output.document_type) if output.document_type in EXPECTED_DOCUMENT_TYPES else len(EXPECTED_DOCUMENT_TYPES))
        arbiter_output = master_input.arbiter_outputs[0] if master_input.arbiter_outputs else None

        disclosures = [
            build_disclosure_card_payload(worker_output, retrieval_result_map.get(worker_output.document_type))
            for worker_output in worker_outputs
        ]
        positive_signals = extract_positive_signals(arbiter_output, worker_outputs)
        negative_signals = extract_negative_signals(arbiter_output, worker_outputs)
        uncertainties = extract_uncertainties(arbiter_output, worker_outputs)
        story_flags = extract_story_substance_flags(arbiter_output, worker_outputs)
        master_sentiment = aggregate_master_sentiment(arbiter_output, worker_outputs)
        master_tone = aggregate_master_tone(arbiter_output, worker_outputs)
        missing_document_types = list(arbiter_output.missing_document_types) if arbiter_output is not None else [document_type for document_type in EXPECTED_DOCUMENT_TYPES if document_type not in {output.document_type for output in worker_outputs}]

        warnings = stable_unique(
            [warning.message for output in worker_outputs for warning in output.warnings]
            + ([warning.message for warning in arbiter_output.warnings] if arbiter_output is not None else [])
        )
        issues = [issue for output in worker_outputs for issue in output.issues]
        if arbiter_output is not None:
            issues.extend(arbiter_issue_to_pipeline_error(issue) for issue in arbiter_output.issues)

        provenance = [
            build_provenance_record(
                stage="master_payload",
                adapter_name="master_node",
                document_type=None,
                note="Master node assembled the only UI-facing payload.",
                metadata={
                    "worker_output_count": len(worker_outputs),
                    "arbiter_output_count": len(master_input.arbiter_outputs),
                    "missing_document_type_count": len(missing_document_types),
                },
            )
        ]
        for disclosure in disclosures:
            provenance.extend(disclosure.provenance[:1])
        if arbiter_output is not None:
            provenance.append(
                build_provenance_record(
                    stage="master_payload",
                    adapter_name=arbiter_output.arbiter_name,
                    note="Master node consumed ArbiterOutput for final UI assembly.",
                    metadata={"arbiter_id": arbiter_output.arbiter_id},
                )
            )

        status = ProcessingStatus.SUCCESS
        if missing_document_types or negative_signals or uncertainties or story_flags or warnings:
            status = ProcessingStatus.PARTIAL
        if not worker_outputs:
            status = ProcessingStatus.NO_DOCUMENT

        return MasterOutput(
            ticker=master_input.ticker,
            status=status,
            master_summary=build_master_summary(
                arbiter_output,
                worker_outputs,
                positive_signals=positive_signals,
                negative_signals=negative_signals,
                uncertainties=uncertainties,
            ),
            master_sentiment=master_sentiment,
            master_tone=master_tone,
            disclosures=disclosures,
            missing_document_types=missing_document_types,
            key_positive_signals=positive_signals[:6],
            key_negative_signals=negative_signals[:6],
            key_uncertainties=uncertainties[:6],
            fogging_or_story_substance_flags=story_flags[:6],
            warnings=warnings,
            issues=issues,
            provenance=provenance,
            ready_for_ui=True,
        )

    def build_payload(self, master_input: MasterInput) -> FinalUIPayload:
        master_output = self.build_master_output(master_input)
        master_sentiment = master_output.master_sentiment
        master_tone = master_output.master_tone
        return FinalUIPayload(
            ticker=master_output.ticker,
            generated_at=master_output.generated_at,
            status=master_output.status,
            overall_summary=master_output.master_summary,
            overall_sentiment_label=master_sentiment.label if master_sentiment else None,
            overall_sentiment_score=master_sentiment.score if master_sentiment else None,
            overall_confidence=master_sentiment.confidence if master_sentiment else None,
            overall_fogging_score=master_tone.fogging_score if master_tone else None,
            overall_hedging_score=master_tone.hedging_score if master_tone else None,
            overall_promotional_score=master_tone.promotional_score if master_tone else None,
            key_positive_signals=list(master_output.key_positive_signals),
            key_negative_signals=list(master_output.key_negative_signals),
            key_uncertainties=list(master_output.key_uncertainties),
            fogging_or_story_substance_flags=list(master_output.fogging_or_story_substance_flags),
            disclosures=list(master_output.disclosures),
            missing_document_types=list(master_output.missing_document_types),
            system_warnings=list(master_output.warnings),
            issues=list(master_output.issues),
            provenance=list(master_output.provenance),
        )
