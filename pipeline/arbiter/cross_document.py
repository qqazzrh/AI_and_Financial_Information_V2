"""Cross-document arbitration: theme building, signal grouping, conflict detection, and arbiter classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from typing import Any, ClassVar, Sequence, Type

from pipeline.config import SENTIMENT_SCORE_MIN, SENTIMENT_SCORE_MAX
from pipeline.enums import (
    ArbiterDecisionType,
    ArbiterKind,
    ArbiterSignalCategory,
    CrossDocumentTheme,
    DocumentType,
    NormalizedSignalDirection,
    ProcessingStatus,
    SentimentLabel,
)
from pipeline.models import (
    ArbiterConflict,
    ArbiterEvidenceReference,
    ArbiterFinding,
    ArbiterInput,
    ArbiterOutput,
    ArbiterSignalGroup,
    ArbiterWarning,
    CrossDocumentJudgment,
    NormalizedWorkerOutput,
    SentimentAssessment,
    ThematicWorkerSignal,
    ToneAssessment,
)

from pipeline.arbiter.models import (
    EXPECTED_DOCUMENT_TYPES,
    STRUCTURED_DOCUMENT_TYPES,
    THEME_DOCUMENT_RELEVANCE,
    THEME_LABELS,
    USABLE_WORKER_STATUSES,
    deduplicate_arbiter_evidence,
    deduplicate_evidence_snippets,
    document_type_label,
    select_best_worker_output_per_type,
    stable_unique,
)
from pipeline.arbiter.normalization import (
    clamp_value,
    normalize_worker_output_for_arbitration,
    safe_mean,
)
from pipeline.arbiter.confidence import (
    assess_cross_document_confidence,
    build_arbiter_warnings,
    build_missing_coverage_notes,
)

__all__ = [
    "build_cross_document_themes",
    "summarize_theme_support",
    "build_theme_judgment",
    "detect_cross_document_alignment",
    "signals_form_conflict",
    "detect_cross_document_conflicts",
    "compare_worker_outputs",
    "make_finding_from_judgment",
    "make_signal_group_from_judgment",
    "group_positive_signals",
    "group_negative_signals",
    "build_sparse_worker_uncertainty_finding",
    "group_uncertainties",
    "detect_story_vs_substance_mismatch",
    "group_fogging_concerns",
    "aggregate_arbiter_sentiment",
    "aggregate_arbiter_tone",
    "BaseArbiter",
    "CrossDocumentArbiter",
    "ARBITER_REGISTRY",
]


# ---------------------------------------------------------------------------
# Theme building
# ---------------------------------------------------------------------------

def build_cross_document_themes(
    normalized_outputs: Sequence[NormalizedWorkerOutput],
) -> dict[CrossDocumentTheme, list[ThematicWorkerSignal]]:
    theme_map: dict[CrossDocumentTheme, list[ThematicWorkerSignal]] = defaultdict(list)
    for normalized_output in normalized_outputs:
        for theme, relevance_map in THEME_DOCUMENT_RELEVANCE.items():
            relevance_weight = relevance_map.get(normalized_output.document_type, 0.0)
            if relevance_weight <= 0.0:
                continue
            adjusted_confidence = clamp_value(
                normalized_output.normalized_confidence * relevance_weight * (0.50 + (0.50 * normalized_output.evidence_density)),
                0.0,
                1.0,
            )
            theme_map[theme].append(
                ThematicWorkerSignal(
                    theme=theme,
                    worker_name=normalized_output.worker_name,
                    document_type=normalized_output.document_type,
                    direction=normalized_output.direction,
                    relevance_weight=relevance_weight,
                    adjusted_confidence=adjusted_confidence,
                    evidence_density=normalized_output.evidence_density,
                    sentiment_score=normalized_output.sentiment_score,
                    fogging_score=normalized_output.fogging_score,
                    hedging_score=normalized_output.hedging_score,
                    promotional_score=normalized_output.promotional_score,
                    is_structured_document=normalized_output.is_structured_document,
                    is_soft_document=normalized_output.is_soft_document,
                    summary=normalized_output.summary,
                    evidence_references=list(normalized_output.evidence_references),
                )
            )
    return dict(theme_map)


# ---------------------------------------------------------------------------
# Theme support & judgment
# ---------------------------------------------------------------------------

def summarize_theme_support(theme_signals: Sequence[ThematicWorkerSignal]) -> dict[str, Any]:
    support: dict[NormalizedSignalDirection, float] = defaultdict(float)
    supporting_docs: dict[NormalizedSignalDirection, list[DocumentType]] = defaultdict(list)
    supporting_workers: dict[NormalizedSignalDirection, list[str]] = defaultdict(list)
    supporting_evidence: dict[NormalizedSignalDirection, list[ArbiterEvidenceReference]] = defaultdict(list)
    for signal in theme_signals:
        support[signal.direction] += signal.adjusted_confidence
        supporting_docs[signal.direction].append(signal.document_type)
        supporting_workers[signal.direction].append(signal.worker_name)
        supporting_evidence[signal.direction].extend(signal.evidence_references[:2])
    return {
        "support": support,
        "supporting_docs": {direction: stable_unique(values) for direction, values in supporting_docs.items()},
        "supporting_workers": {direction: stable_unique(values) for direction, values in supporting_workers.items()},
        "supporting_evidence": {direction: deduplicate_arbiter_evidence(values) for direction, values in supporting_evidence.items()},
    }


def build_theme_judgment(theme: CrossDocumentTheme, theme_signals: Sequence[ThematicWorkerSignal]) -> CrossDocumentJudgment:
    theme_support = summarize_theme_support(theme_signals)
    support = theme_support["support"]
    positive_support = float(support.get(NormalizedSignalDirection.POSITIVE, 0.0))
    negative_support = float(support.get(NormalizedSignalDirection.NEGATIVE, 0.0))
    mixed_support = float(support.get(NormalizedSignalDirection.MIXED, 0.0))
    uncertain_support = float(support.get(NormalizedSignalDirection.UNCERTAIN, 0.0))
    neutral_support = float(support.get(NormalizedSignalDirection.NEUTRAL, 0.0))
    participating_document_types = stable_unique([signal.document_type for signal in theme_signals])
    participating_worker_names = stable_unique([signal.worker_name for signal in theme_signals])
    strongest_support = max(positive_support, negative_support, mixed_support, uncertain_support, neutral_support)
    confidence = clamp_value(strongest_support / max(1.0, len(participating_document_types)), 0.0, 1.0)
    positive_docs = theme_support["supporting_docs"].get(NormalizedSignalDirection.POSITIVE, [])
    negative_docs = theme_support["supporting_docs"].get(NormalizedSignalDirection.NEGATIVE, [])
    uncertain_docs = theme_support["supporting_docs"].get(NormalizedSignalDirection.UNCERTAIN, [])

    if positive_support >= 0.45 and negative_support >= 0.45:
        decision_type = ArbiterDecisionType.CONTRADICTORY_SIGNAL
        direction = NormalizedSignalDirection.MIXED
        summary = f"{THEME_LABELS[theme]} is contradictory across the current disclosures rather than converging on one direction."
        supporting_document_types = positive_docs
        opposing_document_types = negative_docs
        evidence_references = deduplicate_arbiter_evidence(theme_support["supporting_evidence"].get(NormalizedSignalDirection.POSITIVE, []) + theme_support["supporting_evidence"].get(NormalizedSignalDirection.NEGATIVE, []))
    elif negative_support >= 0.60 and negative_support > positive_support + 0.15:
        has_structured_negative_support = any(signal.is_structured_document and signal.direction == NormalizedSignalDirection.NEGATIVE and signal.adjusted_confidence >= 0.30 for signal in theme_signals)
        decision_type = ArbiterDecisionType.MATERIAL_CONCERN if has_structured_negative_support else ArbiterDecisionType.ALIGNED_SIGNAL
        direction = NormalizedSignalDirection.NEGATIVE
        summary = f"{THEME_LABELS[theme]} skews negative across the currently available disclosures."
        supporting_document_types = negative_docs
        opposing_document_types = positive_docs
        evidence_references = theme_support["supporting_evidence"].get(NormalizedSignalDirection.NEGATIVE, [])
    elif positive_support >= 0.60 and positive_support > negative_support + 0.15:
        decision_type = ArbiterDecisionType.MATERIAL_POSITIVE if len(positive_docs) >= 2 else ArbiterDecisionType.ALIGNED_SIGNAL
        direction = NormalizedSignalDirection.POSITIVE
        summary = f"{THEME_LABELS[theme]} is supported positively across the currently available disclosures."
        supporting_document_types = positive_docs
        opposing_document_types = negative_docs
        evidence_references = theme_support["supporting_evidence"].get(NormalizedSignalDirection.POSITIVE, [])
    elif uncertain_support + mixed_support >= max(positive_support, negative_support):
        decision_type = ArbiterDecisionType.CROSS_DOCUMENT_UNCERTAINTY if len(participating_document_types) >= 2 else ArbiterDecisionType.UNRESOLVED_AMBIGUITY
        direction = NormalizedSignalDirection.UNCERTAIN
        summary = f"{THEME_LABELS[theme]} remains uncertain because current disclosures do not resolve the key open questions."
        supporting_document_types = uncertain_docs
        opposing_document_types = positive_docs + negative_docs
        evidence_references = deduplicate_arbiter_evidence(theme_support["supporting_evidence"].get(NormalizedSignalDirection.UNCERTAIN, []) + theme_support["supporting_evidence"].get(NormalizedSignalDirection.MIXED, []))
    elif len(participating_document_types) < 2:
        decision_type = ArbiterDecisionType.UNRESOLVED_AMBIGUITY
        direction = NormalizedSignalDirection.UNCERTAIN
        summary = f"{THEME_LABELS[theme]} has only thin cross-document coverage so far."
        supporting_document_types = participating_document_types
        opposing_document_types = []
        evidence_references = deduplicate_arbiter_evidence([reference for signal in theme_signals for reference in signal.evidence_references[:2]])
    else:
        decision_type = ArbiterDecisionType.UNRESOLVED_AMBIGUITY
        direction = NormalizedSignalDirection.MIXED
        summary = f"{THEME_LABELS[theme]} is directionally mixed and does not yet support a clean synthesis."
        supporting_document_types = positive_docs + negative_docs
        opposing_document_types = uncertain_docs
        evidence_references = deduplicate_arbiter_evidence([reference for signal in theme_signals for reference in signal.evidence_references[:2]])

    return CrossDocumentJudgment(
        judgment_id=f"judgment_{theme.value}",
        theme=theme,
        decision_type=decision_type,
        direction=direction,
        summary=summary,
        supporting_document_types=stable_unique(supporting_document_types),
        opposing_document_types=stable_unique(opposing_document_types),
        worker_names=participating_worker_names,
        confidence=confidence,
        evidence_references=deduplicate_arbiter_evidence(evidence_references),
        reasoning_notes=[
            f"positive_support={positive_support:.2f}",
            f"negative_support={negative_support:.2f}",
            f"mixed_support={mixed_support:.2f}",
            f"uncertain_support={uncertain_support:.2f}",
        ],
    )


# ---------------------------------------------------------------------------
# Alignment & conflict detection
# ---------------------------------------------------------------------------

def detect_cross_document_alignment(judgments: Sequence[CrossDocumentJudgment]) -> list[CrossDocumentJudgment]:
    aligned_types = {ArbiterDecisionType.ALIGNED_SIGNAL, ArbiterDecisionType.MATERIAL_CONCERN, ArbiterDecisionType.MATERIAL_POSITIVE}
    return [judgment for judgment in judgments if judgment.decision_type in aligned_types]


def signals_form_conflict(left_signal: ThematicWorkerSignal, right_signal: ThematicWorkerSignal) -> bool:
    direction_pair = {left_signal.direction, right_signal.direction}
    if direction_pair == {NormalizedSignalDirection.POSITIVE, NormalizedSignalDirection.NEGATIVE}:
        return True
    if direction_pair == {NormalizedSignalDirection.POSITIVE, NormalizedSignalDirection.UNCERTAIN}:
        return left_signal.is_soft_document != right_signal.is_soft_document
    return False


def detect_cross_document_conflicts(
    theme_map: dict[CrossDocumentTheme, list[ThematicWorkerSignal]],
) -> list[ArbiterConflict]:
    conflicts: list[ArbiterConflict] = []
    for theme, theme_signals in theme_map.items():
        for left_signal, right_signal in combinations(theme_signals, 2):
            if not signals_form_conflict(left_signal, right_signal):
                continue
            if min(left_signal.adjusted_confidence, right_signal.adjusted_confidence) < 0.20:
                continue
            positive_documents: list[DocumentType] = []
            negative_documents: list[DocumentType] = []
            if left_signal.direction == NormalizedSignalDirection.POSITIVE:
                positive_documents.append(left_signal.document_type)
            elif left_signal.direction in {NormalizedSignalDirection.NEGATIVE, NormalizedSignalDirection.UNCERTAIN}:
                negative_documents.append(left_signal.document_type)
            if right_signal.direction == NormalizedSignalDirection.POSITIVE:
                positive_documents.append(right_signal.document_type)
            elif right_signal.direction in {NormalizedSignalDirection.NEGATIVE, NormalizedSignalDirection.UNCERTAIN}:
                negative_documents.append(right_signal.document_type)
            conflicts.append(
                ArbiterConflict(
                    conflict_id=f"conflict_{theme.value}_{left_signal.document_type.value}_{right_signal.document_type.value}",
                    theme=theme,
                    title=f"{THEME_LABELS[theme]} conflict",
                    description=f"{document_type_label(left_signal.document_type)} and {document_type_label(right_signal.document_type)} do not support the same story for {THEME_LABELS[theme].lower()}.",
                    positive_document_types=stable_unique(positive_documents),
                    negative_document_types=stable_unique(negative_documents),
                    worker_names=stable_unique([left_signal.worker_name, right_signal.worker_name]),
                    high_confidence_conflict=left_signal.adjusted_confidence >= 0.45 and right_signal.adjusted_confidence >= 0.45,
                    evidence_references=deduplicate_arbiter_evidence(left_signal.evidence_references[:2] + right_signal.evidence_references[:2]),
                    reasoning_notes=[
                        f"left_direction={left_signal.direction.value}",
                        f"right_direction={right_signal.direction.value}",
                        f"left_adjusted_confidence={left_signal.adjusted_confidence:.2f}",
                        f"right_adjusted_confidence={right_signal.adjusted_confidence:.2f}",
                    ],
                )
            )
    deduplicated_conflicts: dict[tuple[Any, ...], ArbiterConflict] = {}
    for conflict in conflicts:
        conflict_key = (conflict.theme.value, tuple(sorted(document_type.value for document_type in conflict.positive_document_types)), tuple(sorted(document_type.value for document_type in conflict.negative_document_types)))
        if conflict_key not in deduplicated_conflicts:
            deduplicated_conflicts[conflict_key] = conflict
    return list(deduplicated_conflicts.values())


def compare_worker_outputs(normalized_outputs: Sequence[NormalizedWorkerOutput]) -> dict[str, Any]:
    theme_map = build_cross_document_themes(normalized_outputs)
    judgments = [build_theme_judgment(theme, signals) for theme, signals in theme_map.items()]
    return {
        "themes": theme_map,
        "judgments": judgments,
        "aligned_judgments": detect_cross_document_alignment(judgments),
        "conflicts": detect_cross_document_conflicts(theme_map),
    }


# ---------------------------------------------------------------------------
# Signal grouping
# ---------------------------------------------------------------------------

def make_finding_from_judgment(judgment: CrossDocumentJudgment, *, category: ArbiterSignalCategory) -> ArbiterFinding:
    title_prefix = {
        ArbiterSignalCategory.POSITIVE: "Positive signal",
        ArbiterSignalCategory.NEGATIVE: "Negative signal",
        ArbiterSignalCategory.UNCERTAINTY: "Uncertainty",
        ArbiterSignalCategory.FOGGING: "Narrative concern",
    }[category]
    return ArbiterFinding(
        finding_id=f"{category.value}_{judgment.theme.value}",
        category=category,
        decision_type=judgment.decision_type,
        theme=judgment.theme,
        title=f"{title_prefix}: {THEME_LABELS[judgment.theme]}",
        summary=judgment.summary,
        supporting_document_types=list(judgment.supporting_document_types),
        contradicting_document_types=list(judgment.opposing_document_types),
        worker_names=list(judgment.worker_names),
        confidence=judgment.confidence,
        evidence_references=list(judgment.evidence_references),
        reasoning_notes=list(judgment.reasoning_notes),
    )


def make_signal_group_from_judgment(judgment: CrossDocumentJudgment, *, category: ArbiterSignalCategory) -> ArbiterSignalGroup:
    finding = make_finding_from_judgment(judgment, category=category)
    return ArbiterSignalGroup(
        group_id=f"group_{category.value}_{judgment.theme.value}",
        category=category,
        theme=judgment.theme,
        title=finding.title,
        description=finding.summary,
        document_types=stable_unique(list(judgment.supporting_document_types) + list(judgment.opposing_document_types)),
        worker_names=list(judgment.worker_names),
        findings=[finding],
        evidence_references=list(judgment.evidence_references),
        confidence=judgment.confidence,
        reasoning_notes=list(judgment.reasoning_notes),
    )


def group_positive_signals(judgments: Sequence[CrossDocumentJudgment]) -> list[ArbiterSignalGroup]:
    positive_decision_types = {ArbiterDecisionType.ALIGNED_SIGNAL, ArbiterDecisionType.MATERIAL_POSITIVE}
    return [make_signal_group_from_judgment(judgment, category=ArbiterSignalCategory.POSITIVE) for judgment in judgments if judgment.direction == NormalizedSignalDirection.POSITIVE and judgment.decision_type in positive_decision_types]


def group_negative_signals(judgments: Sequence[CrossDocumentJudgment]) -> list[ArbiterSignalGroup]:
    negative_decision_types = {ArbiterDecisionType.ALIGNED_SIGNAL, ArbiterDecisionType.MATERIAL_CONCERN}
    return [make_signal_group_from_judgment(judgment, category=ArbiterSignalCategory.NEGATIVE) for judgment in judgments if judgment.direction == NormalizedSignalDirection.NEGATIVE and judgment.decision_type in negative_decision_types]


def build_sparse_worker_uncertainty_finding(normalized_output: NormalizedWorkerOutput) -> ArbiterFinding:
    return ArbiterFinding(
        finding_id=f"worker_uncertainty_{normalized_output.document_type.value}",
        category=ArbiterSignalCategory.UNCERTAINTY,
        decision_type=ArbiterDecisionType.CROSS_DOCUMENT_UNCERTAINTY,
        theme=None,
        title=f"Sparse signal: {document_type_label(normalized_output.document_type)}",
        summary=f"{document_type_label(normalized_output.document_type)} contributes only thin support because confidence or evidence density is limited.",
        supporting_document_types=[normalized_output.document_type],
        contradicting_document_types=[],
        worker_names=[normalized_output.worker_name],
        confidence=min(normalized_output.normalized_confidence, normalized_output.evidence_density),
        evidence_references=list(normalized_output.evidence_references[:2]),
        reasoning_notes=list(normalized_output.normalization_notes),
    )


def group_uncertainties(
    judgments: Sequence[CrossDocumentJudgment],
    normalized_outputs: Sequence[NormalizedWorkerOutput],
    missing_document_types: Sequence[DocumentType],
) -> list[ArbiterFinding]:
    uncertainty_findings = [make_finding_from_judgment(judgment, category=ArbiterSignalCategory.UNCERTAINTY) for judgment in judgments if judgment.decision_type in {ArbiterDecisionType.UNRESOLVED_AMBIGUITY, ArbiterDecisionType.CROSS_DOCUMENT_UNCERTAINTY}]
    for normalized_output in normalized_outputs:
        if normalized_output.status not in USABLE_WORKER_STATUSES:
            continue
        if normalized_output.normalized_confidence < 0.45 or normalized_output.evidence_density < 0.30:
            uncertainty_findings.append(build_sparse_worker_uncertainty_finding(normalized_output))
    for document_type in missing_document_types:
        uncertainty_findings.append(
            ArbiterFinding(
                finding_id=f"missing_coverage_{document_type.value}",
                category=ArbiterSignalCategory.UNCERTAINTY,
                decision_type=ArbiterDecisionType.CROSS_DOCUMENT_UNCERTAINTY,
                theme=None,
                title=f"Missing coverage: {document_type_label(document_type)}",
                summary=f"No usable worker output is available for {document_type_label(document_type)}.",
                supporting_document_types=[document_type],
                contradicting_document_types=[],
                worker_names=[],
                confidence=0.0,
                evidence_references=[],
                reasoning_notes=["Missing coverage lowers cross-document certainty."],
            )
        )
    deduplicated_findings: dict[str, ArbiterFinding] = {}
    for finding in uncertainty_findings:
        if finding.finding_id not in deduplicated_findings:
            deduplicated_findings[finding.finding_id] = finding
    return list(deduplicated_findings.values())


def detect_story_vs_substance_mismatch(normalized_outputs: Sequence[NormalizedWorkerOutput]) -> list[ArbiterFinding]:
    investor_outputs = [output for output in normalized_outputs if output.document_type == DocumentType.INVESTOR_COMMUNICATION and output.status in USABLE_WORKER_STATUSES]
    structured_outputs = [output for output in normalized_outputs if output.document_type in STRUCTURED_DOCUMENT_TYPES and output.status in USABLE_WORKER_STATUSES]
    if not investor_outputs or not structured_outputs:
        return []
    hard_negative_or_uncertain = [output for output in structured_outputs if output.direction in {NormalizedSignalDirection.NEGATIVE, NormalizedSignalDirection.UNCERTAIN} and output.normalized_confidence >= 0.45]
    findings: list[ArbiterFinding] = []
    for investor_output in investor_outputs:
        positive_narrative = investor_output.direction == NormalizedSignalDirection.POSITIVE
        elevated_promotion = (investor_output.promotional_score or 0.0) >= 0.65
        thin_support = investor_output.evidence_density < 0.35
        if not hard_negative_or_uncertain:
            continue
        if not (positive_narrative or elevated_promotion or thin_support):
            continue
        findings.append(
            ArbiterFinding(
                finding_id=f"story_substance_{investor_output.document_type.value}",
                category=ArbiterSignalCategory.FOGGING,
                decision_type=ArbiterDecisionType.STORY_SUBSTANCE_MISMATCH,
                theme=CrossDocumentTheme.NARRATIVE_CREDIBILITY,
                title="Story-vs-substance mismatch",
                summary="Investor-facing narrative is more favorable or more polished than the harder disclosures support.",
                supporting_document_types=[investor_output.document_type],
                contradicting_document_types=stable_unique([output.document_type for output in hard_negative_or_uncertain]),
                worker_names=stable_unique([investor_output.worker_name] + [output.worker_name for output in hard_negative_or_uncertain]),
                confidence=clamp_value(safe_mean([investor_output.normalized_confidence] + [output.normalized_confidence for output in hard_negative_or_uncertain]) or 0.0, 0.0, 1.0),
                evidence_references=deduplicate_arbiter_evidence(investor_output.evidence_references[:2] + [reference for output in hard_negative_or_uncertain for reference in output.evidence_references[:1]]),
                reasoning_notes=[
                    f"investor_direction={investor_output.direction.value}",
                    f"investor_promotional_score={investor_output.promotional_score}",
                    f"investor_evidence_density={investor_output.evidence_density:.2f}",
                ],
            )
        )
    return findings


def group_fogging_concerns(
    normalized_outputs: Sequence[NormalizedWorkerOutput],
    story_mismatch_findings: Sequence[ArbiterFinding],
) -> list[ArbiterFinding]:
    fogging_findings = list(story_mismatch_findings)
    for normalized_output in normalized_outputs:
        highest_tone_score = max(normalized_output.fogging_score or 0.0, normalized_output.hedging_score or 0.0, normalized_output.promotional_score or 0.0)
        if highest_tone_score < 0.65 or normalized_output.status not in USABLE_WORKER_STATUSES:
            continue
        fogging_findings.append(
            ArbiterFinding(
                finding_id=f"tone_flag_{normalized_output.document_type.value}",
                category=ArbiterSignalCategory.FOGGING,
                decision_type=ArbiterDecisionType.STORY_SUBSTANCE_MISMATCH,
                theme=CrossDocumentTheme.NARRATIVE_CREDIBILITY,
                title=f"Elevated tone concern: {document_type_label(normalized_output.document_type)}",
                summary=f"{document_type_label(normalized_output.document_type)} carries elevated fogging, hedging, or promotional tone relative to its arbiter-ready evidence.",
                supporting_document_types=[normalized_output.document_type],
                contradicting_document_types=[],
                worker_names=[normalized_output.worker_name],
                confidence=clamp_value(highest_tone_score, 0.0, 1.0),
                evidence_references=list(normalized_output.evidence_references[:2]),
                reasoning_notes=[
                    f"fogging_score={normalized_output.fogging_score}",
                    f"hedging_score={normalized_output.hedging_score}",
                    f"promotional_score={normalized_output.promotional_score}",
                ],
            )
        )
    deduplicated_findings: dict[str, ArbiterFinding] = {}
    for finding in fogging_findings:
        if finding.finding_id not in deduplicated_findings:
            deduplicated_findings[finding.finding_id] = finding
    return list(deduplicated_findings.values())


# ---------------------------------------------------------------------------
# Sentiment & tone aggregation
# ---------------------------------------------------------------------------

def aggregate_arbiter_sentiment(
    normalized_outputs: Sequence[NormalizedWorkerOutput],
    conflicts: Sequence[ArbiterConflict],
    confidence_assessment: "ArbiterConfidenceAssessment",
) -> SentimentAssessment | None:
    from pipeline.models import ArbiterConfidenceAssessment as _ACA  # noqa: F401

    weighted_scores: list[float] = []
    weights: list[float] = []
    for normalized_output in normalized_outputs:
        if normalized_output.status not in USABLE_WORKER_STATUSES or normalized_output.sentiment_score is None:
            continue
        weight = normalized_output.normalized_confidence * (0.50 + (0.50 * normalized_output.evidence_density))
        if normalized_output.is_soft_document:
            weight *= 0.85
        weighted_scores.append(normalized_output.sentiment_score * weight)
        weights.append(weight)
    if not weights or sum(weights) == 0.0:
        return SentimentAssessment(label=SentimentLabel.INSUFFICIENT_EVIDENCE, score=None, confidence=confidence_assessment.final_confidence, rationale="Arbiter did not receive enough usable worker sentiment data to infer a directional score.")
    aggregate_score = sum(weighted_scores) / sum(weights)
    positive_present = any(output.direction == NormalizedSignalDirection.POSITIVE for output in normalized_outputs)
    negative_present = any(output.direction == NormalizedSignalDirection.NEGATIVE for output in normalized_outputs)
    if conflicts and positive_present and negative_present:
        label = SentimentLabel.MIXED
    elif aggregate_score >= 0.20:
        label = SentimentLabel.POSITIVE
    elif aggregate_score <= -0.20:
        label = SentimentLabel.NEGATIVE
    elif abs(aggregate_score) <= 0.10:
        label = SentimentLabel.NEUTRAL
    else:
        label = SentimentLabel.MIXED
    return SentimentAssessment(label=label, score=clamp_value(aggregate_score, SENTIMENT_SCORE_MIN, SENTIMENT_SCORE_MAX), confidence=confidence_assessment.final_confidence, rationale="Intermediate arbiter sentiment derived from weighted worker sentiment; the master node will decide final UI-facing sentiment later.")


def aggregate_arbiter_tone(normalized_outputs: Sequence[NormalizedWorkerOutput], confidence_assessment: "ArbiterConfidenceAssessment") -> ToneAssessment | None:
    from pipeline.models import ArbiterConfidenceAssessment as _ACA  # noqa: F401

    usable_outputs = [output for output in normalized_outputs if output.status in USABLE_WORKER_STATUSES]
    if not usable_outputs:
        return None

    def weighted_tone_mean(attribute_name: str) -> float | None:
        weighted_values: list[float] = []
        tone_weights: list[float] = []
        for normalized_output in usable_outputs:
            tone_value = getattr(normalized_output, attribute_name)
            if tone_value is None:
                continue
            weight = normalized_output.normalized_confidence
            weighted_values.append(tone_value * weight)
            tone_weights.append(weight)
        if not tone_weights or sum(tone_weights) == 0.0:
            return None
        return clamp_value(sum(weighted_values) / sum(tone_weights), 0.0, 1.0)

    fogging_score = weighted_tone_mean("fogging_score")
    hedging_score = weighted_tone_mean("hedging_score")
    promotional_score = weighted_tone_mean("promotional_score")
    if fogging_score is None and hedging_score is None and promotional_score is None:
        return None
    return ToneAssessment(fogging_score=fogging_score, hedging_score=hedging_score, promotional_score=promotional_score, confidence=confidence_assessment.final_confidence, rationale="Intermediate arbiter tone derived from weighted worker tone outputs.")


# ---------------------------------------------------------------------------
# Base arbiter ABC & CrossDocumentArbiter
# ---------------------------------------------------------------------------

class BaseArbiter(ABC):
    arbiter_name: ClassVar[str] = "base_arbiter"
    arbiter_kind: ClassVar[ArbiterKind]
    input_model: ClassVar[type[ArbiterInput]] = ArbiterInput
    output_model: ClassVar[type[ArbiterOutput]] = ArbiterOutput

    @abstractmethod
    def arbitrate(self, arbiter_input: ArbiterInput) -> ArbiterOutput:
        raise NotImplementedError


class CrossDocumentArbiter(BaseArbiter):
    arbiter_name = "cross_document_arbiter"
    arbiter_kind = ArbiterKind.CROSS_DOCUMENT

    def arbitrate(self, arbiter_input: ArbiterInput) -> ArbiterOutput:
        selected_worker_outputs, selection_warnings = select_best_worker_output_per_type(arbiter_input.worker_outputs)
        normalized_outputs = [normalize_worker_output_for_arbitration(worker_output) for worker_output in selected_worker_outputs]
        covered_document_types = [output.document_type for output in normalized_outputs if output.status in USABLE_WORKER_STATUSES]
        missing_document_types = [document_type for document_type in EXPECTED_DOCUMENT_TYPES if document_type not in covered_document_types]
        comparison = compare_worker_outputs(normalized_outputs)
        story_mismatch_findings = detect_story_vs_substance_mismatch(normalized_outputs)
        positive_signal_groups = group_positive_signals(comparison["judgments"])
        negative_signal_groups = group_negative_signals(comparison["judgments"])
        unresolved_uncertainties = group_uncertainties(comparison["judgments"], normalized_outputs, missing_document_types)
        fogging_findings = group_fogging_concerns(normalized_outputs, story_mismatch_findings)
        confidence_assessment = assess_cross_document_confidence(normalized_outputs, comparison["conflicts"], missing_document_types)
        warnings = selection_warnings + build_arbiter_warnings(normalized_outputs, comparison["conflicts"], missing_document_types, confidence_assessment)
        issues = [issue for normalized_output in normalized_outputs for issue in normalized_output.issues]
        missing_coverage_notes = build_missing_coverage_notes(normalized_outputs, missing_document_types)
        evidence_references = deduplicate_arbiter_evidence(
            [reference for judgment in comparison["judgments"] for reference in judgment.evidence_references]
            + [reference for group in positive_signal_groups for reference in group.evidence_references]
            + [reference for group in negative_signal_groups for reference in group.evidence_references]
            + [reference for conflict in comparison["conflicts"] for reference in conflict.evidence_references]
            + [reference for finding in unresolved_uncertainties for reference in finding.evidence_references]
            + [reference for finding in fogging_findings for reference in finding.evidence_references]
        )
        legacy_evidence = deduplicate_evidence_snippets([evidence_snippet for worker_output in selected_worker_outputs for evidence_snippet in worker_output.evidence])
        status = ProcessingStatus.SUCCESS
        if missing_document_types or comparison["conflicts"] or unresolved_uncertainties:
            status = ProcessingStatus.PARTIAL
        if not normalized_outputs:
            status = ProcessingStatus.NO_DOCUMENT
        aligned_signals = positive_signal_groups + negative_signal_groups
        return ArbiterOutput(
            arbiter_id=f"{arbiter_input.run_id}_cross_document",
            arbiter_name=self.arbiter_name,
            arbiter_kind=self.arbiter_kind,
            status=status,
            summary=f"Intermediate arbiter output covering {len(covered_document_types)} of {len(EXPECTED_DOCUMENT_TYPES)} expected disclosure types.",
            sentiment=aggregate_arbiter_sentiment(normalized_outputs, comparison["conflicts"], confidence_assessment),
            tone=aggregate_arbiter_tone(normalized_outputs, confidence_assessment),
            covered_document_types=covered_document_types,
            missing_document_types=missing_document_types,
            cross_document_judgments=comparison["judgments"],
            positive_signal_groups=positive_signal_groups,
            negative_signal_groups=negative_signal_groups,
            aligned_signals=aligned_signals,
            conflicting_signals=comparison["conflicts"],
            unresolved_uncertainties=unresolved_uncertainties,
            fogging_or_story_substance_flags=fogging_findings,
            confidence_assessment=confidence_assessment,
            missing_coverage_notes=missing_coverage_notes,
            evidence_references=evidence_references,
            warnings=warnings,
            issues=issues,
            reasoning_notes=[
                f"received_worker_outputs={len(arbiter_input.worker_outputs)}",
                f"normalized_outputs={len(normalized_outputs)}",
                f"judgments={len(comparison['judgments'])}",
                f"conflicts={len(comparison['conflicts'])}",
                f"missing_document_types={len(missing_document_types)}",
            ],
            consensus_points=[group.description for group in aligned_signals[:4]],
            conflicts=[conflict.description for conflict in comparison["conflicts"]],
            evidence=legacy_evidence,
            confidence=confidence_assessment.final_confidence,
        )


ARBITER_REGISTRY: dict[ArbiterKind, Type[BaseArbiter]] = {
    ArbiterKind.CROSS_DOCUMENT: CrossDocumentArbiter,
}
