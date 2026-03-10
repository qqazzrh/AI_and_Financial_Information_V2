"""All Pydantic models for the biotech disclosure pipeline.

Uses FINAL versions of models from cells 96-97 of the notebook where models
were redefined with additional fields. All arbiter sub-models are also defined here.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import date, datetime, time, timezone
from typing import Any, ClassVar, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pipeline.enums import (
    ArbiterDecisionType,
    ArbiterKind,
    ArbiterSignalCategory,
    AnalysisDimension,
    AnalysisIssueSeverity,
    CrossDocumentTheme,
    DocumentType,
    EmbeddingProvider,
    EmbeddingStatus,
    EvidenceInterpretation,
    EvidenceType,
    GraphEdgeType,
    GraphNodeType,
    NormalizedSignalDirection,
    ProcessingNoteSeverity,
    ProcessingStatus,
    SectionKind,
    SentimentLabel,
    SourceFamily,
)
from pipeline.config import (
    DEFAULT_FRESHNESS_BONUS_MAX,
    DEFAULT_FRESHNESS_RANK_DECAY,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MIN_USABLE_TEXT_CHARS,
    DEFAULT_MINIMUM_RELEVANCE_SCORE,
    DEFAULT_SOURCE_FAMILY_ORDER,
    SENTIMENT_SCORE_MAX,
    SENTIMENT_SCORE_MIN,
    TONE_SCORE_MAX,
    TONE_SCORE_MIN,
    now_utc,
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_optional_range(
    value: float | None, minimum: float, maximum: float, field_name: str,
) -> float | None:
    if value is None:
        return value
    if not minimum <= value <= maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}.")
    return value


# ---------------------------------------------------------------------------
# Base contract model
# ---------------------------------------------------------------------------

class ContractModel(BaseModel):
    """Base contract model with strict validation for scaffold development."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# Core pipeline models
# ---------------------------------------------------------------------------

class PipelineConfig(ContractModel):
    """Typed notebook configuration shared across retrieval and later analysis nodes."""
    project_name: str
    notebook_version: str
    project_root: "Any"
    data_root: "Any"
    raw_document_dir: "Any"
    processed_document_dir: "Any"
    cache_dir: "Any"
    disclosure_types: list[DocumentType]
    sentiment_score_min: float
    sentiment_score_max: float
    tone_score_min: float
    tone_score_max: float
    default_max_candidates: int
    min_usable_text_chars: int
    minimum_relevance_score: float
    freshness_bonus_max: float
    freshness_rank_decay: float
    default_source_preferences: list[SourceFamily]
    enable_retrieval: bool = True
    enable_embeddings: bool = False
    enable_graph_context: bool = False
    enable_analysis: bool = True
    enable_langchain: bool = False
    enable_langgraph: bool = False
    worker_model_name: str = "configurable_placeholder"
    arbiter_model_name: str = "configurable_placeholder"
    embedding_model_name: str = "configurable_placeholder"
    source_adapter_placeholders: dict[str, str] = Field(default_factory=dict)


class DocumentMetadata(ContractModel):
    """Normalized metadata for a single selected document candidate."""
    document_id: str
    ticker: str
    document_type: DocumentType
    company_name: str | None = None
    title: str | None = None
    source_name: str | None = None
    source_identifier: str | None = None
    source_family: SourceFamily = SourceFamily.UNKNOWN
    source_url: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    event_date: date | None = None
    version_label: str | None = None
    file_type: str | None = None
    language: str = DEFAULT_LANGUAGE
    content_hash: str | None = None
    retrieved_at: datetime | None = None
    is_structured_source: bool = False
    is_mock_data: bool = False
    external_ids: dict[str, str] = Field(default_factory=dict)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class PipelineError(ContractModel):
    """Structured pipeline issue record for warnings and failures."""
    error_code: str
    message: str
    stage: str
    document_type: DocumentType | None = None
    candidate_id: str | None = None
    adapter_name: str | None = None
    recoverable: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class ProvenanceRecord(ContractModel):
    """Trace record describing how a retrieval object moved through the pipeline."""
    stage: str
    adapter_name: str
    candidate_id: str | None = None
    document_type: DocumentType | None = None
    source_name: str | None = None
    source_identifier: str | None = None
    source_url: str | None = None
    retrieved_at: datetime
    note: str | None = None
    ranking_notes: list[str] = Field(default_factory=list)
    validation_notes: list[str] = Field(default_factory=list)
    selection_reason: str | None = None
    is_mock_data: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SentimentAssessment(ContractModel):
    """Standard sentiment assessment shape shared across outputs."""
    label: SentimentLabel = SentimentLabel.INSUFFICIENT_EVIDENCE
    score: float | None = None
    confidence: float | None = None
    rationale: str | None = None

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, SENTIMENT_SCORE_MIN, SENTIMENT_SCORE_MAX, "score")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, 0.0, 1.0, "confidence")


class ToneAssessment(ContractModel):
    """Fogging, hedging, and promotional tone scores for one analysis unit."""
    fogging_score: float | None = None
    hedging_score: float | None = None
    promotional_score: float | None = None
    confidence: float | None = None
    rationale: str | None = None

    @field_validator("fogging_score", "hedging_score", "promotional_score")
    @classmethod
    def validate_tone_score(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, TONE_SCORE_MIN, TONE_SCORE_MAX, "tone score")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, 0.0, 1.0, "confidence")


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------

class SourcePreferencePolicy(ContractModel):
    """Provider-agnostic retrieval source preferences."""
    preferred_source_families: list[SourceFamily] = Field(default_factory=lambda: DEFAULT_SOURCE_FAMILY_ORDER.copy())
    prefer_structured_source: bool = True
    allow_secondary_sources: bool = True
    allow_unknown_sources: bool = True
    allow_mock_data: bool = True


class RetrievalRequest(ContractModel):
    """Generic retrieval request used by all disclosure adapters."""
    request_id: str
    ticker: str
    company_name: str | None = None
    company_aliases: list[str] = Field(default_factory=list)
    document_type: DocumentType
    start_date: date | None = None
    end_date: date | None = None
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    minimum_text_chars: int = DEFAULT_MIN_USABLE_TEXT_CHARS
    minimum_relevance_score: float = DEFAULT_MINIMUM_RELEVANCE_SCORE
    freshness_bonus_max: float = DEFAULT_FRESHNESS_BONUS_MAX
    freshness_rank_decay: float = DEFAULT_FRESHNESS_RANK_DECAY
    source_preferences: SourcePreferencePolicy = Field(default_factory=SourcePreferencePolicy)
    retrieval_notes: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    requested_at: datetime = Field(default_factory=now_utc)
    is_mock_request: bool = False

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("max_candidates")
    @classmethod
    def validate_max_candidates(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_candidates must be at least 1.")
        return value

    @model_validator(mode="after")
    def validate_date_bounds(self) -> "RetrievalRequest":
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date.")
        return self


class RawRetrievalCandidateMetadata(ContractModel):
    """Generic raw metadata returned by an adapter before normalization."""
    raw_candidate_id: str
    adapter_name: str
    document_type: DocumentType
    ticker: str | None = None
    company_name: str | None = None
    source_name: str | None = None
    source_identifier: str | None = None
    source_url: str | None = None
    title: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    event_date: date | None = None
    source_family: SourceFamily = SourceFamily.UNKNOWN
    is_structured_source: bool = False
    is_mock_data: bool = False
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class FetchedDocumentContent(ContractModel):
    """Generic fetched content paired with one raw retrieval candidate."""
    raw_candidate_id: str
    document_text: str | None = None
    content_type: str | None = None
    fetched_at: datetime = Field(default_factory=now_utc)
    fetch_status: ProcessingStatus = ProcessingStatus.SUCCESS
    fetch_notes: list[str] = Field(default_factory=list)
    is_mock_data: bool = False
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class DocumentValidationResult(ContractModel):
    """Validation outcome for a normalized retrieval candidate."""
    candidate_id: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    is_valid: bool = False
    is_partial: bool = False
    has_usable_text: bool = False
    has_minimum_metadata: bool = False
    type_matches_request: bool = False
    identity_matches_request: bool = False
    validation_notes: list[str] = Field(default_factory=list)
    issues: list[PipelineError] = Field(default_factory=list)


class CandidateEvaluation(ContractModel):
    """Transparent heuristic scores used for retrieval selection."""
    candidate_id: str
    is_relevant: bool = False
    relevance_score: float = 0.0
    freshness_score: float = 0.0
    total_score: float = 0.0
    type_compatibility_score: float = 0.0
    identity_alignment_score: float = 0.0
    title_alignment_score: float = 0.0
    structured_source_bonus: float = 0.0
    source_preference_bonus: float = 0.0
    duplicate_key: str | None = None
    effective_timestamp: datetime | None = None
    evaluation_notes: list[str] = Field(default_factory=list)


class RetrievalCandidate(ContractModel):
    """Normalized internal retrieval candidate used by selection logic."""
    candidate_id: str
    adapter_name: str
    document_type: DocumentType
    ticker: str | None = None
    company_name: str | None = None
    source_name: str | None = None
    source_identifier: str | None = None
    source_url: str | None = None
    source_family: SourceFamily = SourceFamily.UNKNOWN
    title: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    event_date: date | None = None
    retrieved_at: datetime
    document_text: str | None = None
    is_structured_source: bool = False
    is_mock_data: bool = False
    relevance_notes: list[str] = Field(default_factory=list)
    freshness_notes: list[str] = Field(default_factory=list)
    validation_notes: list[str] = Field(default_factory=list)
    selection_notes: list[str] = Field(default_factory=list)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    validation: DocumentValidationResult | None = None
    evaluation: CandidateEvaluation | None = None
    duplicate_key: str | None = None
    superseded_by_candidate_id: str | None = None

    def effective_timestamp(self) -> datetime | None:
        if self.updated_at is not None:
            return self.updated_at
        if self.published_at is not None:
            return self.published_at
        if self.event_date is not None:
            return datetime.combine(self.event_date, time.min, tzinfo=timezone.utc)
        return None


class RetrievalSelectionDecision(ContractModel):
    """Selection audit trail for most-recent relevant candidate choice."""
    selected_candidate_id: str | None = None
    selected_reason: str | None = None
    rejected_candidate_ids: list[str] = Field(default_factory=list)
    ranking_notes: list[str] = Field(default_factory=list)
    tie_break_notes: list[str] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)


class RetrievalResult(ContractModel):
    """Retrieval output for one request and one disclosure type."""
    request: RetrievalRequest
    adapter_name: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    selected_candidate: RetrievalCandidate | None = None
    baseline_candidate: RetrievalCandidate | None = None
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    selection_decision: RetrievalSelectionDecision = Field(default_factory=RetrievalSelectionDecision)
    issues: list[PipelineError] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    is_mock_result: bool = False
    retrieved_at: datetime = Field(default_factory=now_utc)


# ---------------------------------------------------------------------------
# Evidence and worker I/O models (FINAL from cell 96)
# ---------------------------------------------------------------------------

class EvidenceSnippet(ContractModel):
    """Structured evidence reference that downstream nodes can cite."""
    evidence_id: str | None = None
    document_id: str
    source_url: str | None = None
    source_chunk_id: str | None = None
    source_section_id: str | None = None
    section_title: str | None = None
    evidence_type: EvidenceType | None = None
    supported_dimensions: list[AnalysisDimension] = Field(default_factory=list)
    interpretation: EvidenceInterpretation | None = None
    snippet_text: str | None = None
    rationale: str
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerInput(ContractModel):
    """Input contract for a disclosure-specific worker node."""
    run_id: str
    ticker: str
    document_type: DocumentType
    retrieval_result: RetrievalResult
    config: PipelineConfig
    graph_context: dict[str, Any] = Field(default_factory=dict)


class WorkerOutput(ContractModel):
    """Final reconciled worker output contract used by arbiter and master stages."""
    worker_name: str
    document_type: DocumentType
    status: ProcessingStatus = ProcessingStatus.PENDING
    summary: str | None = None
    sentiment: SentimentAssessment | None = None
    tone: ToneAssessment | None = None
    key_points: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSnippet] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    issues: list[PipelineError] = Field(default_factory=list)
    confidence: float | None = None
    warnings: list[PipelineError] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    document_metadata: DocumentMetadata | None = None
    reasoning_notes: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, 0.0, 1.0, "confidence")


# ---------------------------------------------------------------------------
# Arbiter sub-models
# ---------------------------------------------------------------------------

class ArbiterIssue(ContractModel):
    issue_code: str
    message: str
    severity: AnalysisIssueSeverity = AnalysisIssueSeverity.WARNING
    document_types: list[DocumentType] = Field(default_factory=list)
    worker_names: list[str] = Field(default_factory=list)
    recoverable: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArbiterWarning(ArbiterIssue):
    severity: AnalysisIssueSeverity = AnalysisIssueSeverity.WARNING


class ArbiterEvidenceReference(ContractModel):
    worker_name: str
    document_type: DocumentType
    evidence_id: str | None = None
    document_id: str
    source_url: str | None = None
    source_chunk_id: str | None = None
    source_section_id: str | None = None
    section_title: str | None = None
    interpretation: EvidenceInterpretation | None = None
    snippet_text: str | None = None
    rationale: str | None = None


class NormalizedWorkerOutput(ContractModel):
    worker_name: str
    document_type: DocumentType
    status: ProcessingStatus
    summary: str | None = None
    sentiment_label: SentimentLabel = SentimentLabel.INSUFFICIENT_EVIDENCE
    sentiment_score: float | None = None
    worker_confidence_raw: float | None = None
    normalized_confidence: float = 0.0
    evidence_density: float = 0.0
    key_points: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    key_point_count: int = 0
    caveat_count: int = 0
    issue_count: int = 0
    direction: NormalizedSignalDirection = NormalizedSignalDirection.UNCERTAIN
    fogging_score: float | None = None
    hedging_score: float | None = None
    promotional_score: float | None = None
    is_structured_document: bool = False
    is_soft_document: bool = False
    warnings: list[ArbiterWarning] = Field(default_factory=list)
    issues: list[ArbiterIssue] = Field(default_factory=list)
    evidence_references: list[ArbiterEvidenceReference] = Field(default_factory=list)
    normalization_notes: list[str] = Field(default_factory=list)

    @field_validator("normalized_confidence", "evidence_density")
    @classmethod
    def validate_unit_interval(cls, value: float) -> float:
        return _validate_optional_range(value, 0.0, 1.0, "arbiter normalized value")


class ThematicWorkerSignal(ContractModel):
    theme: CrossDocumentTheme
    worker_name: str
    document_type: DocumentType
    direction: NormalizedSignalDirection
    relevance_weight: float
    adjusted_confidence: float
    evidence_density: float
    sentiment_score: float | None = None
    fogging_score: float | None = None
    hedging_score: float | None = None
    promotional_score: float | None = None
    is_structured_document: bool = False
    is_soft_document: bool = False
    summary: str | None = None
    evidence_references: list[ArbiterEvidenceReference] = Field(default_factory=list)

    @field_validator("relevance_weight", "adjusted_confidence", "evidence_density")
    @classmethod
    def validate_unit_interval(cls, value: float) -> float:
        return _validate_optional_range(value, 0.0, 1.0, "arbiter thematic value")


class ArbiterFinding(ContractModel):
    finding_id: str
    category: ArbiterSignalCategory
    decision_type: ArbiterDecisionType
    theme: CrossDocumentTheme | None = None
    title: str
    summary: str
    supporting_document_types: list[DocumentType] = Field(default_factory=list)
    contradicting_document_types: list[DocumentType] = Field(default_factory=list)
    worker_names: list[str] = Field(default_factory=list)
    confidence: float | None = None
    evidence_references: list[ArbiterEvidenceReference] = Field(default_factory=list)
    reasoning_notes: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, 0.0, 1.0, "finding confidence")


class ArbiterSignalGroup(ContractModel):
    group_id: str
    category: ArbiterSignalCategory
    theme: CrossDocumentTheme | None = None
    title: str
    description: str
    document_types: list[DocumentType] = Field(default_factory=list)
    worker_names: list[str] = Field(default_factory=list)
    findings: list[ArbiterFinding] = Field(default_factory=list)
    evidence_references: list[ArbiterEvidenceReference] = Field(default_factory=list)
    confidence: float | None = None
    reasoning_notes: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, 0.0, 1.0, "signal group confidence")


class ArbiterConflict(ContractModel):
    conflict_id: str
    theme: CrossDocumentTheme
    title: str
    description: str
    positive_document_types: list[DocumentType] = Field(default_factory=list)
    negative_document_types: list[DocumentType] = Field(default_factory=list)
    worker_names: list[str] = Field(default_factory=list)
    high_confidence_conflict: bool = False
    evidence_references: list[ArbiterEvidenceReference] = Field(default_factory=list)
    reasoning_notes: list[str] = Field(default_factory=list)


class CrossDocumentJudgment(ContractModel):
    judgment_id: str
    theme: CrossDocumentTheme
    decision_type: ArbiterDecisionType
    direction: NormalizedSignalDirection
    summary: str
    supporting_document_types: list[DocumentType] = Field(default_factory=list)
    opposing_document_types: list[DocumentType] = Field(default_factory=list)
    worker_names: list[str] = Field(default_factory=list)
    confidence: float | None = None
    evidence_references: list[ArbiterEvidenceReference] = Field(default_factory=list)
    reasoning_notes: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, 0.0, 1.0, "judgment confidence")


class ArbiterConfidenceAdjustment(ContractModel):
    factor_name: str
    adjustment: float
    rationale: str


class ArbiterConfidenceAssessment(ContractModel):
    base_confidence: float
    final_confidence: float
    worker_confidence_average: float | None = None
    evidence_density_average: float | None = None
    structured_support_ratio: float | None = None
    missing_document_types: list[DocumentType] = Field(default_factory=list)
    high_confidence_conflict_count: int = 0
    adjustments: list[ArbiterConfidenceAdjustment] = Field(default_factory=list)
    reasoning_notes: list[str] = Field(default_factory=list)

    @field_validator(
        "base_confidence", "final_confidence",
        "worker_confidence_average", "evidence_density_average",
        "structured_support_ratio",
    )
    @classmethod
    def validate_unit_interval(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, 0.0, 1.0, "arbiter confidence value")


class ArbiterOutput(ContractModel):
    """Cross-document arbiter output (FINAL version with all fields)."""
    arbiter_id: str
    arbiter_name: str
    arbiter_kind: ArbiterKind = ArbiterKind.CROSS_DOCUMENT
    status: ProcessingStatus = ProcessingStatus.PENDING
    summary: str | None = None
    sentiment: SentimentAssessment | None = None
    tone: ToneAssessment | None = None
    covered_document_types: list[DocumentType] = Field(default_factory=list)
    missing_document_types: list[DocumentType] = Field(default_factory=list)
    cross_document_judgments: list[CrossDocumentJudgment] = Field(default_factory=list)
    positive_signal_groups: list[ArbiterSignalGroup] = Field(default_factory=list)
    negative_signal_groups: list[ArbiterSignalGroup] = Field(default_factory=list)
    aligned_signals: list[ArbiterSignalGroup] = Field(default_factory=list)
    conflicting_signals: list[ArbiterConflict] = Field(default_factory=list)
    unresolved_uncertainties: list[ArbiterFinding] = Field(default_factory=list)
    fogging_or_story_substance_flags: list[ArbiterFinding] = Field(default_factory=list)
    confidence_assessment: ArbiterConfidenceAssessment | None = None
    missing_coverage_notes: list[str] = Field(default_factory=list)
    evidence_references: list[ArbiterEvidenceReference] = Field(default_factory=list)
    warnings: list[ArbiterWarning] = Field(default_factory=list)
    issues: list[ArbiterIssue] = Field(default_factory=list)
    reasoning_notes: list[str] = Field(default_factory=list)
    consensus_points: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSnippet] = Field(default_factory=list)
    confidence: float | None = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, 0.0, 1.0, "arbiter output confidence")


class ArbiterInput(ContractModel):
    """Final reconciled arbiter input using the latest WorkerOutput contract."""
    run_id: str
    ticker: str
    worker_outputs: list[WorkerOutput]
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    config: PipelineConfig


# ---------------------------------------------------------------------------
# Master models (FINAL from cell 96)
# ---------------------------------------------------------------------------

class DisclosureCardPayload(ContractModel):
    """Per-disclosure review entry emitted only by the master node."""
    document_type: DocumentType
    worker_name: str
    status: ProcessingStatus
    title: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    summary: str | None = None
    sentiment_label: SentimentLabel | None = None
    sentiment_score: float | None = None
    confidence: float | None = None
    fogging_score: float | None = None
    hedging_score: float | None = None
    promotional_score: float | None = None
    key_points: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSnippet] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)

    @field_validator("sentiment_score")
    @classmethod
    def validate_sentiment_score(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, SENTIMENT_SCORE_MIN, SENTIMENT_SCORE_MAX, "sentiment_score")

    @field_validator("confidence", "fogging_score", "hedging_score", "promotional_score")
    @classmethod
    def validate_optional_unit_interval(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, 0.0, 1.0, "disclosure payload score")


class FinalUIPayload(ContractModel):
    """Final UI-facing payload emitted exclusively by the master node."""
    ticker: str
    generated_at: datetime = Field(default_factory=now_utc)
    status: ProcessingStatus = ProcessingStatus.PENDING
    overall_summary: str | None = None
    overall_sentiment_label: SentimentLabel | None = None
    overall_sentiment_score: float | None = None
    overall_confidence: float | None = None
    overall_fogging_score: float | None = None
    overall_hedging_score: float | None = None
    overall_promotional_score: float | None = None
    key_positive_signals: list[str] = Field(default_factory=list)
    key_negative_signals: list[str] = Field(default_factory=list)
    key_uncertainties: list[str] = Field(default_factory=list)
    fogging_or_story_substance_flags: list[str] = Field(default_factory=list)
    disclosures: list[DisclosureCardPayload] = Field(default_factory=list)
    missing_document_types: list[DocumentType] = Field(default_factory=list)
    system_warnings: list[str] = Field(default_factory=list)
    issues: list[PipelineError] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)

    @field_validator("overall_sentiment_score")
    @classmethod
    def validate_overall_sentiment_score(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, SENTIMENT_SCORE_MIN, SENTIMENT_SCORE_MAX, "overall_sentiment_score")

    @field_validator("overall_confidence", "overall_fogging_score", "overall_hedging_score", "overall_promotional_score")
    @classmethod
    def validate_optional_unit_interval(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, 0.0, 1.0, "overall payload score")


class MasterInput(ContractModel):
    """Input contract for the single master node."""
    run_id: str
    ticker: str
    worker_outputs: list[WorkerOutput]
    arbiter_outputs: list[ArbiterOutput]
    retrieval_results: list[RetrievalResult]
    config: PipelineConfig


class MasterOutput(ContractModel):
    """Structured master-node output before final UI flattening."""
    ticker: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    master_summary: str | None = None
    master_sentiment: SentimentAssessment | None = None
    master_tone: ToneAssessment | None = None
    disclosures: list[DisclosureCardPayload] = Field(default_factory=list)
    missing_document_types: list[DocumentType] = Field(default_factory=list)
    key_positive_signals: list[str] = Field(default_factory=list)
    key_negative_signals: list[str] = Field(default_factory=list)
    key_uncertainties: list[str] = Field(default_factory=list)
    fogging_or_story_substance_flags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    issues: list[PipelineError] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    ready_for_ui: bool = False
    generated_at: datetime = Field(default_factory=now_utc)


class MasterNode(ABC):
    """Single master node that emits the only UI-facing payload."""
    output_model: ClassVar[type[FinalUIPayload]] = FinalUIPayload

    @abstractmethod
    def build_payload(self, master_input: MasterInput) -> FinalUIPayload:
        raise NotImplementedError


class PipelineState(ContractModel):
    """Shared state container aligned to the final worker and master contracts."""
    run_id: str
    ticker: str
    config: PipelineConfig
    retrieval_results: dict[DocumentType, RetrievalResult] = Field(default_factory=dict)
    worker_outputs: dict[DocumentType, WorkerOutput] = Field(default_factory=dict)
    arbiter_outputs: list[ArbiterOutput] = Field(default_factory=list)
    master_output: MasterOutput | None = None
    graph_context: dict[str, Any] = Field(default_factory=dict)
    shared_cache_refs: dict[str, str] = Field(default_factory=dict)
    errors: list[PipelineError] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)


# ---------------------------------------------------------------------------
# Base worker
# ---------------------------------------------------------------------------

class BaseWorker(ABC):
    """Base interface for one disclosure-specific worker node."""
    worker_name: ClassVar[str] = "base_worker"
    document_type: ClassVar[DocumentType]
    input_model: ClassVar[type[WorkerInput]] = WorkerInput
    output_model: ClassVar[type[WorkerOutput]] = WorkerOutput

    def analyze(self, worker_input: WorkerInput) -> WorkerOutput:
        raise NotImplementedError("Worker analysis is intentionally stubbed in this notebook step.")


# ---------------------------------------------------------------------------
# Processing models
# ---------------------------------------------------------------------------

class ProcessingConfig(ContractModel):
    """Configuration for deterministic text cleanup and section parsing."""
    max_header_chars: int = 120
    max_header_words: int = 14
    strip_page_markers: bool = True
    normalize_tables: bool = True
    preserve_headers: bool = True
    collapse_blank_lines: bool = True


class ProcessingNote(ContractModel):
    """A structured note emitted during deterministic document processing."""
    stage: str
    message: str
    severity: ProcessingNoteSeverity
    document_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def code(self) -> str:
        return self.stage


class SelectedDocument(ContractModel):
    """A retrieval candidate that has been selected for downstream processing."""
    document_id: str
    document_type: DocumentType
    ticker: str
    title: str | None = None
    raw_text: str
    source_name: str | None = None
    source_url: str | None = None
    source_identifier: str | None = None
    metadata: DocumentMetadata | None = None
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    is_mock_data: bool = False


class DocumentSection(ContractModel):
    """A structural section detected in a cleaned document."""
    section_id: str
    document_id: str
    section_index: int
    title: str
    section_kind: SectionKind
    level: int = 1
    reference_label: str | None = None
    line_start: int = 0
    line_end: int = 0
    raw_text: str = ""
    cleaned_text: str = ""
    char_count: int = 0
    word_count: int = 0
    header_detected: bool = False
    parent_section_id: str | None = None


class DocumentChunk(ContractModel):
    """A text chunk produced by the chunking strategy."""
    chunk_id: str
    document_id: str
    document_type: DocumentType
    chunk_index: int = 0
    order_in_section: int = 0
    parent_section_id: str | None = None
    parent_section_title: str | None = None
    section_kind: SectionKind = SectionKind.UNKNOWN
    text: str = ""
    char_count: int = 0
    word_count: int = 0
    overlap_prefix: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    context_before: str | None = None
    context_after: str | None = None
    local_context_summary: str | None = None
    notes: list[ProcessingNote] = Field(default_factory=list)


class ProcessedDocument(ContractModel):
    """Aggregates cleaned text, sections, chunks, and processing notes for one document."""
    document: SelectedDocument
    cleaned_text: str = ""
    sections: list[DocumentSection] = Field(default_factory=list)
    chunks: list[DocumentChunk] = Field(default_factory=list)
    processing_notes: list[ProcessingNote] = Field(default_factory=list)
    raw_char_count: int = 0
    cleaned_char_count: int = 0
    cleaned_word_count: int = 0
    section_count: int = 0
    chunk_count: int = 0
    used_fallback_chunking: bool = False


class ChunkingConfig(ContractModel):
    """Configuration for the chunking strategy."""
    target_chunk_chars: int
    max_chunk_chars: int
    overlap_chars: int
    min_chunk_chars: int
    context_window_chars: int
    include_section_titles: bool = True
    fallback_chunk_chars: int


class EmbeddingConfig(ContractModel):
    """Configuration for embedding generation."""
    provider: EmbeddingProvider = EmbeddingProvider.VOYAGE
    enabled: bool = True
    model_name: str = "configurable_placeholder"
    base_url: str = "https://api.voyageai.com"
    api_path: str = "/v1/embeddings"
    batch_size: int = 8
    request_timeout_seconds: float = 15.0
    normalize_vectors: bool = True


class GraphRetrievalConfig(ContractModel):
    """Configuration for graph-aware in-document chunk retrieval."""
    top_k: int = 3
    candidate_pool_size: int = 6
    neighbor_hops: int = 1
    context_expansion_hops: int = 1
    min_similarity_threshold: float = 0.0
    max_graph_bonus: float = 0.18
    section_title_weight: float = 0.08
    neighbor_similarity_weight: float = 0.06
    cross_reference_bonus: float = 0.03


# ---------------------------------------------------------------------------
# Embedding / graph models
# ---------------------------------------------------------------------------

class ChunkEmbeddingRecord(ContractModel):
    """Embedding record attached to one document chunk."""
    chunk_id: str
    document_id: str
    provider_name: str
    model_name: str
    status: EmbeddingStatus
    embedding: list[float] | None = None
    vector_dimension: int = 0
    generated_at: datetime = Field(default_factory=now_utc)
    is_mock_embedding: bool = False
    notes: list[ProcessingNote] = Field(default_factory=list)


class EmbeddingBatchResult(ContractModel):
    """Batch embedding result returned by an embedding client."""
    provider_name: str
    model_name: str
    status: EmbeddingStatus
    vectors: list[list[float]] = Field(default_factory=list)
    is_mock_embedding: bool = False
    notes: list[ProcessingNote] = Field(default_factory=list)


class GraphNode(ContractModel):
    """Node in the lightweight single-document graph."""
    node_id: str
    document_id: str
    node_type: GraphNodeType
    label: str
    ref_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(ContractModel):
    """Edge in the lightweight single-document graph."""
    edge_id: str
    document_id: str
    source_node_id: str
    target_node_id: str
    edge_type: GraphEdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphDocumentIndex(ContractModel):
    """Inspectable graph index for one processed document."""
    document_id: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    section_to_chunk_ids: dict[str, list[str]] = Field(default_factory=dict)
    chunk_neighbors: dict[str, list[str]] = Field(default_factory=dict)
    chunk_to_section_id: dict[str, str] = Field(default_factory=dict)
    cross_reference_targets: dict[str, list[str]] = Field(default_factory=dict)
    notes: list[ProcessingNote] = Field(default_factory=list)


class ChunkRetrievalHit(ContractModel):
    """Ranked chunk match returned by in-document semantic retrieval."""
    rank: int
    chunk_id: str
    document_id: str
    section_id: str | None = None
    section_title: str | None = None
    similarity_score: float
    graph_bonus: float = 0.0
    adjusted_score: float
    text_excerpt: str
    expanded_context_chunk_ids: list[str] = Field(default_factory=list)
    expanded_context_preview: str | None = None
    notes: list[ProcessingNote] = Field(default_factory=list)


class ChunkRetrievalResult(ContractModel):
    """Result object for graph-aware semantic retrieval within one processed document."""
    document_id: str
    query_text: str
    embedding_provider: str
    embedding_model_name: str
    query_embedding_status: EmbeddingStatus
    used_graph_context: bool = True
    hits: list[ChunkRetrievalHit] = Field(default_factory=list)
    notes: list[ProcessingNote] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Analysis models
# ---------------------------------------------------------------------------

class RubricAnchor(ContractModel):
    """Human-readable anchor for a score within one dimension."""
    score: float
    description: str


class RubricDimensionDefinition(ContractModel):
    """Explicit scoring definition for one analysis dimension."""
    dimension: AnalysisDimension
    minimum_score: float
    maximum_score: float
    objective: str
    evidence_expectation: str
    scoring_guidance: list[str] = Field(default_factory=list)
    anchors: list[RubricAnchor] = Field(default_factory=list)


# Keep legacy alias
AnalysisDimensionDefinition = RubricDimensionDefinition


class AnalysisRubric(ContractModel):
    """Inspectable shared rubric applied before worker specialization."""
    rubric_id: str
    version: str
    summary: str = ""
    core_principles: list[str] = Field(default_factory=list)
    sentiment_labels: list[SentimentLabel] = Field(default_factory=list)
    dimension_definitions: list[RubricDimensionDefinition] = Field(default_factory=list)


class AnalysisScore(ContractModel):
    """Normalized per-dimension score produced by the shared analysis path."""
    dimension: AnalysisDimension
    score: float | None = None
    label: str | None = None
    confidence_score: float | None = None
    rationale: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    rubric_notes: list[str] = Field(default_factory=list)


class AnalysisWarning(ContractModel):
    """Structured non-fatal warning produced during analysis normalization."""
    issue_code: str
    message: str
    severity: AnalysisIssueSeverity = AnalysisIssueSeverity.WARNING
    dimension: AnalysisDimension | None = None
    source_chunk_id: str | None = None
    field_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisIssue(ContractModel):
    """Structured error or major issue produced during analysis normalization."""
    issue_code: str
    message: str
    severity: AnalysisIssueSeverity = AnalysisIssueSeverity.ERROR
    dimension: AnalysisDimension | None = None
    source_chunk_id: str | None = None
    field_name: str | None = None
    recoverable: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisFinding(ContractModel):
    """One normalized worker finding backed by cited evidence."""
    finding_id: str
    dimension: AnalysisDimension
    summary: str
    interpretation: EvidenceInterpretation | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    score: float | None = None
    rationale: str | None = None
    is_material: bool = False


class SectionContextSummary(ContractModel):
    """Compact section metadata supplied to the generic worker-analysis input."""
    section_id: str
    title: str
    level: int
    section_kind: SectionKind
    char_count: int
    word_count: int


class ChunkEvidenceBundle(ContractModel):
    """Bundle of chunk text plus local context used as candidate evidence for analysis."""
    bundle_id: str
    document_id: str
    chunk_id: str
    section_id: str | None = None
    section_title: str | None = None
    retrieval_rank: int | None = None
    adjusted_score: float | None = None
    similarity_score: float | None = None
    graph_bonus: float | None = None
    evidence_type: EvidenceType = EvidenceType.CONTEXTUAL_SUMMARY
    primary_text: str
    expanded_context_text: str | None = None
    local_context_summary: str | None = None
    notes: list[str] = Field(default_factory=list)


class WorkerAnalysisInput(ContractModel):
    """Shared analysis packet assembled before any disclosure-specific prompting exists."""
    run_id: str
    ticker: str
    worker_name: str
    document_type: DocumentType
    document_metadata: DocumentMetadata
    document_title: str | None = None
    document_text: str
    document_text_excerpt: str
    section_context: list[SectionContextSummary] = Field(default_factory=list)
    top_chunk_bundles: list[ChunkEvidenceBundle] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    processing_notes: list[ProcessingNote] = Field(default_factory=list)
    analysis_instructions: str
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    retrieval_query: str | None = None
    rubric_id: str
    rubric_version: str
    is_mock_data: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerReasoningTrace(ContractModel):
    """Trace of the shared worker-analysis path before future specialization."""
    analysis_client_name: str
    model_name: str
    used_mock_client: bool = False
    selected_chunk_ids: list[str] = Field(default_factory=list)
    selected_section_ids: list[str] = Field(default_factory=list)
    reasoning_notes: list[str] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)
    prompt_context_preview: str | None = None
    raw_response_present: bool = False


class WorkerAnalysisOutput(ContractModel):
    """Internal shared analysis result returned before mapping to WorkerOutput."""
    run_id: str
    ticker: str
    worker_name: str
    document_type: DocumentType
    status: ProcessingStatus = ProcessingStatus.PENDING
    summary: str | None = None
    confidence_score: float | None = None
    scores: list[AnalysisScore] = Field(default_factory=list)
    findings: list[AnalysisFinding] = Field(default_factory=list)
    evidence: list[EvidenceSnippet] = Field(default_factory=list)
    warnings: list[AnalysisWarning] = Field(default_factory=list)
    issues: list[AnalysisIssue] = Field(default_factory=list)
    reasoning_trace: WorkerReasoningTrace | None = None
    raw_output: dict[str, Any] = Field(default_factory=dict)
    is_mock_analysis: bool = False
    generated_at: datetime = Field(default_factory=now_utc)

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence_score(cls, value: float | None) -> float | None:
        return _validate_optional_range(value, 0.0, 1.0, "confidence_score")


class AnalysisClientRequest(ContractModel):
    """Normalized request sent to a shared analysis client."""
    worker_name: str
    document_type: DocumentType
    analysis_input: WorkerAnalysisInput
    prompt_context: dict[str, Any] = Field(default_factory=dict)
    prompt_text: str | None = None
    rubric: AnalysisRubric
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisClientResponse(ContractModel):
    """Provider-agnostic response returned by a shared analysis client."""
    client_name: str
    model_name: str
    status: ProcessingStatus
    raw_output: dict[str, Any] | None = None
    raw_text: str | None = None
    warnings: list[AnalysisWarning] = Field(default_factory=list)
    is_mock: bool = False
    generated_at: datetime = Field(default_factory=now_utc)
