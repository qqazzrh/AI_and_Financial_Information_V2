"""Retrieval base classes, validation, scoring, selection, and provenance helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

from pipeline.config import MIN_DATETIME_UTC, now_utc
from pipeline.enums import DocumentType, ProcessingStatus, SourceFamily
from pipeline.models import (
    CandidateEvaluation,
    DocumentMetadata,
    DocumentValidationResult,
    FetchedDocumentContent,
    PipelineError,
    ProvenanceRecord,
    RawRetrievalCandidateMetadata,
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalResult,
    RetrievalSelectionDecision,
)

__all__ = [
    "BaseRetrievalAdapter",
    "RetrievalErrorCode",
    "RetrievalError",
    "build_provenance_record",
    "append_candidate_provenance",
    "build_document_metadata_from_candidate",
    "make_retrieval_error",
    "normalize_token",
    "normalize_document_text",
    "evaluate_candidate_relevance",
    "assign_candidate_freshness",
    "build_candidate_duplicate_key",
    "candidate_effective_timestamp",
    "source_preference_index",
    "candidate_priority_signature",
    "selection_sort_key",
    "validate_candidate_document",
    "deduplicate_candidates",
    "select_most_recent_relevant_candidate",
]


# ---------------------------------------------------------------------------
# Simple whitespace-normalize (cell 16 version, used by retrieval)
# ---------------------------------------------------------------------------

def normalize_document_text(raw_text: str | None) -> str | None:
    """Return whitespace-normalized text or None if the input is empty."""
    if raw_text is None:
        return None
    normalized = " ".join(raw_text.split())
    return normalized or None


# ---------------------------------------------------------------------------
# Retrieval-specific error types
# ---------------------------------------------------------------------------

class RetrievalErrorCode(str, Enum):
    """Retrieval-specific error codes surfaced by adapters and selection logic."""

    NO_CANDIDATES_FOUND = "no_candidates_found"
    CANDIDATES_FOUND_BUT_NONE_VALID = "candidates_found_but_none_valid"
    AMBIGUOUS_RECENT_CANDIDATES = "ambiguous_recent_candidates"
    MISSING_DOCUMENT_TEXT = "missing_document_text"
    INSUFFICIENT_METADATA = "insufficient_metadata"
    DOCUMENT_TYPE_MISMATCH = "document_type_mismatch"
    REQUEST_IDENTITY_MISMATCH = "request_identity_mismatch"
    UNSUPPORTED_DOCUMENT_TYPE = "unsupported_document_type"
    ADAPTER_FAILURE = "adapter_failure"


class RetrievalError(PipelineError):
    """Structured retrieval-layer error object."""

    error_code: RetrievalErrorCode
    is_mock_data: bool = False


def make_retrieval_error(
    code: RetrievalErrorCode,
    *,
    message: str,
    stage: str,
    document_type: DocumentType | None = None,
    candidate_id: str | None = None,
    adapter_name: str | None = None,
    recoverable: bool = True,
    is_mock_data: bool = False,
    details: dict[str, Any] | None = None,
) -> RetrievalError:
    """Create a retrieval error without assuming provider-specific fields."""
    return RetrievalError(
        error_code=code,
        message=message,
        stage=stage,
        document_type=document_type,
        candidate_id=candidate_id,
        adapter_name=adapter_name,
        recoverable=recoverable,
        is_mock_data=is_mock_data,
        details=details or {},
    )


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def build_provenance_record(
    *,
    stage: str,
    adapter_name: str,
    candidate_id: str | None = None,
    document_type: DocumentType | None = None,
    source_name: str | None = None,
    source_identifier: str | None = None,
    source_url: str | None = None,
    note: str | None = None,
    ranking_notes: list[str] | None = None,
    validation_notes: list[str] | None = None,
    selection_reason: str | None = None,
    is_mock_data: bool = False,
    metadata: dict[str, Any] | None = None,
) -> ProvenanceRecord:
    """Build a provider-agnostic provenance record for retrieval auditing."""
    return ProvenanceRecord(
        stage=stage,
        adapter_name=adapter_name,
        candidate_id=candidate_id,
        document_type=document_type,
        source_name=source_name,
        source_identifier=source_identifier,
        source_url=source_url,
        retrieved_at=now_utc(),
        note=note,
        ranking_notes=ranking_notes or [],
        validation_notes=validation_notes or [],
        selection_reason=selection_reason,
        is_mock_data=is_mock_data,
        metadata=metadata or {},
    )


def append_candidate_provenance(
    candidate: RetrievalCandidate,
    *,
    stage: str,
    note: str | None = None,
    ranking_notes: list[str] | None = None,
    validation_notes: list[str] | None = None,
    selection_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RetrievalCandidate:
    """Append one audit trail record to a normalized candidate."""
    candidate.provenance.append(
        build_provenance_record(
            stage=stage,
            adapter_name=candidate.adapter_name,
            candidate_id=candidate.candidate_id,
            document_type=candidate.document_type,
            source_name=candidate.source_name,
            source_identifier=candidate.source_identifier,
            source_url=candidate.source_url,
            note=note,
            ranking_notes=ranking_notes,
            validation_notes=validation_notes,
            selection_reason=selection_reason,
            is_mock_data=candidate.is_mock_data,
            metadata=metadata,
        )
    )
    return candidate


def build_document_metadata_from_candidate(candidate: RetrievalCandidate) -> DocumentMetadata:
    """Convert a selected retrieval candidate into downstream document metadata."""
    return DocumentMetadata(
        document_id=candidate.candidate_id,
        ticker=candidate.ticker or "UNKNOWN",
        document_type=candidate.document_type,
        company_name=candidate.company_name,
        title=candidate.title,
        source_name=candidate.source_name,
        source_identifier=candidate.source_identifier,
        source_family=candidate.source_family,
        source_url=candidate.source_url,
        published_at=candidate.published_at,
        updated_at=candidate.updated_at,
        event_date=candidate.event_date,
        retrieved_at=candidate.retrieved_at,
        is_structured_source=candidate.is_structured_source,
        is_mock_data=candidate.is_mock_data,
        raw_metadata=candidate.raw_metadata,
        notes=(candidate.validation_notes + candidate.selection_notes),
    )


# ---------------------------------------------------------------------------
# Token helpers and scoring
# ---------------------------------------------------------------------------

def normalize_token(value: str | None) -> str:
    """Lowercase and collapse whitespace for simple identity matching."""
    if not value:
        return ""
    return " ".join(value.lower().split())


def candidate_effective_timestamp(candidate: RetrievalCandidate) -> datetime:
    """Return the effective timestamp used for freshness ranking."""
    return candidate.effective_timestamp() or MIN_DATETIME_UTC


def source_preference_index(source_family: SourceFamily, request: RetrievalRequest) -> int:
    """Return the configured source-family order index for tie-breaking."""
    preferences = request.source_preferences.preferred_source_families
    if source_family in preferences:
        return preferences.index(source_family)
    return len(preferences)


def build_candidate_duplicate_key(candidate: RetrievalCandidate) -> str:
    """Build a simple duplicate-group key for superseded-version handling."""
    if candidate.source_identifier:
        return f"source_identifier::{normalize_token(candidate.source_identifier)}"
    if candidate.source_url:
        return f"source_url::{normalize_token(candidate.source_url)}"
    normalized_title = normalize_token(candidate.title) or "untitled"
    date_key = ""
    if candidate.event_date is not None:
        date_key = candidate.event_date.isoformat()
    elif candidate.published_at is not None:
        date_key = candidate.published_at.date().isoformat()
    elif candidate.updated_at is not None:
        date_key = candidate.updated_at.date().isoformat()
    return f"title_date::{candidate.document_type.value}::{normalized_title}::{date_key}"


def _identity_alignment_score(candidate: RetrievalCandidate, request: RetrievalRequest) -> tuple[float, list[str], bool]:
    notes: list[str] = []
    request_ticker = normalize_token(request.ticker)
    candidate_ticker = normalize_token(candidate.ticker)
    if candidate_ticker:
        if candidate_ticker == request_ticker:
            notes.append("Ticker matches request.")
            return 25.0, notes, True
        notes.append("Ticker does not match request.")
        return -25.0, notes, False

    alias_tokens = [normalize_token(request.company_name)] if request.company_name else []
    alias_tokens.extend(normalize_token(alias) for alias in request.company_aliases if alias)
    metadata_blob = normalize_token(" ".join(filter(None, [candidate.company_name, candidate.title, candidate.source_name])))
    if any(alias and alias in metadata_blob for alias in alias_tokens):
        notes.append("Company name or alias aligns with candidate metadata.")
        return 18.0, notes, True

    notes.append("Identity could not be confirmed from candidate metadata.")
    return 0.0, notes, False


def _title_alignment_score(candidate: RetrievalCandidate, request: RetrievalRequest) -> tuple[float, list[str]]:
    notes: list[str] = []
    score = 0.0
    normalized_title = normalize_token(candidate.title)
    if not normalized_title:
        notes.append("Title is missing; no title-alignment bonus applied.")
        return score, notes

    if normalize_token(request.ticker) and normalize_token(request.ticker) in normalized_title:
        score += 5.0
        notes.append("Title contains the requested ticker.")

    alias_tokens = [normalize_token(request.company_name)] if request.company_name else []
    alias_tokens.extend(normalize_token(alias) for alias in request.company_aliases if alias)
    if any(alias and alias in normalized_title for alias in alias_tokens):
        score += 5.0
        notes.append("Title contains the company name or alias.")

    if score == 0.0:
        notes.append("Title provides no direct identity bonus.")
    return score, notes


def evaluate_candidate_relevance(candidate: RetrievalCandidate, request: RetrievalRequest) -> CandidateEvaluation:
    """Assign transparent heuristic relevance scores without LLM calls."""
    notes: list[str] = []

    if candidate.document_type == request.document_type:
        type_score = 50.0
        notes.append("Document type matches request.")
    else:
        type_score = 0.0
        notes.append("Document type does not match request.")

    identity_score, identity_notes, identity_ok = _identity_alignment_score(candidate, request)
    notes.extend(identity_notes)

    title_score, title_notes = _title_alignment_score(candidate, request)
    notes.extend(title_notes)

    if request.source_preferences.prefer_structured_source and candidate.is_structured_source:
        structured_bonus = 5.0
        notes.append("Structured source bonus applied.")
    else:
        structured_bonus = 0.0

    preference_index = source_preference_index(candidate.source_family, request)
    source_bonus_lookup = {0: 5.0, 1: 3.0, 2: 1.0}
    source_preference_bonus = source_bonus_lookup.get(preference_index, 0.0)
    if source_preference_bonus:
        notes.append("Source-family preference bonus applied.")

    if candidate.source_family == SourceFamily.PERMITTED_SECONDARY and not request.source_preferences.allow_secondary_sources:
        notes.append("Secondary sources are disabled for this request.")
    if candidate.source_family == SourceFamily.UNKNOWN and not request.source_preferences.allow_unknown_sources:
        notes.append("Unknown source families are disabled for this request.")

    relevance_score = type_score + identity_score + title_score + structured_bonus + source_preference_bonus
    validation_ok = bool(candidate.validation and candidate.validation.is_valid)
    source_allowed = not (
        (candidate.source_family == SourceFamily.PERMITTED_SECONDARY and not request.source_preferences.allow_secondary_sources)
        or (candidate.source_family == SourceFamily.UNKNOWN and not request.source_preferences.allow_unknown_sources)
        or (candidate.is_mock_data and not request.source_preferences.allow_mock_data)
    )

    is_relevant = bool(
        validation_ok
        and source_allowed
        and type_score > 0.0
        and identity_score >= 0.0
        and relevance_score >= request.minimum_relevance_score
    )
    if not validation_ok:
        notes.append("Candidate failed validation and cannot be selected.")
    if not source_allowed:
        notes.append("Candidate source is not allowed by request preferences.")
    if is_relevant:
        notes.append("Candidate cleared the minimum relevance threshold.")
    else:
        notes.append("Candidate did not clear the minimum relevance threshold.")

    evaluation = CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        is_relevant=is_relevant,
        relevance_score=relevance_score,
        freshness_score=0.0,
        total_score=relevance_score,
        type_compatibility_score=type_score,
        identity_alignment_score=identity_score,
        title_alignment_score=title_score,
        structured_source_bonus=structured_bonus,
        source_preference_bonus=source_preference_bonus,
        duplicate_key=build_candidate_duplicate_key(candidate),
        effective_timestamp=candidate.effective_timestamp(),
        evaluation_notes=notes,
    )
    candidate.evaluation = evaluation
    candidate.relevance_notes.extend(notes)
    candidate.duplicate_key = evaluation.duplicate_key
    return evaluation


def assign_candidate_freshness(candidates: Sequence[RetrievalCandidate], request: RetrievalRequest) -> list[RetrievalCandidate]:
    """Apply freshness scores after validation and duplicate handling."""
    ranked_for_freshness = sorted(
        [candidate for candidate in candidates if candidate.superseded_by_candidate_id is None],
        key=lambda candidate: (
            -candidate_effective_timestamp(candidate).timestamp(),
            source_preference_index(candidate.source_family, request),
            -int(candidate.is_structured_source),
            candidate.candidate_id,
        ),
    )

    for rank, candidate in enumerate(ranked_for_freshness):
        freshness_score = 0.0
        effective_timestamp = candidate.evaluation.effective_timestamp if candidate.evaluation else candidate.effective_timestamp()
        if effective_timestamp is not None:
            freshness_score = max(0.0, request.freshness_bonus_max - (rank * request.freshness_rank_decay))
            candidate.freshness_notes.append(
                f"Freshness rank {rank + 1} among {len(ranked_for_freshness)} non-superseded candidates."
            )
        else:
            candidate.freshness_notes.append("No timestamp available; freshness bonus set to 0.")

        if candidate.evaluation is None:
            candidate.evaluation = evaluate_candidate_relevance(candidate, request)
        candidate.evaluation.freshness_score = freshness_score
        candidate.evaluation.total_score = candidate.evaluation.relevance_score + freshness_score
    return list(candidates)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_candidate_document(candidate: RetrievalCandidate, request: RetrievalRequest) -> DocumentValidationResult:
    """Validate one normalized candidate using generic retrieval rules."""
    notes: list[str] = []
    issues: list[PipelineError] = []

    normalized_text = normalize_document_text(candidate.document_text)
    has_usable_text = bool(normalized_text and len(normalized_text) >= request.minimum_text_chars)
    has_minimum_metadata = bool(candidate.title or candidate.source_identifier or candidate.source_url)
    type_matches_request = candidate.document_type == request.document_type

    identity_score, identity_notes, identity_matches_request = _identity_alignment_score(candidate, request)
    notes.extend(identity_notes)

    if has_usable_text:
        notes.append("Candidate contains usable text.")
    else:
        notes.append("Candidate text is missing or too short.")
        issues.append(
            make_retrieval_error(
                RetrievalErrorCode.MISSING_DOCUMENT_TEXT,
                message="Candidate is missing usable text.",
                stage="validate_document",
                document_type=request.document_type,
                candidate_id=candidate.candidate_id,
                adapter_name=candidate.adapter_name,
                is_mock_data=candidate.is_mock_data,
            )
        )

    if has_minimum_metadata:
        notes.append("Candidate has minimally sufficient metadata.")
    else:
        notes.append("Candidate is missing key metadata fields.")
        issues.append(
            make_retrieval_error(
                RetrievalErrorCode.INSUFFICIENT_METADATA,
                message="Candidate is missing title, source identifier, and source URL.",
                stage="validate_document",
                document_type=request.document_type,
                candidate_id=candidate.candidate_id,
                adapter_name=candidate.adapter_name,
                is_mock_data=candidate.is_mock_data,
            )
        )

    if type_matches_request:
        notes.append("Candidate document type matches the request.")
    else:
        notes.append("Candidate document type does not match the request.")
        issues.append(
            make_retrieval_error(
                RetrievalErrorCode.DOCUMENT_TYPE_MISMATCH,
                message="Candidate document type does not match the request.",
                stage="validate_document",
                document_type=request.document_type,
                candidate_id=candidate.candidate_id,
                adapter_name=candidate.adapter_name,
                is_mock_data=candidate.is_mock_data,
            )
        )

    if identity_matches_request:
        notes.append("Candidate identity aligns with the request.")
    else:
        notes.append("Candidate identity could not be confirmed for the request.")
        if identity_score < 0.0:
            issues.append(
                make_retrieval_error(
                    RetrievalErrorCode.REQUEST_IDENTITY_MISMATCH,
                    message="Candidate ticker conflicts with the request ticker.",
                    stage="validate_document",
                    document_type=request.document_type,
                    candidate_id=candidate.candidate_id,
                    adapter_name=candidate.adapter_name,
                    is_mock_data=candidate.is_mock_data,
                )
            )

    is_valid = has_usable_text and type_matches_request and (identity_matches_request or identity_score == 0.0)
    is_partial = is_valid and not has_minimum_metadata

    if is_valid and is_partial:
        status = ProcessingStatus.PARTIAL
    elif is_valid:
        status = ProcessingStatus.SUCCESS
    elif not has_usable_text:
        status = ProcessingStatus.EXTRACTION_FAILED
    else:
        status = ProcessingStatus.SELECTION_FAILED

    return DocumentValidationResult(
        candidate_id=candidate.candidate_id,
        status=status,
        is_valid=is_valid,
        is_partial=is_partial,
        has_usable_text=has_usable_text,
        has_minimum_metadata=has_minimum_metadata,
        type_matches_request=type_matches_request,
        identity_matches_request=identity_matches_request or identity_score == 0.0,
        validation_notes=notes,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_candidates(
    candidates: Sequence[RetrievalCandidate],
    request: RetrievalRequest,
) -> list[RetrievalCandidate]:
    """Mark older duplicates as superseded and keep the freshest candidate per duplicate group."""
    grouped: dict[str, list[RetrievalCandidate]] = {}
    for candidate in candidates:
        duplicate_key = candidate.duplicate_key or build_candidate_duplicate_key(candidate)
        candidate.duplicate_key = duplicate_key
        grouped.setdefault(duplicate_key, []).append(candidate)

    deduplicated: list[RetrievalCandidate] = []
    for duplicate_key, group in grouped.items():
        ordered_group = sorted(
            group,
            key=lambda candidate: (
                -candidate_effective_timestamp(candidate).timestamp(),
                source_preference_index(candidate.source_family, request),
                -int(candidate.is_structured_source),
                candidate.candidate_id,
            ),
        )
        winner = ordered_group[0]
        deduplicated.append(winner)
        for superseded_candidate in ordered_group[1:]:
            superseded_candidate.superseded_by_candidate_id = winner.candidate_id
            superseded_candidate.selection_notes.append(
                f"Superseded by {winner.candidate_id} within duplicate group {duplicate_key}."
            )
    return deduplicated


def candidate_priority_signature(candidate: RetrievalCandidate, request: RetrievalRequest) -> tuple[Any, ...]:
    """Return the explicit tie-break signature used by selection."""
    evaluation = candidate.evaluation or CandidateEvaluation(candidate_id=candidate.candidate_id)
    return (
        round(evaluation.total_score, 6),
        candidate_effective_timestamp(candidate).timestamp(),
        -source_preference_index(candidate.source_family, request),
        int(candidate.is_structured_source),
        (candidate.updated_at or MIN_DATETIME_UTC).timestamp(),
        (candidate.published_at or MIN_DATETIME_UTC).timestamp(),
    )


def selection_sort_key(candidate: RetrievalCandidate, request: RetrievalRequest) -> tuple[Any, ...]:
    """Return the deterministic sorting key for winner selection."""
    evaluation = candidate.evaluation or CandidateEvaluation(candidate_id=candidate.candidate_id)
    return (
        -evaluation.total_score,
        -candidate_effective_timestamp(candidate).timestamp(),
        source_preference_index(candidate.source_family, request),
        -int(candidate.is_structured_source),
        -(candidate.updated_at or MIN_DATETIME_UTC).timestamp(),
        -(candidate.published_at or MIN_DATETIME_UTC).timestamp(),
        candidate.candidate_id,
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_most_recent_relevant_candidate(
    *,
    candidates: Sequence[RetrievalCandidate],
    request: RetrievalRequest,
    adapter_name: str,
) -> RetrievalResult:
    """Select the most recent relevant valid candidate with explicit ranking notes."""
    result = RetrievalResult(
        request=request,
        adapter_name=adapter_name,
        candidates=list(candidates),
        is_mock_result=bool(request.is_mock_request or any(candidate.is_mock_data for candidate in candidates)),
    )

    if not candidates:
        result.status = ProcessingStatus.NO_DOCUMENT
        result.issues.append(
            make_retrieval_error(
                RetrievalErrorCode.NO_CANDIDATES_FOUND,
                message="No normalized candidates were available for selection.",
                stage="select_candidate",
                document_type=request.document_type,
                adapter_name=adapter_name,
                is_mock_data=result.is_mock_result,
            )
        )
        return result

    for candidate in result.candidates:
        if candidate.validation is None:
            candidate.validation = validate_candidate_document(candidate, request)
            candidate.validation_notes.extend(candidate.validation.validation_notes)
        if candidate.evaluation is None:
            evaluate_candidate_relevance(candidate, request)

    deduplicated_candidates = deduplicate_candidates(result.candidates, request)
    assign_candidate_freshness(deduplicated_candidates, request)

    valid_candidates = [candidate for candidate in deduplicated_candidates if candidate.validation and candidate.validation.is_valid]
    relevant_candidates = [candidate for candidate in valid_candidates if candidate.evaluation and candidate.evaluation.is_relevant]

    if not valid_candidates:
        result.status = ProcessingStatus.SELECTION_FAILED
        result.issues.append(
            make_retrieval_error(
                RetrievalErrorCode.CANDIDATES_FOUND_BUT_NONE_VALID,
                message="Candidates were found, but none passed validation.",
                stage="select_candidate",
                document_type=request.document_type,
                adapter_name=adapter_name,
                is_mock_data=result.is_mock_result,
                details={"candidate_count": len(result.candidates)},
            )
        )
        result.selection_decision.rejected_candidate_ids = [candidate.candidate_id for candidate in result.candidates]
        return result

    if not relevant_candidates:
        result.status = ProcessingStatus.SELECTION_FAILED
        result.issues.append(
            make_retrieval_error(
                RetrievalErrorCode.CANDIDATES_FOUND_BUT_NONE_VALID,
                message="Validated candidates existed, but none were relevant enough to select.",
                stage="select_candidate",
                document_type=request.document_type,
                adapter_name=adapter_name,
                is_mock_data=result.is_mock_result,
                details={"valid_candidate_count": len(valid_candidates)},
            )
        )
        result.selection_decision.rejected_candidate_ids = [candidate.candidate_id for candidate in result.candidates]
        return result

    ranked_candidates = sorted(relevant_candidates, key=lambda candidate: selection_sort_key(candidate, request))
    ranking_notes: list[str] = []
    for rank, candidate in enumerate(ranked_candidates, start=1):
        evaluation = candidate.evaluation or CandidateEvaluation(candidate_id=candidate.candidate_id)
        ranking_notes.append(
            " | ".join(
                [
                    f"rank={rank}",
                    f"candidate_id={candidate.candidate_id}",
                    f"total_score={evaluation.total_score:.1f}",
                    f"relevance_score={evaluation.relevance_score:.1f}",
                    f"freshness_score={evaluation.freshness_score:.1f}",
                    f"effective_timestamp={candidate_effective_timestamp(candidate).isoformat()}",
                ]
            )
        )

    winner = ranked_candidates[0]
    ambiguity_notes: list[str] = []
    tie_break_notes: list[str] = [
        "Tie-break order: total_score desc, effective_timestamp desc, source preference asc, structured source desc, updated_at desc, published_at desc, candidate_id asc."
    ]
    if len(ranked_candidates) > 1:
        top_signature = candidate_priority_signature(ranked_candidates[0], request)
        second_signature = candidate_priority_signature(ranked_candidates[1], request)
        if top_signature == second_signature:
            ambiguity_notes.append(
                "Top candidates matched on all intended ranking dimensions before candidate_id tie-break. Deterministic selection used candidate_id."
            )
            result.issues.append(
                make_retrieval_error(
                    RetrievalErrorCode.AMBIGUOUS_RECENT_CANDIDATES,
                    message="Multiple top candidates remained tied after ranking and required candidate_id tie-break.",
                    stage="select_candidate",
                    document_type=request.document_type,
                    adapter_name=adapter_name,
                    is_mock_data=result.is_mock_result,
                    details={
                        "top_candidate_ids": [ranked_candidates[0].candidate_id, ranked_candidates[1].candidate_id],
                    },
                )
            )

    selection_reason = (
        f"Selected {winner.candidate_id} as the highest-ranked relevant candidate after validation, duplicate handling, and freshness ranking."
    )
    winner.selection_notes.append(selection_reason)
    append_candidate_provenance(
        winner,
        stage="select_candidate",
        note="Candidate selected as retrieval winner.",
        ranking_notes=ranking_notes,
        selection_reason=selection_reason,
    )

    result.selected_candidate = winner
    result.selection_decision = RetrievalSelectionDecision(
        selected_candidate_id=winner.candidate_id,
        selected_reason=selection_reason,
        rejected_candidate_ids=[candidate.candidate_id for candidate in result.candidates if candidate.candidate_id != winner.candidate_id],
        ranking_notes=ranking_notes,
        tie_break_notes=tie_break_notes,
        ambiguity_notes=ambiguity_notes,
    )
    result.provenance.append(
        build_provenance_record(
            stage="select_candidate",
            adapter_name=adapter_name,
            candidate_id=winner.candidate_id,
            document_type=request.document_type,
            source_name=winner.source_name,
            source_identifier=winner.source_identifier,
            source_url=winner.source_url,
            note="Selection completed.",
            ranking_notes=ranking_notes,
            selection_reason=selection_reason,
            is_mock_data=result.is_mock_result,
        )
    )

    if winner.validation and winner.validation.is_partial:
        result.status = ProcessingStatus.PARTIAL
    elif ambiguity_notes:
        result.status = ProcessingStatus.PARTIAL
    else:
        result.status = ProcessingStatus.SUCCESS
    return result


# ---------------------------------------------------------------------------
# Base adapter class
# ---------------------------------------------------------------------------

class BaseRetrievalAdapter(ABC):
    """Base retrieval adapter interface for one disclosure type."""

    adapter_name: ClassVar[str] = "base_retrieval_adapter"
    document_type: ClassVar[DocumentType]

    @abstractmethod
    def search_candidates(self, request: RetrievalRequest) -> list[RawRetrievalCandidateMetadata]:
        """Return raw candidate metadata records for the request."""

    @abstractmethod
    def fetch_document(
        self,
        raw_candidate: RawRetrievalCandidateMetadata,
        request: RetrievalRequest,
    ) -> FetchedDocumentContent:
        """Return document content for one raw candidate."""

    def normalize_candidate(
        self,
        raw_candidate: RawRetrievalCandidateMetadata,
        fetched_document: FetchedDocumentContent,
        request: RetrievalRequest,
    ) -> RetrievalCandidate:
        """Normalize one raw candidate and fetched document into the internal schema."""
        normalized_text = normalize_document_text(fetched_document.document_text)
        candidate = RetrievalCandidate(
            candidate_id=f"{self.adapter_name}:{raw_candidate.raw_candidate_id}",
            adapter_name=self.adapter_name,
            document_type=raw_candidate.document_type,
            ticker=(raw_candidate.ticker or request.ticker or "").upper() or None,
            company_name=raw_candidate.company_name or request.company_name,
            source_name=raw_candidate.source_name,
            source_identifier=raw_candidate.source_identifier,
            source_url=raw_candidate.source_url,
            source_family=raw_candidate.source_family,
            title=raw_candidate.title,
            published_at=raw_candidate.published_at,
            updated_at=raw_candidate.updated_at,
            event_date=raw_candidate.event_date,
            retrieved_at=fetched_document.fetched_at,
            document_text=normalized_text,
            is_structured_source=raw_candidate.is_structured_source,
            is_mock_data=bool(raw_candidate.is_mock_data or fetched_document.is_mock_data or request.is_mock_request),
            raw_metadata={**raw_candidate.raw_metadata, **fetched_document.raw_payload},
        )
        candidate.duplicate_key = build_candidate_duplicate_key(candidate)
        append_candidate_provenance(
            candidate,
            stage="normalize",
            note="Candidate normalized into internal retrieval schema.",
            metadata={"fetch_status": fetched_document.fetch_status.value},
        )
        return candidate

    def validate_document(
        self,
        candidate: RetrievalCandidate,
        request: RetrievalRequest,
    ) -> DocumentValidationResult:
        """Validate one normalized candidate using generic notebook logic."""
        return validate_candidate_document(candidate, request)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Run a full retrieval pass using provider-agnostic notebook logic."""
        result = RetrievalResult(
            request=request,
            adapter_name=self.adapter_name,
            is_mock_result=request.is_mock_request,
        )

        if request.document_type != self.document_type:
            result.status = ProcessingStatus.RETRIEVAL_FAILED
            result.issues.append(
                make_retrieval_error(
                    RetrievalErrorCode.UNSUPPORTED_DOCUMENT_TYPE,
                    message="Request document type does not match adapter type.",
                    stage="retrieve",
                    document_type=request.document_type,
                    adapter_name=self.adapter_name,
                    recoverable=False,
                    is_mock_data=request.is_mock_request,
                    details={
                        "request_document_type": request.document_type.value,
                        "adapter_document_type": self.document_type.value,
                    },
                )
            )
            return result

        try:
            raw_candidates = self.search_candidates(request)[: request.max_candidates]
        except Exception as exc:
            result.status = ProcessingStatus.RETRIEVAL_FAILED
            result.issues.append(
                make_retrieval_error(
                    RetrievalErrorCode.ADAPTER_FAILURE,
                    message=f"Adapter search failed: {exc}",
                    stage="search_candidates",
                    document_type=request.document_type,
                    adapter_name=self.adapter_name,
                    recoverable=True,
                    is_mock_data=request.is_mock_request,
                )
            )
            return result

        if not raw_candidates:
            result.status = ProcessingStatus.NO_DOCUMENT
            result.issues.append(
                make_retrieval_error(
                    RetrievalErrorCode.NO_CANDIDATES_FOUND,
                    message="Adapter returned no candidates for the request.",
                    stage="search_candidates",
                    document_type=request.document_type,
                    adapter_name=self.adapter_name,
                    recoverable=True,
                    is_mock_data=request.is_mock_request,
                )
            )
            result.provenance.append(
                build_provenance_record(
                    stage="search_candidates",
                    adapter_name=self.adapter_name,
                    document_type=request.document_type,
                    note="No candidates returned.",
                    is_mock_data=request.is_mock_request,
                )
            )
            return result

        normalized_candidates: list[RetrievalCandidate] = []
        for raw_candidate in raw_candidates:
            try:
                fetched_document = self.fetch_document(raw_candidate, request)
                candidate = self.normalize_candidate(raw_candidate, fetched_document, request)
                validation = self.validate_document(candidate, request)
                candidate.validation = validation
                candidate.validation_notes.extend(validation.validation_notes)
                append_candidate_provenance(
                    candidate,
                    stage="validate",
                    note="Candidate validated against generic notebook rules.",
                    validation_notes=validation.validation_notes,
                    metadata={
                        "is_valid": validation.is_valid,
                        "is_partial": validation.is_partial,
                        "status": validation.status.value,
                    },
                )
                normalized_candidates.append(candidate)
            except Exception as exc:
                result.issues.append(
                    make_retrieval_error(
                        RetrievalErrorCode.ADAPTER_FAILURE,
                        message=f"Candidate normalization failed: {exc}",
                        stage="normalize_candidate",
                        document_type=request.document_type,
                        candidate_id=getattr(raw_candidate, "raw_candidate_id", None),
                        adapter_name=self.adapter_name,
                        recoverable=True,
                        is_mock_data=request.is_mock_request,
                    )
                )

        selection_result = select_most_recent_relevant_candidate(
            candidates=normalized_candidates,
            request=request,
            adapter_name=self.adapter_name,
        )
        selection_result.issues.extend(result.issues)
        selection_result.is_mock_result = bool(
            request.is_mock_request or any(candidate.is_mock_data for candidate in normalized_candidates)
        )
        selection_result.provenance.insert(
            0,
            build_provenance_record(
                stage="retrieve",
                adapter_name=self.adapter_name,
                document_type=request.document_type,
                note=f"Adapter processed {len(raw_candidates)} raw candidate(s).",
                is_mock_data=selection_result.is_mock_result,
            ),
        )
        return selection_result
