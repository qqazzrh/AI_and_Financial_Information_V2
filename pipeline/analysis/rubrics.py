"""Analysis rubrics, score normalization, and shared parsing utilities."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pipeline.config import (
    SENTIMENT_SCORE_MAX,
    SENTIMENT_SCORE_MIN,
)
from pipeline.enums import (
    AnalysisDimension,
    AnalysisIssueSeverity,
    EvidenceInterpretation,
    EvidenceType,
    ProcessingStatus,
    SectionKind,
    SentimentLabel,
)
from pipeline.models import (
    AnalysisFinding,
    AnalysisIssue,
    AnalysisRubric,
    AnalysisScore,
    AnalysisWarning,
    ChunkEvidenceBundle,
    DocumentMetadata,
    EvidenceSnippet,
    ProcessedDocument,
    ProvenanceRecord,
    RubricAnchor,
    RubricDimensionDefinition,
    SectionContextSummary,
    SentimentAssessment,
    ToneAssessment,
    WorkerAnalysisInput,
    WorkerAnalysisOutput,
    WorkerInput,
    WorkerReasoningTrace,
    _validate_optional_range,
)

__all__ = [
    "ANALYSIS_DIMENSION_SCORE_RANGES",
    "ANALYSIS_RUBRIC_BY_DIMENSION",
    "DEFAULT_ANALYSIS_RUBRIC",
    "DEFAULT_GENERIC_ANALYSIS_INSTRUCTIONS",
    "build_chunk_evidence_bundles",
    "build_evidence_snippet_from_bundle",
    "build_generic_analysis_query",
    "build_section_context_summaries",
    "build_sentiment_assessment_from_scores",
    "build_standardized_analysis_score",
    "build_tone_assessment_from_scores",
    "build_worker_analysis_input",
    "clamp_score_value",
    "coerce_analysis_payload",
    "coerce_float",
    "collect_analysis_warnings",
    "deduplicate_analysis_items",
    "describe_worker_analysis_output_schema",
    "extract_structured_findings",
    "get_analysis_dimension_score_range",
    "infer_evidence_type",
    "make_analysis_issue",
    "make_analysis_warning",
    "map_raw_evidence_references",
    "normalize_analysis_scores",
    "normalize_dimension_label",
    "normalize_sentiment_label",
    "parse_worker_analysis_output",
    "analysis_scores_to_lookup",
]

# ---------------------------------------------------------------------------
# Score ranges
# ---------------------------------------------------------------------------

ANALYSIS_DIMENSION_SCORE_RANGES: dict[AnalysisDimension, tuple[float, float]] = {
    AnalysisDimension.SENTIMENT: (SENTIMENT_SCORE_MIN, SENTIMENT_SCORE_MAX),
    AnalysisDimension.UNCERTAINTY: (0.0, 1.0),
    AnalysisDimension.FOGGING: (0.0, 1.0),
    AnalysisDimension.HEDGING: (0.0, 1.0),
    AnalysisDimension.PROMOTIONAL_TONE: (0.0, 1.0),
    AnalysisDimension.CLARITY: (0.0, 1.0),
    AnalysisDimension.MATERIALITY: (0.0, 1.0),
    AnalysisDimension.COMPLETENESS: (0.0, 1.0),
}


def get_analysis_dimension_score_range(dimension: AnalysisDimension) -> tuple[float, float]:
    """Return the valid score range for one analysis dimension."""
    return ANALYSIS_DIMENSION_SCORE_RANGES[dimension]


# ---------------------------------------------------------------------------
# Default rubric
# ---------------------------------------------------------------------------

DEFAULT_ANALYSIS_RUBRIC = AnalysisRubric(
    rubric_id="shared_worker_analysis_rubric",
    version="0.1.0",
    summary="Document-agnostic rubric for evidence-based biotech disclosure worker analysis.",
    core_principles=[
        "Anchor every score to cited evidence rather than unsupported impression.",
        "Separate factual directionality from rhetorical tone and presentation style.",
        "Treat uncertainty as unresolved dependency or missing specificity, not automatic negativity.",
        "Reward balance, specificity, and completeness while penalizing opacity and unsupported emphasis.",
        "Mark missing evidence as a warning instead of fabricating precision.",
    ],
    sentiment_labels=[
        SentimentLabel.POSITIVE,
        SentimentLabel.NEGATIVE,
        SentimentLabel.NEUTRAL,
        SentimentLabel.MIXED,
        SentimentLabel.INSUFFICIENT_EVIDENCE,
    ],
    dimension_definitions=[
        RubricDimensionDefinition(
            dimension=AnalysisDimension.SENTIMENT,
            minimum_score=-1.0,
            maximum_score=1.0,
            objective="Measure the directional implication of disclosed facts, not the emotional tone of wording alone.",
            evidence_expectation="Use outcome-bearing evidence and note opposing facts before assigning a label.",
            scoring_guidance=[
                "Positive sentiment should be assigned when disclosed facts suggest forward progress, active development, regulatory advancement, strong enrollment, partnerships, or revenue growth — even if stated factually.",
                "Negative sentiment should reflect disclosed setbacks, safety concerns, trial failures, revenue decline, regulatory delays, or material risks.",
                "Mixed sentiment is appropriate when material positive and negative evidence coexist within the same disclosure.",
                "Neutral sentiment should be rare — reserve it only for purely procedural or administrative disclosures with no investment implication whatsoever.",
                "Active clinical trials, FDA submissions, and ongoing development programs inherently carry positive directional implication for investors unless negative signals are present.",
            ],
            anchors=[
                RubricAnchor(score=-1.0, description="Strongly negative disclosed implications with clear evidence."),
                RubricAnchor(score=0.0, description="Directionally neutral or insufficiently directional evidence."),
                RubricAnchor(score=1.0, description="Strongly positive disclosed implications with clear evidence."),
            ],
        ),
        RubricDimensionDefinition(
            dimension=AnalysisDimension.UNCERTAINTY,
            minimum_score=0.0,
            maximum_score=1.0,
            objective="Measure how much the disclosure leaves key outcomes dependent on unresolved events, timing, or missing specifics.",
            evidence_expectation="Reference unresolved milestones, contingencies, missing dates, or acknowledged unknowns.",
            scoring_guidance=[
                "High uncertainty reflects unresolved dependencies, incomplete timing, or significant unknowns.",
                "Moderate uncertainty reflects partial specificity with meaningful open questions remaining.",
                "Low uncertainty reflects concrete timing, scope, and next steps.",
            ],
            anchors=[
                RubricAnchor(score=0.0, description="Low uncertainty; facts and next steps are specific."),
                RubricAnchor(score=0.5, description="Moderate uncertainty; some specifics are present but gaps remain."),
                RubricAnchor(score=1.0, description="High uncertainty; key outcomes remain unresolved or poorly specified."),
            ],
        ),
        RubricDimensionDefinition(
            dimension=AnalysisDimension.FOGGING,
            minimum_score=0.0,
            maximum_score=1.0,
            objective="Measure opacity, evasiveness, or lack of operational specificity in the disclosure.",
            evidence_expectation="Cite vague summaries, missing detail, or unsupported abstraction; do not treat concise clarity as fogging.",
            scoring_guidance=[
                "High fogging requires evidence that material points are obscured or left vague.",
                "Low fogging reflects concrete details, inspectable numbers, or precise procedural descriptions.",
                "Do not treat uncertainty disclosures by themselves as fogging if they are explicit and specific.",
            ],
            anchors=[
                RubricAnchor(score=0.0, description="Low fogging; the disclosure is explicit and inspectable."),
                RubricAnchor(score=0.5, description="Moderate fogging; some material points are underspecified."),
                RubricAnchor(score=1.0, description="High fogging; material detail is obscured or evasive."),
            ],
        ),
        RubricDimensionDefinition(
            dimension=AnalysisDimension.HEDGING,
            minimum_score=0.0,
            maximum_score=1.0,
            objective="Measure how strongly claims are qualified, conditional, or deferred.",
            evidence_expectation="Tie the score to conditionality, caveats, or claim-softening language patterns without assuming deception.",
            scoring_guidance=[
                "High hedging reflects many qualified claims or repeated conditional framing.",
                "Moderate hedging reflects normal caution around forward-looking statements.",
                "Low hedging reflects direct factual reporting with limited qualification.",
            ],
            anchors=[
                RubricAnchor(score=0.0, description="Low hedging; mostly direct factual reporting."),
                RubricAnchor(score=0.5, description="Moderate hedging; caution and conditional framing are noticeable."),
                RubricAnchor(score=1.0, description="High hedging; claims are heavily qualified or deferred."),
            ],
        ),
        RubricDimensionDefinition(
            dimension=AnalysisDimension.PROMOTIONAL_TONE,
            minimum_score=0.0,
            maximum_score=1.0,
            objective="Measure imbalance between claims and support, not mere polish or competent presentation.",
            evidence_expectation="Cite disproportionate emphasis, unsupported upside framing, or selective omission of balancing context.",
            scoring_guidance=[
                "High promotional tone requires evidence of overreach beyond the disclosed support.",
                "Low promotional tone reflects balanced framing and alignment between claims and cited facts.",
                "Polished language alone is not promotional if the document stays balanced and specific.",
            ],
            anchors=[
                RubricAnchor(score=0.0, description="Low promotional tone; framing stays balanced and supported."),
                RubricAnchor(score=0.5, description="Moderate promotional tone; emphasis is noticeable but partly supported."),
                RubricAnchor(score=1.0, description="High promotional tone; claims materially outpace disclosed support."),
            ],
        ),
        RubricDimensionDefinition(
            dimension=AnalysisDimension.CLARITY,
            minimum_score=0.0,
            maximum_score=1.0,
            objective="Measure how clearly the disclosure explains what happened, why it matters, and what comes next.",
            evidence_expectation="Reference organization, specificity, and ease of tracing material points.",
            scoring_guidance=[
                "High clarity reflects coherent structure and concrete explanations.",
                "Low clarity reflects fragmented or difficult-to-follow material points.",
            ],
            anchors=[
                RubricAnchor(score=0.0, description="Low clarity; material meaning is difficult to trace."),
                RubricAnchor(score=0.5, description="Moderate clarity; key facts are present but somewhat diffuse."),
                RubricAnchor(score=1.0, description="High clarity; material facts and next steps are explicit."),
            ],
        ),
        RubricDimensionDefinition(
            dimension=AnalysisDimension.MATERIALITY,
            minimum_score=0.0,
            maximum_score=1.0,
            objective="Measure how consequential the cited facts appear within the document context.",
            evidence_expectation="Use disclosed financial, regulatory, clinical, or operational consequences to support the score.",
            scoring_guidance=[
                "High materiality reflects consequences that could meaningfully affect the issuer, program, or investors.",
                "Low materiality reflects routine or limited-scope information.",
            ],
            anchors=[
                RubricAnchor(score=0.0, description="Low materiality; limited apparent consequence."),
                RubricAnchor(score=0.5, description="Moderate materiality; meaningful but not central consequences."),
                RubricAnchor(score=1.0, description="High materiality; clearly consequential disclosed facts."),
            ],
        ),
        RubricDimensionDefinition(
            dimension=AnalysisDimension.COMPLETENESS,
            minimum_score=0.0,
            maximum_score=1.0,
            objective="Measure whether the disclosure provides enough context, detail, and balance to evaluate the main point.",
            evidence_expectation="Reference presence or absence of timing, scope, tradeoffs, and balancing detail.",
            scoring_guidance=[
                "High completeness reflects balanced context, scope, and next steps.",
                "Low completeness reflects meaningful omissions that limit interpretation.",
            ],
            anchors=[
                RubricAnchor(score=0.0, description="Low completeness; material context is missing."),
                RubricAnchor(score=0.5, description="Moderate completeness; useful context exists but gaps remain."),
                RubricAnchor(score=1.0, description="High completeness; the document supplies enough context to evaluate the point."),
            ],
        ),
    ],
)

ANALYSIS_RUBRIC_BY_DIMENSION = {
    definition.dimension: definition for definition in DEFAULT_ANALYSIS_RUBRIC.dimension_definitions
}

# ---------------------------------------------------------------------------
# Score coercion helpers
# ---------------------------------------------------------------------------


def coerce_float(value: Any) -> float | None:
    """Safely coerce a raw value to float without raising."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        try:
            return float(normalized)
        except ValueError:
            return None
    return None


def clamp_score_value(
    value: float | None,
    *,
    minimum: float,
    maximum: float,
    warning_code: str,
    dimension: AnalysisDimension,
) -> tuple[float | None, AnalysisWarning | None]:
    """Clamp a raw score into range and emit a structured warning when adjustment is needed."""
    if value is None:
        return None, None
    if minimum <= value <= maximum:
        return value, None
    clamped = max(minimum, min(maximum, value))
    warning = AnalysisWarning(
        issue_code=warning_code,
        message=f"{dimension.value} score {value} was outside [{minimum}, {maximum}] and was clamped.",
        dimension=dimension,
        field_name="score",
        metadata={"raw_score": value, "clamped_score": clamped},
    )
    return clamped, warning


def normalize_sentiment_label(value: Any, score: float | None = None) -> tuple[SentimentLabel, AnalysisWarning | None]:
    """Normalize raw sentiment labels and optionally infer a label from a valid score."""
    if isinstance(value, SentimentLabel):
        return value, None
    if isinstance(value, str):
        normalized = value.strip().lower()
        mapping = {
            "positive": SentimentLabel.POSITIVE,
            "negative": SentimentLabel.NEGATIVE,
            "neutral": SentimentLabel.NEUTRAL,
            "mixed": SentimentLabel.MIXED,
            "insufficient_evidence": SentimentLabel.INSUFFICIENT_EVIDENCE,
            "insufficient evidence": SentimentLabel.INSUFFICIENT_EVIDENCE,
        }
        if normalized in mapping:
            return mapping[normalized], None
    if score is None:
        warning = AnalysisWarning(
            issue_code="sentiment_label_missing",
            message="Sentiment label was missing or invalid and no score was available to infer one.",
            dimension=AnalysisDimension.SENTIMENT,
            field_name="label",
        )
        return SentimentLabel.INSUFFICIENT_EVIDENCE, warning

    if score >= 0.10:
        inferred = SentimentLabel.POSITIVE
    elif score <= -0.10:
        inferred = SentimentLabel.NEGATIVE
    elif abs(score) <= 0.02:
        inferred = SentimentLabel.NEUTRAL
    else:
        inferred = SentimentLabel.MIXED

    warning = AnalysisWarning(
        issue_code="sentiment_label_inferred",
        message="Sentiment label was inferred from the normalized sentiment score using explicit thresholds.",
        dimension=AnalysisDimension.SENTIMENT,
        field_name="label",
        metadata={"inferred_label": inferred.value, "score": score},
    )
    return inferred, warning


def normalize_dimension_label(dimension: AnalysisDimension, score: float | None) -> str | None:
    """Map a normalized score to a small explicit label set."""
    if score is None:
        return None
    if dimension == AnalysisDimension.SENTIMENT:
        sentiment_label, _ = normalize_sentiment_label(None, score)
        return sentiment_label.value
    if score < 0.34:
        return "low"
    if score < 0.67:
        return "moderate"
    return "high"


def build_standardized_analysis_score(
    *,
    dimension: AnalysisDimension,
    raw_score: Any,
    raw_label: Any,
    raw_confidence: Any,
    rationale: Any = None,
    evidence_ids: Sequence[str] | None = None,
    rubric_notes: Sequence[str] | None = None,
) -> tuple[AnalysisScore, list[AnalysisWarning]]:
    """Map raw analysis content into one validated score object plus warnings."""
    warnings: list[AnalysisWarning] = []
    minimum, maximum = get_analysis_dimension_score_range(dimension)

    score_value = coerce_float(raw_score)
    clamped_score, clamp_warning = clamp_score_value(
        score_value,
        minimum=minimum,
        maximum=maximum,
        warning_code="score_out_of_range",
        dimension=dimension,
    )
    if clamp_warning is not None:
        warnings.append(clamp_warning)

    confidence_value = coerce_float(raw_confidence)
    clamped_confidence, confidence_warning = clamp_score_value(
        confidence_value,
        minimum=0.0,
        maximum=1.0,
        warning_code="confidence_out_of_range",
        dimension=dimension,
    )
    if confidence_warning is not None:
        warnings.append(confidence_warning)

    if dimension == AnalysisDimension.SENTIMENT:
        sentiment_label, label_warning = normalize_sentiment_label(raw_label, clamped_score)
        if label_warning is not None:
            warnings.append(label_warning)
        normalized_label = sentiment_label.value
    else:
        normalized_label = normalize_dimension_label(dimension, clamped_score)

    score = AnalysisScore(
        dimension=dimension,
        score=clamped_score,
        label=normalized_label,
        confidence_score=clamped_confidence,
        rationale=str(rationale).strip() if isinstance(rationale, str) and rationale.strip() else None,
        evidence_ids=list(evidence_ids or []),
        rubric_notes=list(rubric_notes or ANALYSIS_RUBRIC_BY_DIMENSION[dimension].scoring_guidance),
    )
    return score, warnings


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------


def infer_evidence_type(chunk: Any) -> EvidenceType:
    """Infer a generic evidence type using simple structural cues only."""
    if chunk.section_kind == SectionKind.TABLE or any(char.isdigit() for char in chunk.text):
        return EvidenceType.NUMERIC_DETAIL
    if chunk.section_kind == SectionKind.TRANSCRIPT:
        return EvidenceType.DIRECT_QUOTE
    if chunk.section_kind == SectionKind.LIST:
        return EvidenceType.STATUS_UPDATE
    return EvidenceType.CONTEXTUAL_SUMMARY


def build_section_context_summaries(
    processed_document: ProcessedDocument,
    *,
    max_sections: int = 8,
) -> list[SectionContextSummary]:
    """Build compact section summaries for the shared analysis packet."""
    return [
        SectionContextSummary(
            section_id=section.section_id,
            title=section.title,
            level=section.level,
            section_kind=section.section_kind,
            char_count=section.char_count,
            word_count=section.word_count,
        )
        for section in processed_document.sections[:max_sections]
    ]


def build_chunk_evidence_bundles(
    processed_document: ProcessedDocument,
    chunk_retrieval_result: Any | None = None,
    *,
    max_bundles: int = 4,
) -> tuple[list[ChunkEvidenceBundle], list[AnalysisWarning]]:
    """Convert retrieved chunks into reusable evidence bundles for worker analysis."""
    from pipeline.processing.embeddings import make_text_excerpt

    warnings: list[AnalysisWarning] = []
    chunk_lookup = {chunk.chunk_id: chunk for chunk in processed_document.chunks}
    hit_lookup = {hit.chunk_id: hit for hit in (chunk_retrieval_result.hits if chunk_retrieval_result else [])}

    selected_chunk_ids = [hit.chunk_id for hit in (chunk_retrieval_result.hits if chunk_retrieval_result else [])]
    if not selected_chunk_ids:
        selected_chunk_ids = [chunk.chunk_id for chunk in processed_document.chunks[:max_bundles]]
        warnings.append(
            AnalysisWarning(
                issue_code="insufficient_chunk_evidence",
                message="No ranked chunk hits were available, so the analysis packet used the first processed chunks as fallback evidence.",
                metadata={"fallback_chunk_count": len(selected_chunk_ids)},
            )
        )

    bundles: list[ChunkEvidenceBundle] = []
    for rank, chunk_id in enumerate(selected_chunk_ids[:max_bundles], start=1):
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            warnings.append(
                AnalysisWarning(
                    issue_code="missing_chunk_reference",
                    message=f"Chunk id {chunk_id} was referenced for analysis but was not found in the processed document.",
                    source_chunk_id=chunk_id,
                )
            )
            continue

        hit = hit_lookup.get(chunk_id)
        expanded_context_text = hit.expanded_context_preview if hit is not None else None
        bundles.append(
            ChunkEvidenceBundle(
                bundle_id=f"{processed_document.document.document_id}::bundle_{rank:02d}",
                document_id=processed_document.document.document_id,
                chunk_id=chunk.chunk_id,
                section_id=chunk.parent_section_id,
                section_title=chunk.parent_section_title,
                retrieval_rank=hit.rank if hit is not None else rank,
                adjusted_score=hit.adjusted_score if hit is not None else None,
                similarity_score=hit.similarity_score if hit is not None else None,
                graph_bonus=hit.graph_bonus if hit is not None else None,
                evidence_type=infer_evidence_type(chunk),
                primary_text=chunk.text,
                expanded_context_text=expanded_context_text,
                local_context_summary=chunk.local_context_summary,
                notes=[note.message for note in chunk.notes],
            )
        )

    if len(bundles) < 2:
        warnings.append(
            AnalysisWarning(
                issue_code="limited_evidence_bundle_count",
                message="Fewer than two evidence bundles were available for shared worker analysis.",
                metadata={"bundle_count": len(bundles)},
            )
        )

    return bundles, warnings


def build_evidence_snippet_from_bundle(
    bundle: ChunkEvidenceBundle,
    *,
    source_url: str | None = None,
    rationale: str,
    supported_dimensions: Sequence[AnalysisDimension] | None = None,
    interpretation: EvidenceInterpretation | None = None,
    snippet_text: str | None = None,
) -> EvidenceSnippet:
    """Build a normalized evidence snippet from one chunk evidence bundle."""
    from pipeline.processing.embeddings import make_text_excerpt

    text = snippet_text or make_text_excerpt(bundle.primary_text, 240)
    return EvidenceSnippet(
        evidence_id=f"{bundle.bundle_id}::snippet",
        document_id=bundle.document_id,
        source_url=source_url,
        source_chunk_id=bundle.chunk_id,
        source_section_id=bundle.section_id,
        section_title=bundle.section_title,
        evidence_type=bundle.evidence_type,
        supported_dimensions=list(supported_dimensions or []),
        interpretation=interpretation,
        snippet_text=text,
        rationale=rationale,
        metadata={
            "bundle_id": bundle.bundle_id,
            "retrieval_rank": bundle.retrieval_rank,
            "adjusted_score": bundle.adjusted_score,
            "similarity_score": bundle.similarity_score,
            "graph_bonus": bundle.graph_bonus,
        },
    )


# ---------------------------------------------------------------------------
# Analysis input assembly
# ---------------------------------------------------------------------------

DEFAULT_GENERIC_ANALYSIS_INSTRUCTIONS = """
You are performing shared worker analysis for a biotech disclosure.
Use only the supplied document text, chunk evidence bundles, and provenance.

CRITICAL: You MUST return valid JSON with numeric scores for ALL 8 dimensions. Do NOT omit scores.

Your response must be a single JSON object with this exact structure:
{
  "summary": "2-3 sentence evidence-backed summary",
  "confidence_score": 0.75,
  "sentiment_score": 0.3,
  "sentiment_label": "positive",
  "scores": {
    "sentiment": {"score": 0.3, "label": "positive", "rationale": "why"},
    "uncertainty": {"score": 0.4, "rationale": "why"},
    "fogging": {"score": 0.2, "rationale": "why"},
    "hedging": {"score": 0.3, "rationale": "why"},
    "promotional_tone": {"score": 0.2, "rationale": "why"},
    "clarity": {"score": 0.7, "rationale": "why"},
    "materiality": {"score": 0.6, "rationale": "why"},
    "completeness": {"score": 0.5, "rationale": "why"}
  },
  "findings": [],
  "evidence_refs": [],
  "reasoning_notes": []
}

Score ranges:
- sentiment: -1.0 to 1.0 (negative to positive)
- All other dimensions: 0.0 to 1.0 (low to high)

Scoring guidance:
- Be decisive with sentiment. Favorable facts (approvals, positive trials, revenue growth) -> positive (0.2 to 0.8). Unfavorable facts (failures, safety issues, decline) -> negative (-0.2 to -0.8). Reserve near-zero only for purely procedural content.
- fogging: high (>0.6) when material points are obscured; low (<0.3) when disclosure is explicit
- hedging: high (>0.6) when claims are heavily qualified; low (<0.3) for direct factual reporting
- promotional_tone: high (>0.6) when claims outpace evidence; low (<0.3) when balanced
- clarity: high (>0.7) for coherent, specific explanations; low (<0.3) for fragmented content
- materiality: high (>0.7) for consequential facts; low (<0.3) for routine information
- completeness: high (>0.7) for balanced context with next steps; low (<0.3) for meaningful omissions

Every score field MUST contain a numeric value. Do not leave any score as null or omit it.
"""


def describe_worker_analysis_output_schema(rubric: AnalysisRubric = DEFAULT_ANALYSIS_RUBRIC) -> dict[str, Any]:
    """Describe the expected normalized worker-analysis output shape for later model clients."""
    return {
        "summary": "Short evidence-backed narrative summary.",
        "confidence_score": "Float in [0.0, 1.0].",
        "sentiment_score": "Float in [-1.0, 1.0]. REQUIRED numeric sentiment score.",
        "sentiment_label": "One of: positive, negative, neutral, mixed.",
        "scores": {
            definition.dimension.value: {
                "score": f"REQUIRED numeric float in [{definition.minimum_score}, {definition.maximum_score}]. Must not be null or omitted.",
                "label": "Required for sentiment; optional for other dimensions.",
                "confidence_score": "Optional float in [0.0, 1.0].",
                "rationale": definition.objective,
            }
            for definition in rubric.dimension_definitions
        },
        "findings": [
            {
                "dimension": "analysis dimension",
                "summary": "Finding summary tied to evidence.",
                "interpretation": "positive | negative | neutral | uncertainty | fogging | hedging | promotional | clarifying | material",
                "evidence_refs": ["bundle_id or chunk_id references"],
            }
        ],
        "evidence_refs": [
            {
                "chunk_id": "source chunk id",
                "bundle_id": "optional bundle id",
                "dimension": "supported dimension",
                "interpretation": "how the evidence should be read",
                "why_it_matters": "why the citation matters",
            }
        ],
        "reasoning_notes": ["Optional compact notes describing how the evidence was interpreted."],
    }


def build_worker_analysis_input(
    worker_input: WorkerInput,
    processed_document: ProcessedDocument,
    chunk_retrieval_result: Any | None = None,
    *,
    worker_name: str,
    analysis_instructions: str | None = None,
    rubric: AnalysisRubric = DEFAULT_ANALYSIS_RUBRIC,
    max_document_chars: int = 3500,
    max_evidence_bundles: int = 4,
) -> tuple[WorkerAnalysisInput, list[AnalysisWarning]]:
    """Assemble the reusable shared analysis packet for one worker invocation."""
    from pipeline.processing.embeddings import make_text_excerpt

    metadata = processed_document.document.metadata or DocumentMetadata(
        document_id=processed_document.document.document_id,
        ticker=processed_document.document.ticker,
        document_type=processed_document.document.document_type,
        title=processed_document.document.title,
        source_name=processed_document.document.source_name,
        source_url=processed_document.document.source_url,
        is_mock_data=processed_document.document.is_mock_data,
    )
    document_text = processed_document.cleaned_text[:max_document_chars].strip()
    if not document_text:
        document_text = processed_document.document.raw_text[:max_document_chars].strip()

    evidence_bundles, bundle_warnings = build_chunk_evidence_bundles(
        processed_document,
        chunk_retrieval_result,
        max_bundles=max_evidence_bundles,
    )
    if len(processed_document.cleaned_text) > max_document_chars:
        bundle_warnings.append(
            AnalysisWarning(
                issue_code="document_text_truncated",
                message="The document text in the shared analysis packet was truncated for notebook-safe prompt size.",
                metadata={"max_document_chars": max_document_chars, "original_chars": len(processed_document.cleaned_text)},
            )
        )

    analysis_input = WorkerAnalysisInput(
        run_id=worker_input.run_id,
        ticker=worker_input.ticker,
        worker_name=worker_name,
        document_type=worker_input.document_type,
        document_metadata=metadata,
        document_title=processed_document.document.title,
        document_text=document_text,
        document_text_excerpt=make_text_excerpt(document_text, 320),
        section_context=build_section_context_summaries(processed_document),
        top_chunk_bundles=evidence_bundles,
        provenance=list(processed_document.document.provenance),
        processing_notes=list(processed_document.processing_notes),
        analysis_instructions=(analysis_instructions or DEFAULT_GENERIC_ANALYSIS_INSTRUCTIONS).strip(),
        expected_output_schema=describe_worker_analysis_output_schema(rubric),
        retrieval_query=chunk_retrieval_result.query_text if chunk_retrieval_result is not None else None,
        rubric_id=rubric.rubric_id,
        rubric_version=rubric.version,
        is_mock_data=processed_document.document.is_mock_data,
        metadata={
            "chunk_count": processed_document.chunk_count,
            "section_count": processed_document.section_count,
            "used_fallback_chunking": processed_document.used_fallback_chunking,
        },
    )
    return analysis_input, bundle_warnings


# ---------------------------------------------------------------------------
# Analysis query
# ---------------------------------------------------------------------------


def build_generic_analysis_query(processed_document: ProcessedDocument) -> str:
    """Build a document-agnostic retrieval query for shared worker analysis."""
    title = processed_document.document.title or processed_document.document.document_type.value.replace("_", " ")
    dimensions = ", ".join(
        [
            AnalysisDimension.SENTIMENT.value,
            AnalysisDimension.UNCERTAINTY.value,
            AnalysisDimension.FOGGING.value,
            AnalysisDimension.HEDGING.value,
            AnalysisDimension.PROMOTIONAL_TONE.value,
            AnalysisDimension.CLARITY.value,
            AnalysisDimension.MATERIALITY.value,
            AnalysisDimension.COMPLETENESS.value,
        ]
    )
    return f"{title} evidence for {dimensions}"


# ---------------------------------------------------------------------------
# Warning / issue factories
# ---------------------------------------------------------------------------


def make_analysis_warning(
    issue_code: str,
    message: str,
    *,
    dimension: AnalysisDimension | None = None,
    source_chunk_id: str | None = None,
    field_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AnalysisWarning:
    """Build a normalized analysis warning."""
    return AnalysisWarning(
        issue_code=issue_code,
        message=message,
        dimension=dimension,
        source_chunk_id=source_chunk_id,
        field_name=field_name,
        metadata=metadata or {},
    )


def make_analysis_issue(
    issue_code: str,
    message: str,
    *,
    severity: AnalysisIssueSeverity = AnalysisIssueSeverity.ERROR,
    dimension: AnalysisDimension | None = None,
    source_chunk_id: str | None = None,
    field_name: str | None = None,
    recoverable: bool = True,
    metadata: dict[str, Any] | None = None,
) -> AnalysisIssue:
    """Build a normalized analysis issue."""
    return AnalysisIssue(
        issue_code=issue_code,
        message=message,
        severity=severity,
        dimension=dimension,
        source_chunk_id=source_chunk_id,
        field_name=field_name,
        recoverable=recoverable,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Parsing raw analysis output
# ---------------------------------------------------------------------------


def coerce_analysis_payload(raw_output: Any) -> tuple[dict[str, Any], list[AnalysisWarning]]:
    """Coerce raw client output into a dictionary when possible."""
    warnings: list[AnalysisWarning] = []
    if raw_output is None:
        warnings.append(make_analysis_warning("empty_analysis_output", "Analysis client returned no output."))
        return {}, warnings
    if isinstance(raw_output, dict):
        return raw_output, warnings
    if isinstance(raw_output, str):
        stripped = raw_output.strip()
        if not stripped:
            warnings.append(make_analysis_warning("empty_analysis_output", "Analysis client returned empty text."))
            return {}, warnings
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            warnings.append(
                make_analysis_warning(
                    "analysis_output_not_json",
                    "Analysis output was text instead of JSON; the parser will continue with an empty payload.",
                )
            )
            return {}, warnings
        if isinstance(payload, dict):
            return payload, warnings
        warnings.append(
            make_analysis_warning(
                "analysis_output_wrong_shape",
                "Analysis JSON payload was not a dictionary.",
            )
        )
        return {}, warnings
    warnings.append(
        make_analysis_warning(
            "analysis_output_unsupported_type",
            f"Analysis output type {type(raw_output).__name__} is unsupported by the shared parser.",
        )
    )
    return {}, warnings


def normalize_analysis_scores(
    raw_payload: Mapping[str, Any] | dict[str, Any],
    rubric: AnalysisRubric = DEFAULT_ANALYSIS_RUBRIC,
) -> tuple[list[AnalysisScore], list[AnalysisWarning]]:
    """Normalize raw score content from a worker-analysis payload."""
    warnings: list[AnalysisWarning] = []
    normalized_scores: list[AnalysisScore] = []

    raw_scores = raw_payload.get("scores") if raw_payload else None
    score_lookup: dict[str, Any] = {}
    if isinstance(raw_scores, Mapping):
        score_lookup = {str(key): value for key, value in raw_scores.items()}
    elif isinstance(raw_scores, list):
        for item in raw_scores:
            if isinstance(item, Mapping) and item.get("dimension"):
                score_lookup[str(item["dimension"])] = item
    elif raw_scores is not None:
        warnings.append(
            make_analysis_warning(
                "scores_wrong_shape",
                "Raw score payload was neither a dictionary nor a list of dimension entries.",
                field_name="scores",
            )
        )

    for definition in rubric.dimension_definitions:
        raw_entry = score_lookup.get(definition.dimension.value)
        raw_score = None
        raw_label = None
        raw_confidence = raw_payload.get("confidence_score") if raw_payload else None
        rationale = None
        evidence_ids: list[str] = []

        if isinstance(raw_entry, Mapping):
            raw_score = raw_entry.get("score")
            raw_label = raw_entry.get("label")
            raw_confidence = raw_entry.get("confidence_score", raw_confidence)
            rationale = raw_entry.get("rationale")
            if isinstance(raw_entry.get("evidence_ids"), list):
                evidence_ids = [str(item) for item in raw_entry["evidence_ids"] if str(item).strip()]
        else:
            if definition.dimension == AnalysisDimension.SENTIMENT:
                raw_score = raw_payload.get("sentiment_score")
                raw_label = raw_payload.get("sentiment_label")
            else:
                raw_score = raw_payload.get(f"{definition.dimension.value}_score")

        score, score_warnings = build_standardized_analysis_score(
            dimension=definition.dimension,
            raw_score=raw_score,
            raw_label=raw_label,
            raw_confidence=raw_confidence,
            rationale=rationale,
            evidence_ids=evidence_ids,
            rubric_notes=definition.scoring_guidance,
        )
        if score.score is None:
            warnings.append(
                make_analysis_warning(
                    "missing_dimension_score",
                    f"No explicit score was provided for {definition.dimension.value}.",
                    dimension=definition.dimension,
                    field_name="scores",
                )
            )
        warnings.extend(score_warnings)
        normalized_scores.append(score)

    return normalized_scores, warnings


def map_raw_evidence_references(
    raw_references: Any,
    analysis_input: WorkerAnalysisInput,
) -> tuple[list[EvidenceSnippet], list[AnalysisWarning]]:
    """Map raw evidence references back to structured evidence snippets."""
    warnings: list[AnalysisWarning] = []
    evidence: list[EvidenceSnippet] = []

    bundle_lookup = {bundle.bundle_id: bundle for bundle in analysis_input.top_chunk_bundles}
    chunk_lookup = {bundle.chunk_id: bundle for bundle in analysis_input.top_chunk_bundles}

    if not isinstance(raw_references, list):
        return evidence, warnings

    for index, reference in enumerate(raw_references, start=1):
        if not isinstance(reference, Mapping):
            warnings.append(
                make_analysis_warning(
                    "invalid_evidence_reference",
                    "One evidence reference was not a dictionary and was skipped.",
                    field_name="evidence_refs",
                    metadata={"index": index},
                )
            )
            continue

        bundle = None
        bundle_id = reference.get("bundle_id")
        chunk_id = reference.get("chunk_id")
        if isinstance(bundle_id, str):
            bundle = bundle_lookup.get(bundle_id)
        if bundle is None and isinstance(chunk_id, str):
            bundle = chunk_lookup.get(chunk_id)
        if bundle is None:
            warnings.append(
                make_analysis_warning(
                    "missing_evidence_reference",
                    "Evidence reference did not match any prepared evidence bundle.",
                    field_name="evidence_refs",
                    source_chunk_id=str(chunk_id) if chunk_id is not None else None,
                )
            )
            continue

        dimension_raw = reference.get("dimension")
        interpretation_raw = reference.get("interpretation")
        supported_dimensions: list[AnalysisDimension] = []
        interpretation = None
        if isinstance(dimension_raw, str):
            try:
                supported_dimensions = [AnalysisDimension(dimension_raw)]
            except ValueError:
                warnings.append(
                    make_analysis_warning(
                        "unknown_evidence_dimension",
                        f"Evidence dimension {dimension_raw!r} is not recognized.",
                        field_name="dimension",
                        source_chunk_id=bundle.chunk_id,
                    )
                )
        if isinstance(interpretation_raw, str):
            try:
                interpretation = EvidenceInterpretation(interpretation_raw)
            except ValueError:
                warnings.append(
                    make_analysis_warning(
                        "unknown_evidence_interpretation",
                        f"Evidence interpretation {interpretation_raw!r} is not recognized.",
                        field_name="interpretation",
                        source_chunk_id=bundle.chunk_id,
                    )
                )

        why_it_matters = reference.get("why_it_matters")
        if not isinstance(why_it_matters, str) or not why_it_matters.strip():
            why_it_matters = "Evidence reference returned by the shared worker-analysis client."

        evidence.append(
            build_evidence_snippet_from_bundle(
                bundle,
                source_url=analysis_input.document_metadata.source_url,
                rationale=why_it_matters.strip(),
                supported_dimensions=supported_dimensions,
                interpretation=interpretation,
                snippet_text=str(reference.get("evidence_text")).strip() if reference.get("evidence_text") else None,
            )
        )

    return evidence, warnings


def extract_structured_findings(
    raw_findings: Any,
    *,
    analysis_input: WorkerAnalysisInput,
    normalized_scores: Sequence[AnalysisScore],
    evidence: Sequence[EvidenceSnippet],
) -> tuple[list[AnalysisFinding], list[AnalysisWarning]]:
    """Normalize raw findings or build fallback findings from available evidence."""
    warnings: list[AnalysisWarning] = []
    findings: list[AnalysisFinding] = []
    evidence_lookup = {item.evidence_id: item for item in evidence if item.evidence_id}
    score_lookup = {score.dimension: score for score in normalized_scores}

    if isinstance(raw_findings, list):
        for index, raw_finding in enumerate(raw_findings, start=1):
            if not isinstance(raw_finding, Mapping):
                warnings.append(
                    make_analysis_warning(
                        "invalid_finding_entry",
                        "One finding entry was not a dictionary and was skipped.",
                        field_name="findings",
                        metadata={"index": index},
                    )
                )
                continue

            dimension_raw = raw_finding.get("dimension")
            if not isinstance(dimension_raw, str):
                warnings.append(
                    make_analysis_warning(
                        "finding_dimension_missing",
                        "A finding was missing its analysis dimension and was skipped.",
                        field_name="findings",
                        metadata={"index": index},
                    )
                )
                continue
            try:
                dimension = AnalysisDimension(dimension_raw)
            except ValueError:
                warnings.append(
                    make_analysis_warning(
                        "finding_dimension_unknown",
                        f"Finding dimension {dimension_raw!r} is not recognized.",
                        field_name="findings",
                        metadata={"index": index},
                    )
                )
                continue

            interpretation = None
            interpretation_raw = raw_finding.get("interpretation")
            if isinstance(interpretation_raw, str):
                try:
                    interpretation = EvidenceInterpretation(interpretation_raw)
                except ValueError:
                    warnings.append(
                        make_analysis_warning(
                            "finding_interpretation_unknown",
                            f"Finding interpretation {interpretation_raw!r} is not recognized.",
                            field_name="findings",
                            metadata={"index": index},
                        )
                    )

            evidence_refs = raw_finding.get("evidence_refs")
            if not isinstance(evidence_refs, list):
                evidence_refs = []
            evidence_ids = [str(item) for item in evidence_refs if str(item).strip() and str(item) in evidence_lookup]
            source_chunk_ids = [
                evidence_lookup[evidence_id].source_chunk_id
                for evidence_id in evidence_ids
                if evidence_lookup[evidence_id].source_chunk_id is not None
            ]
            summary = str(raw_finding.get("summary") or "").strip()
            if not summary:
                warnings.append(
                    make_analysis_warning(
                        "finding_summary_missing",
                        "A finding was missing summary text and was skipped.",
                        field_name="findings",
                        metadata={"index": index},
                    )
                )
                continue

            findings.append(
                AnalysisFinding(
                    finding_id=f"{analysis_input.run_id}::{analysis_input.worker_name}::finding_{len(findings) + 1:02d}",
                    dimension=dimension,
                    summary=summary,
                    interpretation=interpretation,
                    evidence_ids=evidence_ids,
                    source_chunk_ids=[chunk_id for chunk_id in source_chunk_ids if chunk_id],
                    score=score_lookup.get(dimension).score if dimension in score_lookup else None,
                    rationale=str(raw_finding.get("rationale")).strip() if raw_finding.get("rationale") else None,
                    is_material=bool(raw_finding.get("is_material", False)),
                )
            )

    if findings:
        return findings, warnings

    fallback_evidence = list(evidence)[:3]
    if not fallback_evidence:
        warnings.append(
            make_analysis_warning(
                "no_findings_available",
                "No normalized findings could be built because both findings and evidence were empty.",
            )
        )
        return findings, warnings

    for index, evidence_item in enumerate(fallback_evidence, start=1):
        dimension = (
            evidence_item.supported_dimensions[0]
            if evidence_item.supported_dimensions
            else AnalysisDimension.MATERIALITY
        )
        findings.append(
            AnalysisFinding(
                finding_id=f"{analysis_input.run_id}::{analysis_input.worker_name}::fallback_finding_{index:02d}",
                dimension=dimension,
                summary=evidence_item.rationale,
                interpretation=evidence_item.interpretation,
                evidence_ids=[evidence_item.evidence_id] if evidence_item.evidence_id else [],
                source_chunk_ids=[evidence_item.source_chunk_id] if evidence_item.source_chunk_id else [],
                score=score_lookup.get(dimension).score if dimension in score_lookup else None,
                rationale="Fallback finding built from structured evidence because raw findings were missing.",
                is_material=dimension == AnalysisDimension.MATERIALITY,
            )
        )
    warnings.append(
        make_analysis_warning(
            "findings_built_from_evidence",
            "Raw findings were missing, so fallback findings were built from the normalized evidence snippets.",
        )
    )
    return findings, warnings


def parse_worker_analysis_output(
    raw_output: Any,
    analysis_input: WorkerAnalysisInput,
    *,
    client_name: str,
    model_name: str,
    is_mock: bool = False,
    client_warnings: Sequence[AnalysisWarning] | None = None,
    prompt_context: dict[str, Any] | None = None,
) -> WorkerAnalysisOutput:
    """Defensively parse raw worker analysis output into a normalized internal result."""
    from pipeline.processing.embeddings import make_text_excerpt

    payload, parse_warnings = coerce_analysis_payload(raw_output)
    warnings: list[AnalysisWarning] = list(client_warnings or []) + parse_warnings
    issues: list[AnalysisIssue] = []

    confidence_score = coerce_float(payload.get("confidence_score")) if payload else None
    if confidence_score is not None:
        confidence_score, confidence_warning = clamp_score_value(
            confidence_score,
            minimum=0.0,
            maximum=1.0,
            warning_code="confidence_out_of_range",
            dimension=AnalysisDimension.CLARITY,
        )
        if confidence_warning is not None:
            warnings.append(confidence_warning)

    normalized_scores, score_warnings = normalize_analysis_scores(payload, DEFAULT_ANALYSIS_RUBRIC)
    warnings.extend(score_warnings)

    evidence, evidence_warnings = map_raw_evidence_references(payload.get("evidence_refs"), analysis_input)
    warnings.extend(evidence_warnings)

    findings, finding_warnings = extract_structured_findings(
        payload.get("findings"),
        analysis_input=analysis_input,
        normalized_scores=normalized_scores,
        evidence=evidence,
    )
    warnings.extend(finding_warnings)

    summary = payload.get("summary")
    if isinstance(summary, str):
        summary = summary.strip() or None
    else:
        summary = None
    if summary is None and findings:
        summary = findings[0].summary
        warnings.append(
            make_analysis_warning(
                "summary_missing",
                "Analysis summary was missing, so the first finding summary was used as a fallback.",
                field_name="summary",
            )
        )
    elif summary is None:
        warnings.append(make_analysis_warning("summary_missing", "Analysis summary was missing.", field_name="summary"))

    reasoning_notes = payload.get("reasoning_notes")
    if not isinstance(reasoning_notes, list):
        reasoning_notes = []
    reasoning_notes = [str(note).strip() for note in reasoning_notes if str(note).strip()]

    unresolved_items = payload.get("unresolved_items")
    if not isinstance(unresolved_items, list):
        unresolved_items = []
    unresolved_items = [str(item).strip() for item in unresolved_items if str(item).strip()]

    if not reasoning_notes:
        warnings.append(make_analysis_warning("empty_reasoning_notes", "Analysis output did not include reasoning notes."))

    if not evidence:
        warnings.append(
            make_analysis_warning(
                "missing_evidence_references",
                "Analysis output did not contain valid evidence references.",
            )
        )

    if payload and not normalized_scores:
        issues.append(
            make_analysis_issue(
                "score_normalization_failed",
                "No normalized scores could be produced from the raw analysis payload.",
            )
        )

    prompt_context_preview = None
    if prompt_context is not None:
        prompt_context_preview = make_text_excerpt(json.dumps(prompt_context, default=str), 320)

    reasoning_trace = WorkerReasoningTrace(
        analysis_client_name=client_name,
        model_name=model_name,
        used_mock_client=is_mock,
        selected_chunk_ids=[bundle.chunk_id for bundle in analysis_input.top_chunk_bundles],
        selected_section_ids=[bundle.section_id for bundle in analysis_input.top_chunk_bundles if bundle.section_id],
        reasoning_notes=reasoning_notes,
        unresolved_items=unresolved_items,
        prompt_context_preview=prompt_context_preview,
        raw_response_present=bool(payload),
    )

    status = ProcessingStatus.SUCCESS
    if issues:
        status = ProcessingStatus.ANALYSIS_FAILED
    elif warnings:
        status = ProcessingStatus.PARTIAL
    if not payload:
        status = ProcessingStatus.ANALYSIS_FAILED

    return WorkerAnalysisOutput(
        run_id=analysis_input.run_id,
        ticker=analysis_input.ticker,
        worker_name=analysis_input.worker_name,
        document_type=analysis_input.document_type,
        status=status,
        summary=summary,
        confidence_score=confidence_score,
        scores=normalized_scores,
        findings=findings,
        evidence=evidence,
        warnings=warnings,
        issues=issues,
        reasoning_trace=reasoning_trace,
        raw_output=payload,
        is_mock_analysis=is_mock,
    )


# ---------------------------------------------------------------------------
# Assessment builders and output helpers
# ---------------------------------------------------------------------------


def deduplicate_analysis_items(items: Sequence[Any], key_builder: Callable[[Any], tuple[Any, ...]]) -> list[Any]:
    """Deduplicate warnings or issues while preserving order."""
    seen: set[tuple[Any, ...]] = set()
    deduplicated: list[Any] = []
    for item in items:
        key = key_builder(item)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
    return deduplicated


def collect_analysis_warnings(analysis_output: WorkerAnalysisOutput) -> list[AnalysisWarning]:
    """Add secondary warnings after parsing and score normalization."""
    warnings = list(analysis_output.warnings)
    if analysis_output.confidence_score is not None and analysis_output.confidence_score < 0.4:
        warnings.append(
            make_analysis_warning(
                "low_confidence_output",
                "Shared worker analysis returned low confidence and should be treated as provisional.",
            )
        )

    sentiment_score = next(
        (score for score in analysis_output.scores if score.dimension == AnalysisDimension.SENTIMENT),
        None,
    )
    if sentiment_score is not None and sentiment_score.label == SentimentLabel.MIXED.value and sentiment_score.score is not None:
        if abs(sentiment_score.score) < 0.2:
            warnings.append(
                make_analysis_warning(
                    "ambiguous_sentiment_label",
                    "Sentiment was labeled mixed while the normalized sentiment score remained close to zero.",
                    dimension=AnalysisDimension.SENTIMENT,
                )
            )

    if analysis_output.reasoning_trace is None or not analysis_output.reasoning_trace.reasoning_notes:
        warnings.append(
            make_analysis_warning(
                "empty_reasoning_notes",
                "No reasoning notes were captured in the shared worker-analysis trace.",
            )
        )

    if len(analysis_output.evidence) < 2:
        warnings.append(
            make_analysis_warning(
                "insufficient_chunk_evidence",
                "Shared worker analysis finished with fewer than two structured evidence snippets.",
            )
        )

    return deduplicate_analysis_items(
        warnings,
        lambda item: (item.issue_code, item.message, item.dimension, item.source_chunk_id, item.field_name),
    )


def analysis_scores_to_lookup(scores: Sequence[AnalysisScore]) -> dict[AnalysisDimension, AnalysisScore]:
    """Build a dimension lookup for normalized scores."""
    return {score.dimension: score for score in scores}


def build_sentiment_assessment_from_scores(scores: Sequence[AnalysisScore]) -> SentimentAssessment | None:
    """Convert normalized analysis scores into the legacy sentiment assessment shape."""
    sentiment_score = analysis_scores_to_lookup(scores).get(AnalysisDimension.SENTIMENT)
    if sentiment_score is None:
        return None
    sentiment_label, _ = normalize_sentiment_label(sentiment_score.label, sentiment_score.score)
    return SentimentAssessment(
        label=sentiment_label,
        score=sentiment_score.score,
        confidence=sentiment_score.confidence_score,
        rationale=sentiment_score.rationale,
    )


def build_tone_assessment_from_scores(scores: Sequence[AnalysisScore]) -> ToneAssessment | None:
    """Convert normalized analysis scores into the legacy tone assessment shape."""
    lookup = analysis_scores_to_lookup(scores)
    if not any(dimension in lookup for dimension in (AnalysisDimension.FOGGING, AnalysisDimension.HEDGING, AnalysisDimension.PROMOTIONAL_TONE)):
        return None
    confidence_candidates = [
        lookup[dimension].confidence_score
        for dimension in (AnalysisDimension.FOGGING, AnalysisDimension.HEDGING, AnalysisDimension.PROMOTIONAL_TONE)
        if dimension in lookup and lookup[dimension].confidence_score is not None
    ]
    return ToneAssessment(
        fogging_score=lookup.get(AnalysisDimension.FOGGING).score if AnalysisDimension.FOGGING in lookup else None,
        hedging_score=lookup.get(AnalysisDimension.HEDGING).score if AnalysisDimension.HEDGING in lookup else None,
        promotional_score=(
            lookup.get(AnalysisDimension.PROMOTIONAL_TONE).score
            if AnalysisDimension.PROMOTIONAL_TONE in lookup
            else None
        ),
        confidence=(sum(confidence_candidates) / len(confidence_candidates)) if confidence_candidates else None,
        rationale="Shared worker-analysis tone scores normalized from the common rubric.",
    )
