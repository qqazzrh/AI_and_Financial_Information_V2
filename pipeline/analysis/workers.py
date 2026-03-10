"""Disclosure-specific analysis workers, prompt assembly, and output assembly."""
from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date, datetime
from textwrap import dedent
from typing import Any, ClassVar

from pipeline.config import logger
from pipeline.enums import (
    AnalysisDimension,
    AnalysisIssueSeverity,
    DocumentType,
    ProcessingStatus,
)
from pipeline.models import (
    AnalysisClientRequest,
    AnalysisIssue,
    AnalysisRubric,
    AnalysisScore,
    AnalysisWarning,
    BaseWorker,
    ChunkRetrievalResult,
    DocumentMetadata,
    EvidenceSnippet,
    GraphDocumentIndex,
    PipelineConfig,
    PipelineError,
    ProcessedDocument,
    ProvenanceRecord,
    SelectedDocument,
    WorkerAnalysisInput,
    WorkerAnalysisOutput,
    WorkerInput,
    WorkerOutput,
    WorkerReasoningTrace,
)
from pipeline.analysis.client import BaseAnalysisClient, build_analysis_client
from pipeline.analysis.rubrics import (
    DEFAULT_ANALYSIS_RUBRIC,
    build_generic_analysis_query,
    build_sentiment_assessment_from_scores,
    build_tone_assessment_from_scores,
    build_worker_analysis_input,
    collect_analysis_warnings,
    deduplicate_analysis_items,
    make_analysis_issue,
    make_analysis_warning,
    normalize_sentiment_label,
    parse_worker_analysis_output,
)

__all__ = [
    "CLINICAL_TRIAL_UPDATE_MASTER_PROMPT",
    "FDA_REVIEW_MASTER_PROMPT",
    "FINANCING_DILUTION_MASTER_PROMPT",
    "INVESTOR_COMMUNICATION_MASTER_PROMPT",
    "MATERIAL_EVENT_MASTER_PROMPT",
    "WORKER_PROMPT_REQUIRED_SECTIONS",
    "BaseAnalysisWorker",
    "ClinicalTrialUpdateWorker",
    "FDAReviewWorker",
    "FinancingDilutionWorker",
    "InvestorCommunicationWorker",
    "MaterialEventWorker",
    "analysis_issue_to_pipeline_error",
    "analysis_warning_to_pipeline_error",
    "assemble_worker_output",
    "build_analysis_prompt_context",
    "build_shared_worker_instruction",
    "build_worker_prompt",
    "build_worker_specific_instruction",
    "render_analysis_context_for_prompt",
    "render_input_metadata_for_prompt",
    "render_output_schema_for_prompt",
    "render_prompt_bullet_list",
    "retrieve_analysis_context",
    "run_generic_analysis",
    "validate_prompt_analysis_context",
    "validate_worker_prompt_sections",
]


# ---------------------------------------------------------------------------
# Prompt rendering helpers
# ---------------------------------------------------------------------------

WORKER_PROMPT_REQUIRED_SECTIONS = (
    "## Shared Worker Base Instruction",
    "## Worker-Specific Analysis Instruction",
    "## Input Metadata",
    "## Evidence and Context",
    "## Output Schema",
)


def render_prompt_bullet_list(items: Sequence[str]) -> str:
    """Render one bullet list for prompt text blocks."""
    cleaned_items = [str(item).strip() for item in items if str(item).strip()]
    return "\n".join(f"- {item}" for item in cleaned_items) if cleaned_items else "- none"


def format_optional_datetime(value: datetime | None) -> str:
    """Format optional datetimes consistently for prompt metadata blocks."""
    return value.isoformat() if isinstance(value, datetime) else "n/a"


def format_optional_date(value: date | None) -> str:
    """Format optional dates consistently for prompt metadata blocks."""
    return value.isoformat() if isinstance(value, date) else "n/a"


def build_shared_worker_instruction(rubric: AnalysisRubric = DEFAULT_ANALYSIS_RUBRIC) -> str:
    """Build the common instruction block that every disclosure worker shares."""
    dimension_lines = [
        f"{definition.dimension.value}: {definition.objective}"
        for definition in rubric.dimension_definitions
    ]
    return dedent(
        f"""
        ## Shared Worker Base Instruction

        You are one disclosure-specific worker inside a biotech disclosure analysis system.
        Analyze exactly one document of the assigned type.
        Use only the supplied document text, section context, chunk evidence bundles, and provenance.
        Keep every conclusion proportional to the cited evidence.
        Separate disclosed facts from management framing and separate directional implication from tone.
        When evidence shows active development, regulatory progress, trial advancement, or positive operational metrics, assign a clearly positive sentiment score (0.3 to 0.7).
        When evidence shows setbacks, failures, delays, safety concerns, or declining metrics, assign a clearly negative sentiment score (-0.3 to -0.7).
        Reserve neutral (near 0) only for purely administrative or procedural content with zero investment implication.
        Do not rely on canned phrase matching, undocumented metadata, or outside assumptions.
        Return structured output only in the requested schema.

        ### Shared Rubric Priorities
        {render_prompt_bullet_list(rubric.core_principles)}

        ### Shared Dimensions
        {render_prompt_bullet_list(dimension_lines)}
        """
    ).strip()


def build_worker_specific_instruction(
    *,
    worker_label: str,
    worker_job: str,
    document_native_focus: str,
    what_to_look_for: Sequence[str],
    fogging_guidance: Sequence[str],
    signal_guidance: Sequence[str],
    output_requirements: Sequence[str] | None = None,
) -> str:
    """Build the worker-specific instruction block for one disclosure type."""
    resolved_output_requirements = list(
        output_requirements
        or [
            "Tie every material conclusion to cited evidence bundle ids or chunk ids.",
            "Call out missing expected detail explicitly rather than filling gaps with outside knowledge.",
            "Keep positive, negative, mixed, or neutral signal grounded in disclosed facts, not wording polish alone.",
            "If the evidence is thin or ambiguous, reflect that explicitly in uncertainty, caveats, and confidence.",
        ]
    )
    return dedent(
        f"""
        ## Worker-Specific Analysis Instruction

        ### Worker
        {worker_label}

        ### Your Job
        {worker_job}

        ### Document-Native Focus
        {document_native_focus}

        ### What To Look For
        {render_prompt_bullet_list(what_to_look_for)}

        ### Fogging and Evasiveness
        {render_prompt_bullet_list(fogging_guidance)}

        ### Positive vs Negative Signal
        {render_prompt_bullet_list(signal_guidance)}

        ### Output Discipline
        {render_prompt_bullet_list(resolved_output_requirements)}
        """
    ).strip()


def render_input_metadata_for_prompt(analysis_input: WorkerAnalysisInput) -> str:
    """Render one compact metadata block for a disclosure-specific worker prompt."""
    metadata = analysis_input.document_metadata
    metadata_lines = [
        f"- run_id: {analysis_input.run_id}",
        f"- worker_name: {analysis_input.worker_name}",
        f"- document_type: {analysis_input.document_type.value}",
        f"- ticker: {analysis_input.ticker}",
        f"- company_name: {metadata.company_name or 'n/a'}",
        f"- document_id: {metadata.document_id}",
        f"- title: {analysis_input.document_title or metadata.title or 'n/a'}",
        f"- source_name: {metadata.source_name or 'n/a'}",
        f"- source_family: {metadata.source_family.value}",
        f"- source_url: {metadata.source_url or 'n/a'}",
        f"- published_at: {format_optional_datetime(metadata.published_at)}",
        f"- updated_at: {format_optional_datetime(metadata.updated_at)}",
        f"- event_date: {format_optional_date(metadata.event_date)}",
        f"- version_label: {metadata.version_label or 'n/a'}",
        f"- section_count: {analysis_input.metadata.get('section_count', 'n/a')}",
        f"- chunk_count: {analysis_input.metadata.get('chunk_count', 'n/a')}",
        f"- retrieval_query: {analysis_input.retrieval_query or 'n/a'}",
        f"- mock_data: {analysis_input.is_mock_data}",
    ]
    processing_lines = [
        f"- {note.code}: {note.message}"
        for note in analysis_input.processing_notes[:5]
    ]
    if not processing_lines:
        processing_lines = ["- none"]
    return "\n".join(metadata_lines + ["", "### Processing Notes", *processing_lines])


def render_analysis_context_for_prompt(analysis_input: WorkerAnalysisInput) -> str:
    """Render the evidence and context block used by every specialized worker prompt."""
    from pipeline.processing.embeddings import make_text_excerpt

    section_lines = [
        (
            f"- {section.section_id} | title={section.title!r} | kind={section.section_kind.value} "
            f"| level={section.level} | words={section.word_count}"
        )
        for section in analysis_input.section_context
    ]
    if not section_lines:
        section_lines = ["- none"]

    evidence_blocks: list[str] = []
    for bundle in analysis_input.top_chunk_bundles:
        bundle_lines = [
            f"- bundle_id: {bundle.bundle_id}",
            f"  chunk_id: {bundle.chunk_id}",
            f"  section_title: {bundle.section_title or 'n/a'}",
            f"  retrieval_rank: {bundle.retrieval_rank if bundle.retrieval_rank is not None else 'n/a'}",
            f"  evidence_type: {bundle.evidence_type.value}",
            f"  adjusted_score: {bundle.adjusted_score if bundle.adjusted_score is not None else 'n/a'}",
            f"  primary_text: {make_text_excerpt(bundle.primary_text, 280)}",
        ]
        if bundle.expanded_context_text:
            bundle_lines.append(f"  expanded_context: {make_text_excerpt(bundle.expanded_context_text, 220)}")
        if bundle.local_context_summary:
            bundle_lines.append(f"  local_context_summary: {bundle.local_context_summary}")
        if bundle.notes:
            bundle_lines.append(f"  notes: {' | '.join(bundle.notes[:3])}")
        evidence_blocks.append("\n".join(bundle_lines))
    if not evidence_blocks:
        evidence_blocks = ["- none"]

    provenance_lines = [
        (
            f"- stage={record.stage} | adapter={record.adapter_name} | "
            f"source={record.source_name or 'n/a'} | note={record.note or 'n/a'}"
        )
        for record in analysis_input.provenance[:5]
    ]
    if not provenance_lines:
        provenance_lines = ["- none"]

    return "\n".join(
        [
            "### Document Text",
            analysis_input.document_text,
            "",
            "### Section Context",
            *section_lines,
            "",
            "### Evidence Bundles",
            *evidence_blocks,
            "",
            "### Provenance",
            *provenance_lines,
        ]
    )


def render_output_schema_for_prompt(expected_output_schema: dict[str, Any]) -> str:
    """Render the structured output contract for prompt assembly."""
    return "\n".join(
        [
            "Return one JSON object only that matches this schema exactly.",
            "```json",
            json.dumps(expected_output_schema, indent=2, default=str),
            "```",
        ]
    )


def build_worker_prompt(
    *,
    shared_instruction: str,
    worker_specific_instruction: str,
    analysis_input: WorkerAnalysisInput,
) -> str:
    """Assemble the full prompt text passed to one specialized worker."""
    return "\n\n".join(
        [
            shared_instruction.strip(),
            worker_specific_instruction.strip(),
            "## Input Metadata\n\n" + render_input_metadata_for_prompt(analysis_input),
            "## Evidence and Context\n\n" + render_analysis_context_for_prompt(analysis_input),
            "## Output Schema\n\n" + render_output_schema_for_prompt(analysis_input.expected_output_schema),
        ]
    ).strip()


def validate_worker_prompt_sections(prompt_text: str) -> list[AnalysisWarning]:
    """Warn when an assembled worker prompt is missing required structural blocks."""
    missing_sections = [section for section in WORKER_PROMPT_REQUIRED_SECTIONS if section not in prompt_text]
    if not missing_sections:
        return []
    return [
        make_analysis_warning(
            "worker_prompt_missing_required_sections",
            "Assembled worker prompt is missing one or more required sections.",
            field_name="prompt_text",
            metadata={"missing_sections": missing_sections},
        )
    ]


def validate_prompt_analysis_context(analysis_input: WorkerAnalysisInput) -> list[AnalysisWarning]:
    """Warn when the assembled prompt context is too thin for confident analysis."""
    warnings: list[AnalysisWarning] = []
    if len(analysis_input.top_chunk_bundles) < 2:
        warnings.append(
            make_analysis_warning(
                "worker_evidence_bundle_too_thin",
                "Disclosure-specific analysis is proceeding with fewer than two evidence bundles, so confidence should remain conservative.",
                field_name="top_chunk_bundles",
                metadata={"bundle_count": len(analysis_input.top_chunk_bundles)},
            )
        )
    if len(analysis_input.document_text.strip()) < 250 or not analysis_input.section_context:
        warnings.append(
            make_analysis_warning(
                "prompt_context_too_sparse",
                "Prompt context is sparse because the document excerpt or section context is limited.",
                field_name="document_text",
                metadata={
                    "document_char_count": len(analysis_input.document_text.strip()),
                    "section_count": len(analysis_input.section_context),
                },
            )
        )
    if not analysis_input.expected_output_schema:
        warnings.append(
            make_analysis_warning(
                "expected_output_schema_missing",
                "Prompt assembly did not receive the expected structured output schema.",
                field_name="expected_output_schema",
            )
        )
    return warnings


# ---------------------------------------------------------------------------
# Prompt context and generic analysis runner
# ---------------------------------------------------------------------------


def build_analysis_prompt_context(
    analysis_input: WorkerAnalysisInput,
    rubric: AnalysisRubric = DEFAULT_ANALYSIS_RUBRIC,
) -> dict[str, Any]:
    """Build a compact provider-agnostic prompt context object."""
    return {
        "worker_name": analysis_input.worker_name,
        "document_type": analysis_input.document_type.value,
        "ticker": analysis_input.ticker,
        "document_title": analysis_input.document_title,
        "document_excerpt": analysis_input.document_text_excerpt,
        "analysis_instructions": analysis_input.analysis_instructions,
        "rubric": {
            "rubric_id": rubric.rubric_id,
            "version": rubric.version,
            "summary": rubric.summary,
            "core_principles": rubric.core_principles,
            "dimensions": [
                {
                    "dimension": definition.dimension.value,
                    "objective": definition.objective,
                    "evidence_expectation": definition.evidence_expectation,
                }
                for definition in rubric.dimension_definitions
            ],
        },
        "sections": [section.model_dump(mode="json") for section in analysis_input.section_context],
        "evidence_bundles": [
            {
                "bundle_id": bundle.bundle_id,
                "chunk_id": bundle.chunk_id,
                "section_title": bundle.section_title,
                "retrieval_rank": bundle.retrieval_rank,
                "evidence_type": bundle.evidence_type.value,
                "primary_text": bundle.primary_text,
                "expanded_context_text": bundle.expanded_context_text,
                "local_context_summary": bundle.local_context_summary,
            }
            for bundle in analysis_input.top_chunk_bundles
        ],
        "expected_output_schema": analysis_input.expected_output_schema,
    }


def retrieve_analysis_context(
    processed_document: ProcessedDocument,
    *,
    graph_index: GraphDocumentIndex | None = None,
    chunk_embeddings: Sequence[Any] | None = None,
    chunk_retrieval_result: ChunkRetrievalResult | None = None,
    query_text: str | None = None,
    embedding_config: Any | None = None,
    retrieval_config: Any | None = None,
    preferred_embedding_client: Any | None = None,
) -> tuple[ChunkRetrievalResult, GraphDocumentIndex, list[Any], list[AnalysisWarning]]:
    """Resolve the chunk-level context that shared worker analysis will inspect."""
    from pipeline.processing.embeddings import (
        build_chunk_embedding_records,
        build_document_graph,
        retrieve_relevant_chunks,
    )

    warnings: list[AnalysisWarning] = []
    resolved_graph_index = graph_index or build_document_graph(processed_document)
    resolved_chunk_embeddings = list(chunk_embeddings or [])

    if chunk_retrieval_result is not None:
        return chunk_retrieval_result, resolved_graph_index, resolved_chunk_embeddings, warnings

    if not resolved_chunk_embeddings:
        resolved_chunk_embeddings, embedding_notes = build_chunk_embedding_records(
            processed_document,
            embedding_config,
            preferred_client=preferred_embedding_client,
        )
        processed_document.processing_notes.extend(embedding_notes)
        if not resolved_chunk_embeddings:
            warnings.append(
                make_analysis_warning(
                    "missing_chunk_embeddings",
                    "Shared analysis context had to proceed without precomputed chunk embeddings.",
                )
            )

    effective_query = query_text or build_generic_analysis_query(processed_document)
    retrieval_result = retrieve_relevant_chunks(
        effective_query,
        processed_document,
        resolved_chunk_embeddings,
        resolved_graph_index,
        embedding_config,
        retrieval_config,
        preferred_client=preferred_embedding_client,
    )
    if not retrieval_result.hits:
        warnings.append(
            make_analysis_warning(
                "no_ranked_chunk_hits",
                "Chunk retrieval did not return ranked hits for shared worker analysis.",
            )
        )
    return retrieval_result, resolved_graph_index, resolved_chunk_embeddings, warnings


def run_generic_analysis(
    analysis_input: WorkerAnalysisInput,
    *,
    analysis_client: BaseAnalysisClient,
    rubric: AnalysisRubric = DEFAULT_ANALYSIS_RUBRIC,
    prompt_context: dict[str, Any] | None = None,
    prompt_text: str | None = None,
) -> WorkerAnalysisOutput:
    """Run the shared worker-analysis path with the configured analysis client."""
    resolved_prompt_context = prompt_context or build_analysis_prompt_context(analysis_input, rubric)
    resolved_prompt_text = prompt_text or json.dumps(resolved_prompt_context, default=str, indent=2)
    request = AnalysisClientRequest(
        worker_name=analysis_input.worker_name,
        document_type=analysis_input.document_type,
        analysis_input=analysis_input,
        prompt_context=resolved_prompt_context,
        prompt_text=resolved_prompt_text,
        rubric=rubric,
    )
    response = analysis_client.run_analysis(request)
    raw_output = response.raw_output if response.raw_output is not None else response.raw_text
    analysis_output = parse_worker_analysis_output(
        raw_output,
        analysis_input,
        client_name=response.client_name,
        model_name=response.model_name,
        is_mock=response.is_mock,
        client_warnings=response.warnings,
        prompt_context=resolved_prompt_context,
    )
    analysis_output.warnings = collect_analysis_warnings(analysis_output)
    if response.status == ProcessingStatus.ANALYSIS_FAILED and analysis_output.status != ProcessingStatus.ANALYSIS_FAILED:
        analysis_output.status = ProcessingStatus.PARTIAL
    return analysis_output


# ---------------------------------------------------------------------------
# Error mapping helpers
# ---------------------------------------------------------------------------


def analysis_issue_to_pipeline_error(issue: AnalysisIssue, document_type: DocumentType) -> PipelineError:
    """Map an analysis-layer issue into the pipeline-wide error shape."""
    return PipelineError(
        error_code=issue.issue_code,
        message=issue.message,
        stage="worker_analysis",
        document_type=document_type,
        recoverable=issue.recoverable,
        details={
            "severity": issue.severity.value,
            "dimension": issue.dimension.value if issue.dimension else None,
            "source_chunk_id": issue.source_chunk_id,
            "field_name": issue.field_name,
            **issue.metadata,
        },
    )


def analysis_warning_to_pipeline_error(warning: AnalysisWarning, document_type: DocumentType) -> PipelineError:
    """Map an analysis warning into the pipeline-wide warning shape."""
    return PipelineError(
        error_code=warning.issue_code,
        message=warning.message,
        stage="worker_analysis",
        document_type=document_type,
        recoverable=True,
        details={
            "severity": warning.severity.value,
            "dimension": warning.dimension.value if warning.dimension else None,
            "source_chunk_id": warning.source_chunk_id,
            "field_name": warning.field_name,
            **warning.metadata,
        },
    )


# ---------------------------------------------------------------------------
# Worker output assembly
# ---------------------------------------------------------------------------


def assemble_worker_output(
    analysis_output: WorkerAnalysisOutput,
    *,
    worker_input: WorkerInput | None = None,
) -> WorkerOutput:
    """Final worker-output assembly with warnings, document metadata, and provenance."""
    from pipeline.retrieval.base import build_provenance_record
    from pipeline.processing.sections import selected_document_from_retrieval_result

    key_points = [finding.summary for finding in analysis_output.findings[:5]]
    caveats = [warning.message for warning in analysis_output.warnings]
    warnings = [analysis_warning_to_pipeline_error(warning, analysis_output.document_type) for warning in analysis_output.warnings]
    issues = [analysis_issue_to_pipeline_error(issue, analysis_output.document_type) for issue in analysis_output.issues]

    selected_document = None
    retrieval_provenance: list[ProvenanceRecord] = []
    document_metadata = None
    if worker_input is not None:
        retrieval_provenance = list(worker_input.retrieval_result.provenance)
        selected_document = selected_document_from_retrieval_result(worker_input.retrieval_result)
        if selected_document is not None:
            document_metadata = selected_document.metadata
            retrieval_provenance = list(selected_document.provenance) + retrieval_provenance

    reasoning_notes = list(analysis_output.reasoning_trace.reasoning_notes) if analysis_output.reasoning_trace else []
    unresolved_items = list(analysis_output.reasoning_trace.unresolved_items) if analysis_output.reasoning_trace else []
    provenance = retrieval_provenance + [
        build_provenance_record(
            stage="worker_analysis",
            adapter_name=analysis_output.reasoning_trace.analysis_client_name if analysis_output.reasoning_trace else analysis_output.worker_name,
            candidate_id=document_metadata.document_id if document_metadata else None,
            document_type=analysis_output.document_type,
            source_name=document_metadata.source_name if document_metadata else None,
            source_identifier=document_metadata.source_identifier if document_metadata else None,
            source_url=document_metadata.source_url if document_metadata else None,
            note="Worker output assembled from the shared analysis path.",
            metadata={
                "worker_name": analysis_output.worker_name,
                "model_name": analysis_output.reasoning_trace.model_name if analysis_output.reasoning_trace else None,
                "used_mock_client": analysis_output.reasoning_trace.used_mock_client if analysis_output.reasoning_trace else None,
                "unresolved_items": unresolved_items,
            },
        )
    ]

    return WorkerOutput(
        worker_name=analysis_output.worker_name,
        document_type=analysis_output.document_type,
        status=analysis_output.status,
        summary=analysis_output.summary,
        sentiment=build_sentiment_assessment_from_scores(analysis_output.scores),
        tone=build_tone_assessment_from_scores(analysis_output.scores),
        key_points=key_points,
        evidence=list(analysis_output.evidence),
        caveats=caveats,
        issues=issues,
        confidence=analysis_output.confidence_score,
        warnings=warnings,
        provenance=provenance,
        document_metadata=document_metadata,
        reasoning_notes=reasoning_notes + unresolved_items,
    )


# ---------------------------------------------------------------------------
# BaseAnalysisWorker
# ---------------------------------------------------------------------------


class BaseAnalysisWorker(BaseWorker):
    """Reusable shared worker implementation with modular prompt assembly."""

    rubric: ClassVar[AnalysisRubric] = DEFAULT_ANALYSIS_RUBRIC
    worker_label: ClassVar[str] = "Base Analysis Worker"
    worker_specific_instruction: ClassVar[str] = ""
    analysis_focus_query: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        analysis_client: BaseAnalysisClient | None = None,
        rubric: AnalysisRubric | None = None,
        analysis_instructions: str | None = None,
    ) -> None:
        self.analysis_client = analysis_client
        self.runtime_rubric = rubric or self.rubric
        self.analysis_instructions = analysis_instructions

    def get_analysis_client(self, worker_input: WorkerInput) -> BaseAnalysisClient:
        """Resolve the effective analysis client for this worker invocation."""
        return build_analysis_client(worker_input.config, preferred_client=self.analysis_client)

    def build_analysis_query(self, processed_document: ProcessedDocument) -> str:
        """Build the retrieval query used to select chunk evidence for this worker."""
        return self.analysis_focus_query or build_generic_analysis_query(processed_document)

    def get_worker_specific_instruction(self, worker_input: WorkerInput | None = None) -> str:
        """Return the disclosure-type-specific master prompt block."""
        return self.worker_specific_instruction.strip()

    def get_analysis_instructions(self, worker_input: WorkerInput) -> str:
        """Return the shared plus worker-specific instruction text for the analysis packet."""
        if self.analysis_instructions:
            return self.analysis_instructions.strip()
        blocks = [
            build_shared_worker_instruction(self.runtime_rubric),
            self.get_worker_specific_instruction(worker_input),
        ]
        return "\n\n".join(block for block in blocks if block.strip()).strip()

    def validate_worker_input(
        self,
        worker_input: WorkerInput,
        selected_document: SelectedDocument,
    ) -> list[AnalysisWarning]:
        """Warn when worker input or selected-document types do not match the specialized worker."""
        warnings: list[AnalysisWarning] = []
        if worker_input.document_type != self.document_type:
            warnings.append(
                make_analysis_warning(
                    "worker_document_type_mismatch",
                    f"{self.worker_name} expected {self.document_type.value} but received worker input for {worker_input.document_type.value}.",
                    field_name="document_type",
                    metadata={
                        "expected_document_type": self.document_type.value,
                        "received_document_type": worker_input.document_type.value,
                    },
                )
            )
        if selected_document.document_type != self.document_type:
            warnings.append(
                make_analysis_warning(
                    "selected_document_type_mismatch",
                    f"Selected document type {selected_document.document_type.value} does not match {self.worker_name} specialization {self.document_type.value}.",
                    field_name="selected_document.document_type",
                    metadata={
                        "expected_document_type": self.document_type.value,
                        "selected_document_type": selected_document.document_type.value,
                    },
                )
            )
        return warnings

    def build_prompt_text(self, analysis_input: WorkerAnalysisInput) -> str:
        """Build the final prompt text for one worker invocation."""
        return build_worker_prompt(
            shared_instruction=build_shared_worker_instruction(self.runtime_rubric),
            worker_specific_instruction=self.get_worker_specific_instruction(),
            analysis_input=analysis_input,
        )

    def build_prompt_context(
        self,
        analysis_input: WorkerAnalysisInput,
        prompt_text: str,
    ) -> dict[str, Any]:
        """Build a compact structured context object alongside the final prompt text."""
        from pipeline.processing.embeddings import make_text_excerpt

        prompt_context = build_analysis_prompt_context(analysis_input, self.runtime_rubric)
        prompt_context.update(
            {
                "worker_label": self.worker_label,
                "analysis_focus_query": analysis_input.retrieval_query,
                "prompt_required_sections": list(WORKER_PROMPT_REQUIRED_SECTIONS),
                "worker_specific_instruction_excerpt": make_text_excerpt(self.get_worker_specific_instruction(), 360),
                "assembled_prompt_excerpt": make_text_excerpt(prompt_text, 900),
            }
        )
        return prompt_context

    def build_prompt_package(
        self,
        analysis_input: WorkerAnalysisInput,
    ) -> tuple[str, dict[str, Any], list[AnalysisWarning]]:
        """Assemble prompt text, structured prompt context, and prompt-level warnings."""
        prompt_text = self.build_prompt_text(analysis_input)
        prompt_context = self.build_prompt_context(analysis_input, prompt_text)
        prompt_warnings = deduplicate_analysis_items(
            validate_worker_prompt_sections(prompt_text) + validate_prompt_analysis_context(analysis_input),
            lambda item: (item.issue_code, item.message, item.dimension, item.source_chunk_id, item.field_name),
        )
        return prompt_text, prompt_context, prompt_warnings

    def prepare_analysis_input(
        self,
        worker_input: WorkerInput,
    ) -> tuple[WorkerAnalysisInput | None, list[AnalysisWarning], list[AnalysisIssue]]:
        """Prepare the shared analysis packet from retrieval output plus processed context."""
        from pipeline.processing.embeddings import build_document_graph
        from pipeline.processing.sections import (
            process_selected_document,
            selected_document_from_retrieval_result,
        )

        selected_document = selected_document_from_retrieval_result(worker_input.retrieval_result)
        if selected_document is None:
            return None, [], [
                make_analysis_issue(
                    "no_selected_document",
                    "Worker analysis could not start because retrieval did not produce a selected document.",
                    recoverable=True,
                )
            ]

        input_warnings = self.validate_worker_input(worker_input, selected_document)
        graph_context = worker_input.graph_context
        processed_document = graph_context.get("processed_document")
        if not isinstance(processed_document, ProcessedDocument):
            processed_document = process_selected_document(selected_document)

        graph_index = graph_context.get("graph_index")
        if not isinstance(graph_index, GraphDocumentIndex):
            graph_index = build_document_graph(processed_document)

        chunk_embeddings = graph_context.get("chunk_embeddings")
        chunk_retrieval_result = graph_context.get("chunk_retrieval_result")
        explicit_query = graph_context.get("analysis_query")
        analysis_query = explicit_query if isinstance(explicit_query, str) else self.build_analysis_query(processed_document)
        analysis_context, _, _, context_warnings = retrieve_analysis_context(
            processed_document,
            graph_index=graph_index,
            chunk_embeddings=chunk_embeddings if isinstance(chunk_embeddings, list) else None,
            chunk_retrieval_result=(
                chunk_retrieval_result if isinstance(chunk_retrieval_result, ChunkRetrievalResult) else None
            ),
            query_text=analysis_query,
        )

        analysis_input, builder_warnings = build_worker_analysis_input(
            worker_input,
            processed_document,
            analysis_context,
            worker_name=self.worker_name,
            analysis_instructions=self.get_analysis_instructions(worker_input),
            rubric=self.runtime_rubric,
        )
        return analysis_input, input_warnings + context_warnings + builder_warnings, []

    def analyze_to_internal_output(self, worker_input: WorkerInput) -> WorkerAnalysisOutput:
        """Run the shared worker-analysis path and keep the richer internal output."""
        analysis_input, preparation_warnings, preparation_issues = self.prepare_analysis_input(worker_input)
        if analysis_input is None:
            issue_messages = [issue.message for issue in preparation_issues]
            return WorkerAnalysisOutput(
                run_id=worker_input.run_id,
                ticker=worker_input.ticker,
                worker_name=self.worker_name,
                document_type=worker_input.document_type,
                status=ProcessingStatus.NO_DOCUMENT,
                summary=None,
                confidence_score=0.0,
                warnings=[],
                issues=preparation_issues,
                reasoning_trace=WorkerReasoningTrace(
                    analysis_client_name="none",
                    model_name="none",
                    used_mock_client=False,
                    reasoning_notes=[],
                    unresolved_items=issue_messages,
                    raw_response_present=False,
                ),
                raw_output={},
                is_mock_analysis=False,
            )

        prompt_text, prompt_context, prompt_warnings = self.build_prompt_package(analysis_input)
        analysis_output = run_generic_analysis(
            analysis_input,
            analysis_client=self.get_analysis_client(worker_input),
            rubric=self.runtime_rubric,
            prompt_context=prompt_context,
            prompt_text=prompt_text,
        )
        analysis_output.warnings = deduplicate_analysis_items(
            list(preparation_warnings) + list(prompt_warnings) + list(analysis_output.warnings),
            lambda item: (item.issue_code, item.message, item.dimension, item.source_chunk_id, item.field_name),
        )
        return analysis_output

    def build_worker_output(self, analysis_output: WorkerAnalysisOutput, worker_input: WorkerInput | None = None) -> WorkerOutput:
        """Map the internal analysis result into the shared worker output contract."""
        return assemble_worker_output(analysis_output, worker_input=worker_input)

    def analyze(self, worker_input: WorkerInput) -> WorkerOutput:
        """Run disclosure-specific shared analysis end to end for one selected disclosure."""
        analysis_output = self.analyze_to_internal_output(worker_input)
        return self.build_worker_output(analysis_output, worker_input=worker_input)


# ---------------------------------------------------------------------------
# Worker-specific master prompts
# ---------------------------------------------------------------------------

MATERIAL_EVENT_MASTER_PROMPT = build_worker_specific_instruction(
    worker_label="Material Event 8-K / Reg FD Press Release",
    worker_job=(
        "Evaluate one material-event disclosure as an analyst would: isolate the triggering event, "
        "the announced change, the operational and financial consequences, the dependency structure, "
        "and the detail the disclosure does or does not provide."
    ),
    document_native_focus=(
        "Treat the document as a formal disclosure of a material event, amendment, termination, obligation, "
        "or Reg FD update that should make a concrete development inspectable."
    ),
    what_to_look_for=[
        "Identify the triggering event or decision and the specific part of the business, program, agreement, obligation, or governance process it affects.",
        "Explain what actually changed relative to the prior state described in the document, not what the tone implies.",
        "Evaluate whether the disclosure makes the economic, operational, legal, regulatory, or strategic consequences inspectable.",
        "Note next steps, closing conditions, contingencies, milestones, or third-party dependencies that drive what happens next.",
        "Flag expected details that are missing even though the event is presented as material.",
    ],
    fogging_guidance=[
        "Treat the disclosure as evasive when it announces a material development but leaves scope, economics, counterparties, timing, or consequences too abstract to inspect.",
        "Do not call it fogging merely because some facts remain confidential if the document clearly explains what is known, what is withheld, and why.",
        "Separate defensive legal framing from true operational opacity; a cautious tone alone is not enough."
    ],
    signal_guidance=[
        "Positive signal comes from favorable disclosed changes such as new partnerships, licensing agreements, executive appointments, patent grants, product launches, or strategic acquisitions.",
        "Negative signal comes from disclosed burdens such as layoffs, restructuring, litigation settlements, regulatory warnings, executive departures, asset impairments, or operational downsizing.",
        "Mixed signal is appropriate when beneficial and adverse consequences coexist.",
        "IMPORTANT: Most 8-K filings describe material corporate events that DO have directional investment implications. Classify the event type and assign an appropriate positive or negative score. Avoid defaulting to neutral unless the filing is purely procedural (e.g., bylaw amendment, committee appointment).",
    ],
)


CLINICAL_TRIAL_UPDATE_MASTER_PROMPT = build_worker_specific_instruction(
    worker_label="Clinical Trial Registry Update / ClinicalTrials.gov",
    worker_job=(
        "Evaluate one structured registry update by focusing on study status, timing, design, enrollment, endpoints, results status, "
        "record coherence, and whether the update improves or reduces analyst clarity."
    ),
    document_native_focus=(
        "Treat the document as a structured registry record rather than a narrative press release. "
        "Prioritize operational state, metadata quality, and interpretability over rhetorical tone."
    ),
    what_to_look_for=[
        "Identify what appears changed or newly specified in status, timing, enrollment, eligibility, design, endpoints, locations, or results fields.",
        "Assess whether the update increases clarity, reduces clarity, or leaves key comparability questions unresolved.",
        "Evaluate whether execution appears stable, drifting, paused, or only partially explained based on the disclosed metadata.",
        "Check whether the registry record is internally coherent across dates, statuses, enrollment figures, endpoint wording, and site information.",
        "Note where partial reporting, metadata evolution, or missing historical context limits interpretability."
    ],
    fogging_guidance=[
        "Treat the record as evasive when study-state implications are material but the metadata shifts are not explained enough to assess what changed or why.",
        "Do not confuse ordinary structured brevity with fogging if the fields are precise, internally consistent, and sufficient to interpret the study state.",
        "If prior-state comparison is uncertain because the earlier registry version is absent, say so explicitly instead of inventing a delta."
    ],
    signal_guidance=[
        "Positive signal comes from cleaner metadata, more specific timing or endpoint definitions, operational stabilization, or improved comparability and transparency.",
        "Negative signal comes from drift, delayed timelines, enrollment slippage, endpoint ambiguity, incoherent field updates, or partial reporting that reduces interpretability.",
        "A registry update can be neutral in business direction but still positive or negative for clarity; keep those ideas distinct.",
    ],
)


FDA_REVIEW_MASTER_PROMPT = build_worker_specific_instruction(
    worker_label="FDA Review Materials / Regulatory Review Documents",
    worker_job=(
        "Evaluate one FDA review or regulatory review document with disciplined regulatory interpretation. "
        "Focus on posture, evidence strength, unresolved issues, safety, CMC readiness, label scope, and timing implications."
    ),
    document_native_focus=(
        "Treat the document as a regulatory review artifact, not an emotional tone document. "
        "The goal is to interpret how the review record frames benefit, risk, remaining deficiencies, and likely pathway implications."
    ),
    what_to_look_for=[
        "Assess overall regulatory posture from the evidence presented, including where the review appears supportive, conditional, unresolved, or constrained.",
        "Evaluate the strength and limitations of the efficacy and safety evidence as described in the review materials.",
        "Identify unresolved review issues, especially safety, tolerability, CMC/manufacturing, process validation, inspection, or post-marketing obligation implications.",
        "Note label or use constraints, population narrowing, monitoring requirements, or pathway implications that affect practical approval value.",
        "Distinguish factual review conclusions from sponsor framing if both appear in the same package."
    ],
    fogging_guidance=[
        "Treat the document as evasive only when material review implications are abstracted away, contradictory, or not linked clearly to the cited evidence.",
        "Do not translate normal regulatory caution into negativity by default; regulatory specificity can be cautious and still clear.",
        "If timing implications depend on unresolved items, state the dependency explicitly rather than pretending the timetable is known."
    ],
    signal_guidance=[
        "Positive signal comes from supportive benefit-risk framing, strong evidence, manageable safety findings, resolved CMC questions, or clear approval pathway support.",
        "Negative signal comes from unresolved deficiencies, meaningful safety liabilities, weak evidence, manufacturing readiness gaps, restrictive labeling, or timing risks tied to open review issues.",
        "Mixed signal is appropriate when the review supports some elements but limits scope, timing, or commercial usefulness."
    ],
)


FINANCING_DILUTION_MASTER_PROMPT = build_worker_specific_instruction(
    worker_label="Financing / Dilution Disclosure",
    worker_job=(
        "Evaluate one financing or dilution-bearing disclosure by analyzing structure, economic implications, use-of-proceeds clarity, restrictions, "
        "runway implications, and what the deal suggests about urgency versus opportunism."
    ),
    document_native_focus=(
        "Treat the document as a capital-structure disclosure rather than a generic good-news or bad-news announcement. "
        "Focus on terms, contingencies, counterparties, dilution mechanics, and downstream constraints."
    ),
    what_to_look_for=[
        "Identify the financing structure, security mix, pricing mechanics, warrants, prefunded instruments, tranches, closing conditions, and counterparties when disclosed.",
        "Assess dilution, overhang, anti-dilution protections, variable-rate exposure, or structural complexity that changes the economic meaning of the financing.",
        "Evaluate use-of-proceeds specificity, liquidity implications, runway claims, and whether management makes the capital need inspectable.",
        "Consider what the financing suggests about bargaining position, urgency, optionality, and whether the company appears constrained or opportunistic.",
        "Note disclosure gaps around fees, restrictions, covenants, registration obligations, lockups, or contingency structure if those omissions matter."
    ],
    fogging_guidance=[
        "Treat the disclosure as evasive when economic meaning is hard to inspect because pricing, dilution mechanics, contingencies, counterparties, or restrictions remain too vague.",
        "Do not label a financing negative solely because it is dilutive; judge the structure, necessity, and clarity of what was disclosed.",
        "A concise term summary can still be clear if it makes proceeds, securities, and constraints inspectable."
    ],
    signal_guidance=[
        "Positive signal comes from non-dilutive financing, strategic partnerships, or capital raised at a premium to market price with minimal restrictions.",
        "Negative signal comes from equity dilution, share offerings below market price, warrant-heavy structures, forced conversions, at-the-market offerings, or any financing that increases share count. Most common equity offerings are inherently dilutive and should lean negative.",
        "Mixed signal is appropriate ONLY when the company secures capital at favorable terms that meaningfully offset the dilution impact.",
        "IMPORTANT: Standard secondary offerings, shelf registrations, and ATM programs are dilutive events that typically warrant a negative sentiment score.",
    ],
)


INVESTOR_COMMUNICATION_MASTER_PROMPT = build_worker_specific_instruction(
    worker_label="Investor Presentation / Corporate Update / Transcript",
    worker_job=(
        "Evaluate one investor-facing communication by separating rhetoric from substance, identifying the story management is trying to tell, "
        "and judging whether the communication actually improves analyst clarity."
    ),
    document_native_focus=(
        "Treat the document as a narrative communication that may mix useful operational updates, milestone framing, risk discussion, and presentational polish. "
        "The task is to assess substance, internal consistency, omissions, and framing choices."
    ),
    what_to_look_for=[
        "Identify the central story management is advancing and the specific facts used to support it.",
        "Note what is newly emphasized, de-emphasized, or omitted relative to the rest of the document's disclosed priorities and risk discussion.",
        "Evaluate specificity versus polish: milestones, timelines, operational facts, and quantified statements should carry more weight than presentation quality.",
        "Assess risk treatment, internal consistency across speakers or slides, and whether answers actually resolve analyst-relevant questions.",
        "Distinguish genuine clarity gains from perception steering that leaves the underlying operating picture largely unchanged."
    ],
    fogging_guidance=[
        "Treat the communication as evasive when it repeats high-level themes but avoids operational specifics, tradeoffs, or risk-bearing detail that analysts would reasonably expect.",
        "Do not penalize ordinary presentation structure; polished delivery is only a concern when it substitutes for inspectable substance.",
        "If the communication is balanced and specific about risks, that should lower fogging even when management remains optimistic."
    ],
    signal_guidance=[
        "Positive signal comes from strong financial results, pipeline advancement, regulatory progress, growing revenue, expanding margins, or confirmed milestones.",
        "Negative signal comes from revenue decline, margin compression, pipeline setbacks, regulatory delays, guidance cuts, restructuring, or weakening competitive position.",
        "Mixed signal is appropriate when management reports some wins alongside material challenges or declining metrics.",
        "IMPORTANT: Investor communications (earnings calls, shareholder letters, press releases) almost always contain directional financial or operational signals. Assign a clearly positive or negative sentiment based on the underlying business trajectory, not the management tone.",
    ],
)


# ---------------------------------------------------------------------------
# Concrete worker classes
# ---------------------------------------------------------------------------


class MaterialEventWorker(BaseAnalysisWorker):
    worker_name = "material_event_worker"
    worker_label = "Material Event 8-K / Reg FD Press Release"
    document_type = DocumentType.MATERIAL_EVENT
    analysis_focus_query = (
        "Triggering event, disclosed change, consequences, dependencies, next steps, framing quality, and missing material detail."
    )
    worker_specific_instruction = MATERIAL_EVENT_MASTER_PROMPT


class ClinicalTrialUpdateWorker(BaseAnalysisWorker):
    worker_name = "clinical_trial_update_worker"
    worker_label = "Clinical Trial Registry Update / ClinicalTrials.gov"
    document_type = DocumentType.CLINICAL_TRIAL_UPDATE
    analysis_focus_query = (
        "Study status, timing, design, enrollment, endpoints, results status, metadata coherence, clarity change, and interpretability."
    )
    worker_specific_instruction = CLINICAL_TRIAL_UPDATE_MASTER_PROMPT


class FDAReviewWorker(BaseAnalysisWorker):
    worker_name = "fda_review_worker"
    worker_label = "FDA Review Materials / Regulatory Review Documents"
    document_type = DocumentType.FDA_REVIEW
    analysis_focus_query = (
        "Regulatory posture, benefit-risk framing, evidence strength, unresolved issues, safety, manufacturing readiness, label scope, and timing implications."
    )
    worker_specific_instruction = FDA_REVIEW_MASTER_PROMPT


class FinancingDilutionWorker(BaseAnalysisWorker):
    worker_name = "financing_dilution_worker"
    worker_label = "Financing / Dilution Disclosure"
    document_type = DocumentType.FINANCING_DILUTION
    analysis_focus_query = (
        "Financing structure, dilution, overhang, use of proceeds, urgency versus opportunism, restrictions, runway, and bargaining position."
    )
    worker_specific_instruction = FINANCING_DILUTION_MASTER_PROMPT


class InvestorCommunicationWorker(BaseAnalysisWorker):
    worker_name = "investor_communication_worker"
    worker_label = "Investor Presentation / Corporate Update / Transcript"
    document_type = DocumentType.INVESTOR_COMMUNICATION
    analysis_focus_query = (
        "Management narrative, emphasis versus omission, specificity, milestone framing, risk treatment, consistency, and analyst-useful clarity."
    )
    worker_specific_instruction = INVESTOR_COMMUNICATION_MASTER_PROMPT
