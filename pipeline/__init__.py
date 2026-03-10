"""Biotech Disclosure Pipeline — public API re-exports.

Uses direct Python imports and package exports for runtime orchestration.
Every public symbol required by API/CLI entrypoints is re-exported here.
"""
from pipeline.config import GLOBAL_CONFIG, PIPELINE_CONFIG, logger
from pipeline.enums import (
    AnalysisTier,
    DocumentType,
    ProcessingStatus,
    SentimentLabel,
    SourceFamily,
)
from pipeline.models import (
    ArbiterOutput,
    DocumentChunk,
    DocumentMetadata,
    DocumentSection,
    DocumentValidationResult,
    FinalUIPayload,
    MasterInput,
    MasterOutput,
    PipelineState,
    ProcessedDocument,
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalResult,
    WorkerAnalysisInput,
    WorkerOutput,
)
from pipeline.retrieval.base import (
    build_document_metadata_from_candidate,
    select_most_recent_relevant_candidate,
    validate_candidate_document,
)
from pipeline.processing.normalization import (
    normalize_bullets_and_lists,
    normalize_line_breaks,
    normalize_table_like_text,
    normalize_whitespace,
)
from pipeline.processing.sections import (
    detect_document_sections,
    process_selected_document,
    selected_document_from_retrieval_result,
)
from pipeline.analysis.rubrics import (
    build_standardized_analysis_score,
    normalize_sentiment_label,
)
from pipeline.analysis.workers import (
    BaseAnalysisWorker,
    MaterialEventWorker,
    ClinicalTrialUpdateWorker,
    FDAReviewWorker,
    FinancingDilutionWorker,
    InvestorCommunicationWorker,
    assemble_worker_output,
)
from pipeline.analysis.registry import WORKER_REGISTRY
from pipeline.arbiter.normalization import clamp_value
from pipeline.arbiter.cross_document import CrossDocumentArbiter, ARBITER_REGISTRY
from pipeline.arbiter.models import EXPECTED_DOCUMENT_TYPES
from pipeline.master import IntegratedMasterNode
from pipeline.orchestration import (
    build_retrieval_requests,
    resolve_company_from_ticker,
    run_full_pipeline,
    run_retrieval,
    run_tiered_pipeline,
    set_database_handles,
    TieredPipelineRequest,
)
from pipeline.fixtures import (
    ARBITER_DEMO_CASES,
    DEMO_ARBITER_OUTPUTS,
    DEMO_WORKER_SELECTED_DOCUMENTS,
    make_demo_worker_output,
    make_test_selected_document,
)

__all__ = [
    # Config
    "GLOBAL_CONFIG",
    "PIPELINE_CONFIG",
    "logger",
    # Enums
    "AnalysisTier",
    "DocumentType",
    "ProcessingStatus",
    "SentimentLabel",
    "SourceFamily",
    # Models
    "ArbiterOutput",
    "DocumentChunk",
    "DocumentMetadata",
    "DocumentSection",
    "DocumentValidationResult",
    "FinalUIPayload",
    "MasterInput",
    "MasterOutput",
    "PipelineState",
    "ProcessedDocument",
    "RetrievalCandidate",
    "RetrievalRequest",
    "RetrievalResult",
    "WorkerAnalysisInput",
    "WorkerOutput",
    # Retrieval
    "build_document_metadata_from_candidate",
    "select_most_recent_relevant_candidate",
    "validate_candidate_document",
    # Processing
    "detect_document_sections",
    "normalize_bullets_and_lists",
    "normalize_line_breaks",
    "normalize_table_like_text",
    "normalize_whitespace",
    "process_selected_document",
    "selected_document_from_retrieval_result",
    # Analysis
    "BaseAnalysisWorker",
    "ClinicalTrialUpdateWorker",
    "FDAReviewWorker",
    "FinancingDilutionWorker",
    "InvestorCommunicationWorker",
    "MaterialEventWorker",
    "WORKER_REGISTRY",
    "assemble_worker_output",
    "build_standardized_analysis_score",
    "normalize_sentiment_label",
    # Arbiter
    "ARBITER_REGISTRY",
    "CrossDocumentArbiter",
    "EXPECTED_DOCUMENT_TYPES",
    "clamp_value",
    # Master
    "IntegratedMasterNode",
    # Orchestration
    "TieredPipelineRequest",
    "build_retrieval_requests",
    "resolve_company_from_ticker",
    "run_full_pipeline",
    "run_retrieval",
    "run_tiered_pipeline",
    "set_database_handles",
    # Fixtures
    "ARBITER_DEMO_CASES",
    "DEMO_ARBITER_OUTPUTS",
    "DEMO_WORKER_SELECTED_DOCUMENTS",
    "make_demo_worker_output",
    "make_test_selected_document",
]
