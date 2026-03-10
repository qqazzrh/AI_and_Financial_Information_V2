"""Retrieval subpackage — adapters, scoring, validation, selection, and registry."""

from pipeline.retrieval.base import (
    BaseRetrievalAdapter,
    RetrievalErrorCode,
    RetrievalError,
    build_provenance_record,
    append_candidate_provenance,
    build_document_metadata_from_candidate,
    make_retrieval_error,
    normalize_token,
    normalize_document_text,
    validate_candidate_document,
    deduplicate_candidates,
    select_most_recent_relevant_candidate,
    evaluate_candidate_relevance,
    assign_candidate_freshness,
    build_candidate_duplicate_key,
    candidate_effective_timestamp,
    source_preference_index,
    candidate_priority_signature,
    selection_sort_key,
)
from pipeline.retrieval.edgar import (
    EdgarClient,
    _parse_edgar_date,
    SECMaterialEventRetrievalAdapter,
    SECFinancingDilutionRetrievalAdapter,
    SECInvestorCommunicationRetrievalAdapter,
)
from pipeline.retrieval.clinical_trials import (
    ClinicalTrialsGovRetrievalAdapter,
    _parse_ct_date,
)
from pipeline.retrieval.openfda import (
    OpenFDAReviewRetrievalAdapter,
    _parse_fda_date,
)
from pipeline.retrieval.registry import RETRIEVAL_ADAPTER_REGISTRY

__all__ = [
    # base.py
    "BaseRetrievalAdapter",
    "RetrievalErrorCode",
    "RetrievalError",
    "build_provenance_record",
    "append_candidate_provenance",
    "build_document_metadata_from_candidate",
    "make_retrieval_error",
    "normalize_token",
    "normalize_document_text",
    "validate_candidate_document",
    "deduplicate_candidates",
    "select_most_recent_relevant_candidate",
    "evaluate_candidate_relevance",
    "assign_candidate_freshness",
    "build_candidate_duplicate_key",
    "candidate_effective_timestamp",
    "source_preference_index",
    "candidate_priority_signature",
    "selection_sort_key",
    # edgar.py
    "EdgarClient",
    "_parse_edgar_date",
    "SECMaterialEventRetrievalAdapter",
    "SECFinancingDilutionRetrievalAdapter",
    "SECInvestorCommunicationRetrievalAdapter",
    # clinical_trials.py
    "ClinicalTrialsGovRetrievalAdapter",
    "_parse_ct_date",
    # openfda.py
    "OpenFDAReviewRetrievalAdapter",
    "_parse_fda_date",
    # registry.py
    "RETRIEVAL_ADAPTER_REGISTRY",
]
