"""Live retrieval adapter registry mapping DocumentType to adapter classes."""

from __future__ import annotations

from pipeline.enums import DocumentType
from pipeline.retrieval.clinical_trials import ClinicalTrialsGovRetrievalAdapter
from pipeline.retrieval.edgar import (
    SECFinancingDilutionRetrievalAdapter,
    SECInvestorCommunicationRetrievalAdapter,
    SECMaterialEventRetrievalAdapter,
)
from pipeline.retrieval.openfda import OpenFDAReviewRetrievalAdapter

__all__ = ["RETRIEVAL_ADAPTER_REGISTRY"]

RETRIEVAL_ADAPTER_REGISTRY = {
    DocumentType.MATERIAL_EVENT: SECMaterialEventRetrievalAdapter,
    DocumentType.CLINICAL_TRIAL_UPDATE: ClinicalTrialsGovRetrievalAdapter,
    DocumentType.FDA_REVIEW: OpenFDAReviewRetrievalAdapter,
    DocumentType.FINANCING_DILUTION: SECFinancingDilutionRetrievalAdapter,
    DocumentType.INVESTOR_COMMUNICATION: SECInvestorCommunicationRetrievalAdapter,
}
