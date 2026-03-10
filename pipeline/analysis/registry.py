"""Worker registry mapping DocumentType to concrete worker classes."""
from __future__ import annotations

from typing import Type

from pipeline.enums import DocumentType
from pipeline.models import BaseWorker
from pipeline.analysis.workers import (
    ClinicalTrialUpdateWorker,
    FDAReviewWorker,
    FinancingDilutionWorker,
    InvestorCommunicationWorker,
    MaterialEventWorker,
)

__all__ = [
    "WORKER_REGISTRY",
]

WORKER_REGISTRY: dict[DocumentType, Type[BaseWorker]] = {
    DocumentType.MATERIAL_EVENT: MaterialEventWorker,
    DocumentType.CLINICAL_TRIAL_UPDATE: ClinicalTrialUpdateWorker,
    DocumentType.FDA_REVIEW: FDAReviewWorker,
    DocumentType.FINANCING_DILUTION: FinancingDilutionWorker,
    DocumentType.INVESTOR_COMMUNICATION: InvestorCommunicationWorker,
}
