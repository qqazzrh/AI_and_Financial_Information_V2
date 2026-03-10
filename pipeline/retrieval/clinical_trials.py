"""ClinicalTrials.gov retrieval adapter."""

from __future__ import annotations

from datetime import datetime, timezone

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pipeline.config import now_utc
from pipeline.enums import DocumentType, ProcessingStatus, SourceFamily
from pipeline.models import (
    FetchedDocumentContent,
    RawRetrievalCandidateMetadata,
    RetrievalRequest,
)
from pipeline.retrieval.base import BaseRetrievalAdapter

__all__ = [
    "ClinicalTrialsGovRetrievalAdapter",
    "_parse_ct_date",
]


def _parse_ct_date(date_str: str | None) -> datetime | None:
    """Parse ClinicalTrials.gov date formats (YYYY-MM-DD or Month YYYY)."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        return datetime.strptime(date_str, "%B %Y").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        return datetime.strptime(date_str, "%B %d, %Y").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class ClinicalTrialsGovRetrievalAdapter(BaseRetrievalAdapter):
    adapter_name = "clinicaltrials_gov_adapter"
    document_type = DocumentType.CLINICAL_TRIAL_UPDATE

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
    )
    def _search(self, company_name: str, max_results: int) -> list[dict]:
        params = {
            "query.spons": company_name,
            "sort": "LastUpdatePostDate",
            "pageSize": min(max_results, 20),
            "format": "json",
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("studies", [])

    def search_candidates(self, request: RetrievalRequest) -> list[RawRetrievalCandidateMetadata]:
        company = request.company_name or request.ticker
        studies = self._search(company, request.max_candidates)
        candidates = []
        for study in studies:
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            nct_id = ident.get("nctId", "")
            title = ident.get("officialTitle") or ident.get("briefTitle") or ""
            org_name = ident.get("organization", {}).get("fullName", "")
            overall_status = status_mod.get("overallStatus", "")
            last_update = status_mod.get("lastUpdateSubmitDate", "")

            published = _parse_ct_date(last_update) or now_utc()

            candidates.append(RawRetrievalCandidateMetadata(
                raw_candidate_id=f"ct_{nct_id}",
                adapter_name=self.adapter_name,
                document_type=self.document_type,
                ticker=request.ticker,
                company_name=org_name or company,
                source_name="ClinicalTrials.gov",
                source_identifier=nct_id,
                source_url=f"https://clinicaltrials.gov/study/{nct_id}",
                title=title,
                source_family=SourceFamily.OFFICIAL_REGULATORY,
                published_at=published,
                updated_at=published,
                event_date=published.date() if published else None,
                is_structured_source=True,
                raw_metadata={
                    "nct_id": nct_id,
                    "overall_status": overall_status,
                    "org_name": org_name,
                    "protocol_section": proto,
                },
            ))
        return candidates

    def fetch_document(self, raw_candidate: RawRetrievalCandidateMetadata, request: RetrievalRequest) -> FetchedDocumentContent:
        proto = raw_candidate.raw_metadata.get("protocol_section", {})
        text_parts = []

        # Identification
        ident = proto.get("identificationModule", {})
        text_parts.append(f"Study Title: {ident.get('officialTitle', ident.get('briefTitle', 'N/A'))}")
        text_parts.append(f"NCT ID: {ident.get('nctId', 'N/A')}")
        text_parts.append(f"Organization: {ident.get('organization', {}).get('fullName', 'N/A')}")

        # Status
        status_mod = proto.get("statusModule", {})
        text_parts.append(f"\nStudy Status: {status_mod.get('overallStatus', 'N/A')}")
        start_date = status_mod.get("startDateStruct", {}).get("date", "N/A")
        completion_date = status_mod.get("completionDateStruct", {}).get("date", "N/A")
        text_parts.append(f"Start Date: {start_date}")
        text_parts.append(f"Estimated Completion: {completion_date}")

        # Description
        desc = proto.get("descriptionModule", {})
        if desc.get("briefSummary"):
            text_parts.append(f"\nBrief Summary:\n{desc['briefSummary']}")
        if desc.get("detailedDescription"):
            text_parts.append(f"\nDetailed Description:\n{desc['detailedDescription']}")

        # Design
        design = proto.get("designModule", {})
        if design:
            text_parts.append(f"\nStudy Type: {design.get('studyType', 'N/A')}")
            phases = design.get("phases", [])
            if phases:
                text_parts.append(f"Phase: {', '.join(phases)}")
            enrollment = design.get("enrollmentInfo", {})
            if enrollment:
                text_parts.append(f"Enrollment: {enrollment.get('count', 'N/A')} ({enrollment.get('type', '')})")

        # Arms & Interventions
        arms_mod = proto.get("armsInterventionsModule", {})
        interventions = arms_mod.get("interventions", [])
        if interventions:
            text_parts.append("\nInterventions:")
            for intv in interventions[:5]:
                text_parts.append(f"  - {intv.get('type', '')}: {intv.get('name', '')} — {intv.get('description', '')}")

        # Outcomes
        outcomes_mod = proto.get("outcomesModule", {})
        primary = outcomes_mod.get("primaryOutcomes", [])
        if primary:
            text_parts.append("\nPrimary Outcomes:")
            for out in primary[:5]:
                text_parts.append(f"  - {out.get('measure', '')} (timeframe: {out.get('timeFrame', 'N/A')})")

        # Eligibility
        elig = proto.get("eligibilityModule", {})
        if elig:
            text_parts.append(f"\nEligibility: Ages {elig.get('minimumAge', 'N/A')} to {elig.get('maximumAge', 'N/A')}, Sex: {elig.get('sex', 'N/A')}")

        document_text = "\n".join(text_parts)
        return FetchedDocumentContent(
            raw_candidate_id=raw_candidate.raw_candidate_id,
            document_text=document_text,
            content_type="text/plain",
            fetch_status=ProcessingStatus.SUCCESS,
            fetch_notes=["Synthesized from ClinicalTrials.gov protocol sections."],
            raw_payload={"nct_id": raw_candidate.raw_metadata.get("nct_id")},
        )
