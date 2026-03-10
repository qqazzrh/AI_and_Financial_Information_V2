"""OpenFDA drug review retrieval adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pipeline.config import OPENFDA_API_KEY, now_utc
from pipeline.enums import DocumentType, ProcessingStatus, SourceFamily
from pipeline.models import (
    FetchedDocumentContent,
    RawRetrievalCandidateMetadata,
    RetrievalRequest,
)
from pipeline.retrieval.base import BaseRetrievalAdapter

__all__ = [
    "OpenFDAReviewRetrievalAdapter",
    "_parse_fda_date",
]


def _parse_fda_date(date_str: str | None) -> datetime | None:
    """Parse openFDA date format (YYYYMMDD)."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


class OpenFDAReviewRetrievalAdapter(BaseRetrievalAdapter):
    adapter_name = "openfda_review_adapter"
    document_type = DocumentType.FDA_REVIEW

    DRUGSFDA_URL = "https://api.fda.gov/drug/drugsfda.json"
    LABEL_URL = "https://api.fda.gov/drug/label.json"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
    )
    def _query_fda(self, url: str, search: str, limit: int) -> list[dict]:
        params: dict[str, Any] = {"search": search, "limit": limit}
        if OPENFDA_API_KEY:
            params["api_key"] = OPENFDA_API_KEY
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        return resp.json().get("results", [])

    def search_candidates(self, request: RetrievalRequest) -> list[RawRetrievalCandidateMetadata]:
        company = request.company_name or request.ticker
        search_query = f'sponsor_name:"{company}"'
        results = self._query_fda(self.DRUGSFDA_URL, search_query, request.max_candidates)
        candidates = []
        for result in results:
            app_num = result.get("application_number", "")
            sponsor = result.get("sponsor_name", company)
            products = result.get("products", [])
            brand_names = [p.get("brand_name", "") for p in products if p.get("brand_name")]
            submissions = result.get("submissions", [])

            # Use most recent submission date
            latest_date = None
            for sub in submissions:
                sub_date = sub.get("submission_status_date")
                parsed = _parse_fda_date(sub_date)
                if parsed and (latest_date is None or parsed > latest_date):
                    latest_date = parsed

            published = latest_date or now_utc()
            title_brand = f" ({', '.join(brand_names[:3])})" if brand_names else ""
            candidates.append(RawRetrievalCandidateMetadata(
                raw_candidate_id=f"fda_{app_num}",
                adapter_name=self.adapter_name,
                document_type=self.document_type,
                ticker=request.ticker,
                company_name=sponsor,
                source_name="openFDA",
                source_identifier=app_num,
                source_url=f"https://api.fda.gov/drug/drugsfda.json?search=application_number:{app_num}",
                title=f"FDA Application {app_num}{title_brand}",
                source_family=SourceFamily.OFFICIAL_REGULATORY,
                published_at=published,
                updated_at=published,
                event_date=published.date() if published else None,
                is_structured_source=True,
                raw_metadata={
                    "application_number": app_num,
                    "sponsor_name": sponsor,
                    "products": products,
                    "submissions": submissions,
                },
            ))
        return candidates

    def fetch_document(self, raw_candidate: RawRetrievalCandidateMetadata, request: RetrievalRequest) -> FetchedDocumentContent:
        meta = raw_candidate.raw_metadata
        text_parts = []

        # Application overview
        text_parts.append(f"FDA Application Number: {meta.get('application_number', 'N/A')}")
        text_parts.append(f"Sponsor: {meta.get('sponsor_name', 'N/A')}")

        # Products
        products = meta.get("products", [])
        if products:
            text_parts.append("\nApproved Products:")
            for product in products:
                brand = product.get("brand_name", "N/A")
                active = product.get("active_ingredients", [])
                ingredients = ", ".join(
                    f"{ai.get('name', '')} ({ai.get('strength', '')})"
                    for ai in active
                ) if active else "N/A"
                dosage = product.get("dosage_form", "N/A")
                route = product.get("route", "N/A")
                text_parts.append(f"  - {brand}: {ingredients}, {dosage}, {route}")

        # Submissions history
        submissions = meta.get("submissions", [])
        if submissions:
            text_parts.append("\nSubmission History:")
            for sub in sorted(submissions, key=lambda s: s.get("submission_status_date", ""), reverse=True)[:10]:
                sub_type = sub.get("submission_type", "N/A")
                sub_num = sub.get("submission_number", "")
                sub_status = sub.get("submission_status", "N/A")
                sub_date = sub.get("submission_status_date", "N/A")
                text_parts.append(f"  - {sub_type} #{sub_num}: {sub_status} ({sub_date})")

        # Try to fetch drug label for additional context
        company = request.company_name or request.ticker
        try:
            label_results = self._query_fda(
                self.LABEL_URL,
                f'openfda.manufacturer_name:"{company}"',
                limit=1,
            )
            if label_results:
                label = label_results[0]
                if label.get("indications_and_usage"):
                    indications = label["indications_and_usage"]
                    if isinstance(indications, list):
                        indications = " ".join(indications)
                    text_parts.append(f"\nIndications and Usage:\n{indications[:3000]}")
                if label.get("warnings"):
                    warnings_text = label["warnings"]
                    if isinstance(warnings_text, list):
                        warnings_text = " ".join(warnings_text)
                    text_parts.append(f"\nWarnings:\n{warnings_text[:2000]}")
        except Exception:
            text_parts.append("\n[Drug label data not available]")

        document_text = "\n".join(text_parts)
        return FetchedDocumentContent(
            raw_candidate_id=raw_candidate.raw_candidate_id,
            document_text=document_text if len(document_text.strip()) > 40 else None,
            content_type="text/plain",
            fetch_status=ProcessingStatus.SUCCESS if len(document_text.strip()) > 40 else ProcessingStatus.EXTRACTION_FAILED,
            fetch_notes=["Assembled from openFDA drugsfda and label endpoints."],
            raw_payload={"application_number": meta.get("application_number")},
        )
