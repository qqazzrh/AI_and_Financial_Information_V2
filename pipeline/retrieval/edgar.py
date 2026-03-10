"""SEC EDGAR retrieval adapters and shared EdgarClient."""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pipeline.config import SEC_EDGAR_USER_AGENT, logger, now_utc
from pipeline.enums import DocumentType, ProcessingStatus, SourceFamily
from pipeline.models import (
    FetchedDocumentContent,
    RawRetrievalCandidateMetadata,
    RetrievalRequest,
)
from pipeline.retrieval.base import BaseRetrievalAdapter

__all__ = [
    "EdgarClient",
    "_parse_edgar_date",
    "SECMaterialEventRetrievalAdapter",
    "SECFinancingDilutionRetrievalAdapter",
    "SECInvestorCommunicationRetrievalAdapter",
]


# ---------------------------------------------------------------------------
# Shared EDGAR HTTP client
# ---------------------------------------------------------------------------

class EdgarClient:
    """Shared HTTP client for SEC EDGAR EFTS search and filing retrieval."""

    EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    FILING_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
    RATE_LIMIT_DELAY = 0.12  # Stay under 10 req/s

    def __init__(self, user_agent: str | None = None):
        self.user_agent = user_agent or SEC_EDGAR_USER_AGENT
        if not self.user_agent:
            raise ValueError("SEC_EDGAR_USER_AGENT must be set for live EDGAR access.")
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        })
        self._last_request_time: float = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
    )
    def search_efts(
        self,
        *,
        ticker: str | None = None,
        company_name: str | None = None,
        form_types: list[str],
        start_date: date | None = None,
        end_date: date | None = None,
        query_text: str | None = None,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search EDGAR EFTS for filings matching the given criteria."""
        self._rate_limit()
        params: dict[str, Any] = {
            "forms": ",".join(form_types),
            "from": 0,
            "size": max_results,
        }
        # Build query parts
        query_parts: list[str] = []
        if query_text:
            query_parts.append(query_text)

        if query_parts:
            params["q"] = " ".join(query_parts)

        if ticker:
            params["tickers"] = ticker.upper()

        if start_date:
            params["startdt"] = start_date.isoformat()
        if end_date:
            params["enddt"] = end_date.isoformat()

        resp = self._session.get(self.EFTS_SEARCH_URL, params=params, timeout=30)
        if resp.status_code == 429:
            logger.warning("EDGAR rate limited (429). Backing off...")
            time.sleep(5)
            resp = self._session.get(self.EFTS_SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("hits", {}).get("hits", [])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
    )
    def search_submissions(
        self,
        cik: str,
        form_types: list[str],
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        query_text: str | None = None,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search filings by CIK via the data.sec.gov submissions API.

        Returns hits in a format compatible with search_efts, so existing
        fetch_filing_index_and_primary_doc works unchanged.
        """
        self._rate_limit()
        cik_padded = cik.lstrip("0").zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        resp = self._session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        company_name = data.get("name", "")
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        form_types_set = set(ft.upper() for ft in form_types)
        query_lower = query_text.lower() if query_text else None
        hits = []
        for i, (form, dt, acc, pdoc) in enumerate(zip(forms, dates, accessions, primary_docs)):
            if form.upper() not in form_types_set:
                continue
            if start_date and dt < start_date.isoformat():
                continue
            if end_date and dt > end_date.isoformat():
                continue
            # Build EFTS-compatible hit dict
            hit = {
                "_id": acc,
                "_source": {
                    "accession_no": acc,
                    "file_date": dt,
                    "form_type": form,
                    "display_names": [company_name],
                    "entity_id": cik.lstrip("0"),
                    "ciks": [cik_padded],
                    "primary_document": pdoc,
                },
            }
            hits.append(hit)
            if len(hits) >= max_results:
                break
        return hits

    def fetch_filing_text(self, filing_url: str, max_chars: int = 80_000) -> str:
        """Fetch filing HTML and return stripped plain text."""
        self._rate_limit()
        resp = self._session.get(filing_url, timeout=30)
        if resp.status_code == 429:
            time.sleep(5)
            resp = self._session.get(filing_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # Remove script and style elements
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:max_chars]

    def get_filing_primary_doc_url(self, filing_entry: dict[str, Any]) -> str | None:
        """Extract the primary document URL from an EFTS search hit."""
        source = filing_entry.get("_source", {})
        file_num = source.get("file_num", "")
        accession_raw = source.get("accession_no", "") or filing_entry.get("_id", "")
        primary_doc = source.get("file_description", "")

        # Build URL from accession number
        accession_clean = accession_raw.replace("-", "")
        if not accession_clean:
            return None

        # Try to find the primary document from the filing index
        entity_id = source.get("entity_id", "")
        if entity_id and accession_raw:
            return f"https://www.sec.gov/Archives/edgar/data/{entity_id}/{accession_clean}/{accession_raw}.txt"
        return None

    def get_filing_index_url(self, filing_entry: dict[str, Any]) -> str | None:
        """Build the filing index page URL for an EFTS hit."""
        source = filing_entry.get("_source", {})
        accession_raw = source.get("accession_no", "") or filing_entry.get("_id", "")
        entity_id = source.get("entity_id", "")
        accession_clean = accession_raw.replace("-", "")
        if entity_id and accession_clean:
            return f"https://www.sec.gov/Archives/edgar/data/{entity_id}/{accession_clean}/"
        return None

    def fetch_filing_index_and_primary_doc(self, filing_entry: dict[str, Any], max_chars: int = 80_000) -> tuple[str | None, str]:
        """Fetch filing index page, find primary doc link, fetch and return its text."""
        source = filing_entry.get("_source", {})
        # Fast path: if primary_document is known (submissions API), fetch directly
        primary_doc = source.get("primary_document")
        entity_id = source.get("entity_id", "")
        accession_raw = source.get("accession_no", "") or filing_entry.get("_id", "")
        if primary_doc and entity_id and accession_raw:
            acc_nodash = accession_raw.replace("-", "")
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{entity_id}/{acc_nodash}/{primary_doc}"
            try:
                return doc_url, self.fetch_filing_text(doc_url, max_chars)
            except Exception:
                pass  # Fall through to index-page approach

        index_url = self.get_filing_index_url(filing_entry)
        if not index_url:
            return None, ""

        self._rate_limit()
        try:
            resp = self._session.get(index_url, timeout=30)
            resp.raise_for_status()
        except Exception:
            # Fallback: try direct filing text URL
            doc_url = self.get_filing_primary_doc_url(filing_entry)
            if doc_url:
                return doc_url, self.fetch_filing_text(doc_url, max_chars)
            return None, ""

        soup = BeautifulSoup(resp.text, "lxml")
        # Find primary document link in the filing index table
        primary_url = None
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 3:
                desc = cells[1].get_text(strip=True).lower() if len(cells) > 1 else ""
                link = cells[2].find("a") if len(cells) > 2 else None
                if link is None:
                    link = cells[0].find("a")
                if link and link.get("href"):
                    href = link["href"]
                    # Prefer the main filing document (htm/html)
                    if href.endswith((".htm", ".html", ".txt")):
                        if not primary_url or "primary" in desc or "complete" in desc:
                            if href.startswith("/"):
                                primary_url = f"https://www.sec.gov{href}"
                            elif href.startswith("http"):
                                primary_url = href
                            else:
                                primary_url = index_url + href

        if not primary_url:
            # Fallback: use the direct filing text
            doc_url = self.get_filing_primary_doc_url(filing_entry)
            if doc_url:
                return doc_url, self.fetch_filing_text(doc_url, max_chars)
            return None, ""

        return primary_url, self.fetch_filing_text(primary_url, max_chars)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_edgar_date(date_str: str | None) -> datetime | None:
    """Parse an EDGAR date string into a UTC datetime."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# SEC Material Event (8-K) adapter
# ---------------------------------------------------------------------------

class SECMaterialEventRetrievalAdapter(BaseRetrievalAdapter):
    adapter_name = "sec_material_event_adapter"
    document_type = DocumentType.MATERIAL_EVENT

    def __init__(self):
        self._client = EdgarClient()

    def search_candidates(self, request: RetrievalRequest) -> list[RawRetrievalCandidateMetadata]:
        cik = request.filters.get("cik")
        if cik:
            hits = self._client.search_submissions(
                cik=cik, form_types=["8-K", "8-K/A"],
                start_date=request.start_date, end_date=request.end_date,
                max_results=request.max_candidates,
            )
        else:
            hits = self._client.search_efts(
                ticker=request.ticker, form_types=["8-K", "8-K/A"],
                start_date=request.start_date, end_date=request.end_date,
                max_results=request.max_candidates,
            )
        candidates = []
        for hit in hits:
            source = hit.get("_source", {})
            accession = source.get("accession_no", hit.get("_id", ""))
            filed_date = _parse_edgar_date(source.get("file_date") or source.get("period_of_report"))
            display_names = source.get("display_names", [])
            company_display = display_names[0] if display_names else (request.company_name or "")
            form_type = source.get("form_type", "8-K")
            candidates.append(RawRetrievalCandidateMetadata(
                raw_candidate_id=f"edgar_8k_{accession}",
                adapter_name=self.adapter_name,
                document_type=self.document_type,
                ticker=request.ticker,
                company_name=company_display,
                source_name="SEC EDGAR",
                source_identifier=accession,
                source_url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={request.ticker}&type=8-K&dateb=&owner=include&count=10",
                title=f"{request.ticker} {form_type} filing ({accession})",
                source_family=SourceFamily.OFFICIAL_REGULATORY,
                published_at=filed_date or now_utc(),
                updated_at=filed_date,
                event_date=filed_date.date() if filed_date else None,
                is_structured_source=True,
                raw_metadata={"form_type": form_type, "accession": accession, "_source": source},
            ))
        return candidates

    def fetch_document(self, raw_candidate: RawRetrievalCandidateMetadata, request: RetrievalRequest) -> FetchedDocumentContent:
        hit = {"_source": raw_candidate.raw_metadata.get("_source", {}), "_id": raw_candidate.source_identifier}
        doc_url, text = self._client.fetch_filing_index_and_primary_doc(hit)
        if not text.strip():
            return FetchedDocumentContent(
                raw_candidate_id=raw_candidate.raw_candidate_id,
                document_text=None,
                fetch_status=ProcessingStatus.EXTRACTION_FAILED,
                fetch_notes=["Could not extract text from EDGAR filing."],
                raw_payload={"doc_url": doc_url},
            )
        return FetchedDocumentContent(
            raw_candidate_id=raw_candidate.raw_candidate_id,
            document_text=text,
            content_type="text/plain",
            fetch_status=ProcessingStatus.SUCCESS,
            fetch_notes=[f"Fetched from {doc_url}"],
            raw_payload={"doc_url": doc_url},
        )


# ---------------------------------------------------------------------------
# SEC Financing/Dilution adapter
# ---------------------------------------------------------------------------

class SECFinancingDilutionRetrievalAdapter(BaseRetrievalAdapter):
    adapter_name = "sec_financing_dilution_adapter"
    document_type = DocumentType.FINANCING_DILUTION

    FORM_TYPES = ["S-3", "S-3/A", "424B5", "424B2", "S-1", "S-1/A"]

    def __init__(self):
        self._client = EdgarClient()

    def search_candidates(self, request: RetrievalRequest) -> list[RawRetrievalCandidateMetadata]:
        cik = request.filters.get("cik")
        if cik:
            hits = self._client.search_submissions(
                cik=cik, form_types=self.FORM_TYPES,
                start_date=request.start_date, end_date=request.end_date,
                max_results=request.max_candidates,
            )
        else:
            hits = self._client.search_efts(
                ticker=request.ticker, form_types=self.FORM_TYPES,
                start_date=request.start_date, end_date=request.end_date,
                max_results=request.max_candidates,
            )
        candidates = []
        for hit in hits:
            source = hit.get("_source", {})
            accession = source.get("accession_no", hit.get("_id", ""))
            filed_date = _parse_edgar_date(source.get("file_date") or source.get("period_of_report"))
            display_names = source.get("display_names", [])
            company_display = display_names[0] if display_names else (request.company_name or "")
            form_type = source.get("form_type", "S-3")
            candidates.append(RawRetrievalCandidateMetadata(
                raw_candidate_id=f"edgar_fin_{accession}",
                adapter_name=self.adapter_name,
                document_type=self.document_type,
                ticker=request.ticker,
                company_name=company_display,
                source_name="SEC EDGAR",
                source_identifier=accession,
                source_url=self._client.get_filing_index_url(hit),
                title=f"{request.ticker} {form_type} filing ({accession})",
                source_family=SourceFamily.OFFICIAL_REGULATORY,
                published_at=filed_date or now_utc(),
                updated_at=filed_date,
                event_date=filed_date.date() if filed_date else None,
                is_structured_source=True,
                raw_metadata={"form_type": form_type, "accession": accession, "_source": source},
            ))
        return candidates

    def fetch_document(self, raw_candidate: RawRetrievalCandidateMetadata, request: RetrievalRequest) -> FetchedDocumentContent:
        hit = {"_source": raw_candidate.raw_metadata.get("_source", {}), "_id": raw_candidate.source_identifier}
        doc_url, text = self._client.fetch_filing_index_and_primary_doc(hit)
        if not text.strip():
            return FetchedDocumentContent(
                raw_candidate_id=raw_candidate.raw_candidate_id,
                document_text=None,
                fetch_status=ProcessingStatus.EXTRACTION_FAILED,
                fetch_notes=["Could not extract text from EDGAR filing."],
                raw_payload={"doc_url": doc_url},
            )
        return FetchedDocumentContent(
            raw_candidate_id=raw_candidate.raw_candidate_id,
            document_text=text,
            content_type="text/plain",
            fetch_status=ProcessingStatus.SUCCESS,
            fetch_notes=[f"Fetched from {doc_url}"],
            raw_payload={"doc_url": doc_url},
        )


# ---------------------------------------------------------------------------
# SEC Investor Communication adapter
# ---------------------------------------------------------------------------

class SECInvestorCommunicationRetrievalAdapter(BaseRetrievalAdapter):
    adapter_name = "sec_investor_communication_adapter"
    document_type = DocumentType.INVESTOR_COMMUNICATION

    KEYWORDS = ["earnings", "investor", "presentation", "quarterly", "results of operations"]

    def __init__(self):
        self._client = EdgarClient()

    def search_candidates(self, request: RetrievalRequest) -> list[RawRetrievalCandidateMetadata]:
        cik = request.filters.get("cik")
        if cik:
            hits = self._client.search_submissions(
                cik=cik, form_types=["8-K"],
                start_date=request.start_date, end_date=request.end_date,
                max_results=request.max_candidates * 2,
            )
        else:
            hits = self._client.search_efts(
                ticker=request.ticker, form_types=["8-K"],
                start_date=request.start_date, end_date=request.end_date,
                query_text=" OR ".join(f'"{kw}"' for kw in self.KEYWORDS),
                max_results=request.max_candidates * 2,
            )
        candidates = []
        for hit in hits:
            source = hit.get("_source", {})
            accession = source.get("accession_no", hit.get("_id", ""))
            filed_date = _parse_edgar_date(source.get("file_date") or source.get("period_of_report"))
            display_names = source.get("display_names", [])
            company_display = display_names[0] if display_names else (request.company_name or "")
            form_type = source.get("form_type", "8-K")
            candidates.append(RawRetrievalCandidateMetadata(
                raw_candidate_id=f"edgar_inv_{accession}",
                adapter_name=self.adapter_name,
                document_type=self.document_type,
                ticker=request.ticker,
                company_name=company_display,
                source_name="SEC EDGAR",
                source_identifier=accession,
                source_url=self._client.get_filing_index_url(hit),
                title=f"{request.ticker} {form_type} investor communication ({accession})",
                source_family=SourceFamily.ISSUER_PUBLISHED,
                published_at=filed_date or now_utc(),
                updated_at=filed_date,
                event_date=filed_date.date() if filed_date else None,
                is_structured_source=False,
                raw_metadata={"form_type": form_type, "accession": accession, "_source": source},
            ))
        return candidates[:request.max_candidates]

    def fetch_document(self, raw_candidate: RawRetrievalCandidateMetadata, request: RetrievalRequest) -> FetchedDocumentContent:
        hit = {"_source": raw_candidate.raw_metadata.get("_source", {}), "_id": raw_candidate.source_identifier}
        doc_url, text = self._client.fetch_filing_index_and_primary_doc(hit)
        if not text.strip():
            return FetchedDocumentContent(
                raw_candidate_id=raw_candidate.raw_candidate_id,
                document_text=None,
                fetch_status=ProcessingStatus.EXTRACTION_FAILED,
                fetch_notes=["Could not extract text from EDGAR filing."],
                raw_payload={"doc_url": doc_url},
            )
        return FetchedDocumentContent(
            raw_candidate_id=raw_candidate.raw_candidate_id,
            document_text=text,
            content_type="text/plain",
            fetch_status=ProcessingStatus.SUCCESS,
            fetch_notes=[f"Fetched from {doc_url}"],
            raw_payload={"doc_url": doc_url},
        )
