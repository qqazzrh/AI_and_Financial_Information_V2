"""Demo data and test factories for pipeline testing."""

from __future__ import annotations

__all__ = [
    "make_test_selected_document",
    "make_worker_demo_selected_document",
    "DEMO_WORKER_SELECTED_DOCUMENTS",
    "make_demo_arbiter_evidence",
    "make_demo_worker_output",
    "ARBITER_DEMO_CASES",
    "DEMO_ARBITER_OUTPUTS",
]

from collections.abc import Sequence
from datetime import date, datetime, timezone
from typing import Any

from pipeline.models import (
    ArbiterInput,
    ArbiterOutput,
    DocumentMetadata,
    EvidenceSnippet,
    PipelineError,
    ProvenanceRecord,
    SelectedDocument,
    SentimentAssessment,
    ToneAssessment,
    WorkerOutput,
)
from pipeline.enums import (
    AnalysisDimension,
    ArbiterKind,
    ArbiterSignalCategory,
    CrossDocumentTheme,
    DocumentType,
    EvidenceInterpretation,
    EvidenceType,
    NormalizedSignalDirection,
    ProcessingStatus,
    SentimentLabel,
    SourceFamily,
)
from pipeline.config import now_utc, PIPELINE_CONFIG
from pipeline.retrieval import build_provenance_record
from pipeline.arbiter import CrossDocumentArbiter


# ---------------------------------------------------------------------------
# Cell 52 – lightweight test factory
# ---------------------------------------------------------------------------

def make_test_selected_document(
    *,
    document_id: str = "test_doc_001",
    document_type: DocumentType = DocumentType.MATERIAL_EVENT,
    title: str = "Test Document",
    ticker: str = "TEST",
    raw_text: str = "This is a test document with enough text to pass minimum length checks for pipeline processing.",
    source_name: str = "Test Source",
    source_url: str = "https://example.com/test",
) -> SelectedDocument:
    """Build a lightweight test selected document for fixture data."""
    metadata = DocumentMetadata(
        document_id=document_id,
        ticker=ticker,
        document_type=document_type,
        title=title,
        source_name=source_name,
        source_url=source_url,
        source_family=SourceFamily.ISSUER_PUBLISHED,
        retrieved_at=now_utc(),
        is_structured_source=False,
        is_mock_data=True,
        notes=["Test fixture document."],
    )
    return SelectedDocument(
        document_id=document_id,
        document_type=document_type,
        ticker=ticker,
        title=title,
        raw_text=raw_text,
        source_name=source_name,
        source_url=source_url,
        metadata=metadata,
        is_mock_data=True,
    )


# ---------------------------------------------------------------------------
# Cell 77 – richer worker-demo document factory
# ---------------------------------------------------------------------------

def make_worker_demo_selected_document(
    *,
    document_id: str,
    document_type: DocumentType,
    title: str,
    ticker: str,
    raw_text: str,
    source_name: str,
    source_url: str,
    source_family: SourceFamily,
    is_structured_source: bool,
    company_name: str = "FakeBio Therapeutics",
    source_identifier: str | None = None,
    published_at: datetime | None = None,
    updated_at: datetime | None = None,
    event_date: date | None = None,
    raw_metadata: dict[str, Any] | None = None,
) -> SelectedDocument:
    """Create one richer mock selected document specifically for Prompt 6 worker demos."""
    selected_document = make_test_selected_document(
        document_id=document_id,
        document_type=document_type,
        title=title,
        ticker=ticker,
        raw_text=raw_text,
        source_name=source_name,
        source_url=source_url,
    )
    if selected_document.metadata is None:
        raise ValueError("Prompt 6 worker demos require metadata on the mock selected document.")
    selected_document.source_identifier = source_identifier
    selected_document.metadata.company_name = company_name
    selected_document.metadata.source_identifier = source_identifier
    selected_document.metadata.source_family = source_family
    selected_document.metadata.is_structured_source = is_structured_source
    selected_document.metadata.published_at = published_at
    selected_document.metadata.updated_at = updated_at
    selected_document.metadata.event_date = event_date
    selected_document.metadata.raw_metadata = {"prompt6_demo": True, **(raw_metadata or {})}
    selected_document.metadata.notes.append("Prompt 6 worker demo document.")
    selected_document.provenance.append(
        build_provenance_record(
            stage="demo_worker_document",
            adapter_name="prompt6_demo_source",
            candidate_id=document_id,
            document_type=document_type,
            source_name=source_name,
            source_identifier=source_identifier,
            source_url=source_url,
            note="Synthetic Prompt 6 worker demo selected document.",
            is_mock_data=True,
            metadata={"prompt6_demo": True},
        )
    )
    return selected_document


# ---------------------------------------------------------------------------
# Cell 77 – demo worker selected documents
# ---------------------------------------------------------------------------

DEMO_WORKER_SELECTED_DOCUMENTS: dict[DocumentType, SelectedDocument] = {
    DocumentType.MATERIAL_EVENT: make_worker_demo_selected_document(
        document_id="prompt6_demo_material_event",
        document_type=DocumentType.MATERIAL_EVENT,
        title="FakeBio Announces Manufacturing Transfer and Collaboration Reset",
        ticker="FAKE",
        source_name="Mock SEC Filing Feed",
        source_url="https://example.com/prompt6/material-event",
        source_identifier="prompt6-material-event-001",
        source_family=SourceFamily.OFFICIAL_REGULATORY,
        is_structured_source=True,
        published_at=datetime(2026, 3, 3, 16, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 3, 18, 0, tzinfo=timezone.utc),
        event_date=date(2026, 3, 3),
        raw_text="""FORM 8-K

Item 1.01 Entry into a Material Definitive Agreement
On March 3, 2026, FakeBio Therapeutics terminated its prior ex-U.S. commercialization option with Alder Peak and entered a replacement co-development and supply agreement with Northwind Biologics for FB-401.
Northwind will pay $18 million upfront at closing and reimburse transition costs up to $6 million.
Closing is subject to transfer of manufacturing records, antitrust clearance, and assignment of certain third-party licenses.
If closing is delayed beyond June 30, 2026, either party may terminate.

Operational Effects
FakeBio will move drug-substance manufacturing from its internal pilot plant to Northwind's commercial network and expects one quarter of transition work.
The company paused enrollment expansion in Cohort B until comparability data are reviewed.

Next Steps
Management expects to file a Form 8-K amendment with schedules after confidentiality review.
Forward-looking statements apply.""",
    ),
    DocumentType.CLINICAL_TRIAL_UPDATE: make_worker_demo_selected_document(
        document_id="prompt6_demo_clinical_trial",
        document_type=DocumentType.CLINICAL_TRIAL_UPDATE,
        title="FakeBio FB-201 Phase 2 Registry Update",
        ticker="FAKE",
        source_name="Mock Clinical Trial Registry",
        source_url="https://example.com/prompt6/clinical-trial-update",
        source_identifier="prompt6-trial-update-001",
        source_family=SourceFamily.OFFICIAL_REGULATORY,
        is_structured_source=True,
        published_at=datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc),
        event_date=date(2026, 3, 2),
        raw_text="""Study Record
Study Title: Randomized Phase 2 Study of FB-201 in Recurrent Disease
Overall Recruitment Status: Active, not recruiting
Last Update Posted: March 2, 2026

Enrollment
Actual Enrollment: 84
Planned Enrollment: 120
Primary Completion Date: December 2026

Study Design
Randomized, open-label, dose-optimization expansion added in Cohort 3.
Steroid washout requirement clarified to 14 days.

Outcome Measures
Primary Outcome: Change from baseline in biomarker score at Week 24
Secondary Outcomes: Confirmed objective response rate through Month 9; durability follow-up through Month 12
No results have been posted.

Contacts and Locations
Boston site withdrawn.
Houston and Denver sites listed as recruiting pending IRB activation.
Imaging schedule updated in the protocol record.""",
    ),
    DocumentType.FDA_REVIEW: make_worker_demo_selected_document(
        document_id="prompt6_demo_fda_review",
        document_type=DocumentType.FDA_REVIEW,
        title="FDA Multidisciplinary Review Excerpt for FB-401",
        ticker="FAKE",
        source_name="Mock FDA Review Archive",
        source_url="https://example.com/prompt6/fda-review",
        source_identifier="prompt6-fda-review-001",
        source_family=SourceFamily.OFFICIAL_REGULATORY,
        is_structured_source=True,
        published_at=datetime(2026, 2, 28, 8, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 2, 28, 8, 30, tzinfo=timezone.utc),
        event_date=date(2026, 2, 28),
        raw_text="""FDA Multidisciplinary Review

Clinical Review
Study 301 met the primary endpoint, but confirmatory support from subgroup analyses remains limited.
The review team noted a clinically meaningful response pattern in biomarker-positive patients.

Safety Review
Grade 3 hepatic laboratory abnormalities occurred in 8% of treated patients versus 2% in control.
The review recommends dose interruption and monitoring language in labeling.

CMC Review
Commercial process validation lots were representative, but one comparability package for a post-approval manufacturing site remains under review.

Labeling Considerations
The proposed indication should be limited to biomarker-positive adult patients after prior therapy.
First-line use is not supported by the submitted data.

Action Timing
No late-cycle meeting issue was identified, but approval timing depends on closure of the manufacturing comparability item.""",
    ),
    DocumentType.FINANCING_DILUTION: make_worker_demo_selected_document(
        document_id="prompt6_demo_financing",
        document_type=DocumentType.FINANCING_DILUTION,
        title="FakeBio Registered Direct Offering and Warrant Financing",
        ticker="FAKE",
        source_name="Mock SEC Filing Feed",
        source_url="https://example.com/prompt6/financing",
        source_identifier="prompt6-financing-001",
        source_family=SourceFamily.OFFICIAL_REGULATORY,
        is_structured_source=True,
        published_at=datetime(2026, 3, 6, 16, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 6, 16, 25, tzinfo=timezone.utc),
        event_date=date(2026, 3, 6),
        raw_text="""Item 1.01 Securities Purchase Agreement
FakeBio Therapeutics entered into a registered direct offering with two healthcare-focused investors.
The company agreed to sell 12.5 million common shares and prefunded warrants for 5.0 million shares at a combined purchase price of $3.20.
Investors also received five-year warrants for 50% of the purchased share count with an exercise price of $4.10.
Gross proceeds are expected to be approximately $56 million before placement agent fees and expenses.

Use of Proceeds
Net proceeds are expected to support Phase 2 execution, process validation work, and general corporate purposes.

Closing Conditions and Restrictions
Closing is expected on March 9, 2026, subject to customary closing conditions.
The company agreed not to issue variable-rate securities for 45 days after closing.
Management stated that the financing extends runway into the third quarter of 2027 under the current operating plan.""",
    ),
    DocumentType.INVESTOR_COMMUNICATION: make_worker_demo_selected_document(
        document_id="prompt6_demo_investor_update",
        document_type=DocumentType.INVESTOR_COMMUNICATION,
        title="FakeBio Corporate Update Transcript",
        ticker="FAKE",
        source_name="Mock Investor Events Page",
        source_url="https://example.com/prompt6/investor-update",
        source_identifier="prompt6-investor-update-001",
        source_family=SourceFamily.ISSUER_PUBLISHED,
        is_structured_source=False,
        published_at=datetime(2026, 3, 5, 13, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 5, 13, 10, tzinfo=timezone.utc),
        event_date=date(2026, 3, 5),
        raw_text="""FakeBio Therapeutics Corporate Update Transcript

Chief Executive Officer:
We are entering 2026 with three priorities: finish site activation, complete process validation, and prepare an end-of-phase 2 package.
The main change since last quarter is slower-than-planned activation at two European sites, partly offset by stronger U.S. screening.

Slide 7 Milestones
Top-line data in the fourth quarter of 2026.
Regulatory meeting by year-end.
Manufacturing readiness work continues in parallel.

Chief Financial Officer:
Cash runway extends into the third quarter of 2027 after the March financing.
We are sequencing spend to support trial execution and CMC readiness.

Analyst Question:
What are the biggest remaining risks?

Chief Executive Officer:
Enrollment consistency, assay turnaround time, and manufacturing comparability remain the main execution risks.
We are not assuming partnership revenue in the current operating plan.""",
    ),
}


# ---------------------------------------------------------------------------
# Cell 93 – arbiter demo evidence / worker-output helpers
# ---------------------------------------------------------------------------

def make_demo_arbiter_evidence(
    *,
    document_id: str,
    chunk_id: str,
    rationale: str,
    snippet_text: str,
    interpretation: EvidenceInterpretation,
    source_url: str,
    section_title: str = "Demo Section",
) -> EvidenceSnippet:
    return EvidenceSnippet(
        evidence_id=f"{document_id}_{chunk_id}",
        document_id=document_id,
        source_url=source_url,
        source_chunk_id=chunk_id,
        source_section_id=f"{chunk_id}_section",
        section_title=section_title,
        evidence_type=EvidenceType.CONTEXTUAL_SUMMARY,
        supported_dimensions=[AnalysisDimension.SENTIMENT, AnalysisDimension.CLARITY],
        interpretation=interpretation,
        snippet_text=snippet_text,
        rationale=rationale,
        metadata={"prompt7_demo": True},
    )


def make_demo_worker_output(
    *,
    worker_name: str,
    document_type: DocumentType,
    status: ProcessingStatus,
    summary: str | None,
    sentiment_label: SentimentLabel | None = None,
    sentiment_score: float | None = None,
    confidence: float | None = None,
    key_points: Sequence[str] | None = None,
    caveats: Sequence[str] | None = None,
    evidence_specs: Sequence[dict[str, Any]] | None = None,
    fogging_score: float | None = None,
    hedging_score: float | None = None,
    promotional_score: float | None = None,
    tone_confidence: float | None = None,
) -> WorkerOutput:
    evidence = [make_demo_arbiter_evidence(**evidence_spec) for evidence_spec in (evidence_specs or [])]
    sentiment = None
    if sentiment_label is not None:
        sentiment = SentimentAssessment(label=sentiment_label, score=sentiment_score, confidence=confidence, rationale="Synthetic Prompt 7 worker-demo sentiment.")
    tone = None
    if any(value is not None for value in [fogging_score, hedging_score, promotional_score, tone_confidence]):
        tone = ToneAssessment(fogging_score=fogging_score, hedging_score=hedging_score, promotional_score=promotional_score, confidence=tone_confidence, rationale="Synthetic Prompt 7 worker-demo tone.")
    return WorkerOutput(worker_name=worker_name, document_type=document_type, status=status, summary=summary, sentiment=sentiment, tone=tone, key_points=list(key_points or []), evidence=evidence, caveats=list(caveats or []), issues=[], confidence=confidence)


# ---------------------------------------------------------------------------
# Cell 93 – arbiter demo cases
# ---------------------------------------------------------------------------

ARBITER_DEMO_CASES: dict[str, list[WorkerOutput]] = {
    "mostly_aligned": [
        make_demo_worker_output(
            worker_name="material_event_worker",
            document_type=DocumentType.MATERIAL_EVENT,
            status=ProcessingStatus.SUCCESS,
            summary="The 8-K frames the catalyst as on track and adds concrete next steps.",
            sentiment_label=SentimentLabel.POSITIVE,
            sentiment_score=0.55,
            confidence=0.78,
            key_points=["Company disclosed a partner opt-in milestone.", "Near-term execution steps were made explicit."],
            caveats=["Manufacturing readiness still needs execution follow-through."],
            evidence_specs=[
                {"document_id": "aligned_material_001", "chunk_id": "material_chunk_1", "rationale": "8-K provides concrete milestone detail.", "snippet_text": "Partner opted in after reviewing the updated package.", "interpretation": EvidenceInterpretation.POSITIVE, "source_url": "https://example.com/aligned/material"},
                {"document_id": "aligned_material_001", "chunk_id": "material_chunk_2", "rationale": "Next operational steps are explicitly listed.", "snippet_text": "Site activation and process validation continue in parallel.", "interpretation": EvidenceInterpretation.CLARIFYING, "source_url": "https://example.com/aligned/material"},
            ],
            fogging_score=0.15,
            hedging_score=0.20,
            promotional_score=0.25,
            tone_confidence=0.70,
        ),
        make_demo_worker_output(
            worker_name="clinical_trial_update_worker",
            document_type=DocumentType.CLINICAL_TRIAL_UPDATE,
            status=ProcessingStatus.SUCCESS,
            summary="Registry update clarifies enrollment status without pushing out the primary completion date.",
            sentiment_label=SentimentLabel.POSITIVE,
            sentiment_score=0.45,
            confidence=0.82,
            key_points=["Enrollment remains active.", "Primary completion timing was preserved."],
            evidence_specs=[
                {"document_id": "aligned_clinical_001", "chunk_id": "clinical_chunk_1", "rationale": "Registry confirms continued enrollment.", "snippet_text": "Estimated primary completion remains December 2026.", "interpretation": EvidenceInterpretation.POSITIVE, "source_url": "https://example.com/aligned/clinical"},
                {"document_id": "aligned_clinical_001", "chunk_id": "clinical_chunk_2", "rationale": "Endpoint and enrollment fields were clarified.", "snippet_text": "Secondary biomarker analysis plan was updated for clarity.", "interpretation": EvidenceInterpretation.CLARIFYING, "source_url": "https://example.com/aligned/clinical"},
            ],
            fogging_score=0.10,
            hedging_score=0.18,
            promotional_score=0.10,
            tone_confidence=0.75,
        ),
        make_demo_worker_output(
            worker_name="fda_review_worker",
            document_type=DocumentType.FDA_REVIEW,
            status=ProcessingStatus.SUCCESS,
            summary="Regulatory materials are broadly supportive but still preserve label and CMC follow-up items.",
            sentiment_label=SentimentLabel.POSITIVE,
            sentiment_score=0.30,
            confidence=0.76,
            key_points=["Benefit-risk framing remains supportive.", "Outstanding items look manageable rather than thesis-breaking."],
            caveats=["Label scope still depends on ongoing discussion."],
            evidence_specs=[
                {"document_id": "aligned_fda_001", "chunk_id": "fda_chunk_1", "rationale": "Review package supports benefit-risk balance.", "snippet_text": "The efficacy package supports a favorable benefit-risk assessment.", "interpretation": EvidenceInterpretation.POSITIVE, "source_url": "https://example.com/aligned/fda"},
                {"document_id": "aligned_fda_001", "chunk_id": "fda_chunk_2", "rationale": "Remaining issues are specific and bounded.", "snippet_text": "CMC validation data will be reviewed in the final labeling cycle.", "interpretation": EvidenceInterpretation.UNCERTAINTY, "source_url": "https://example.com/aligned/fda"},
            ],
            fogging_score=0.12,
            hedging_score=0.28,
            promotional_score=0.08,
            tone_confidence=0.72,
        ),
        make_demo_worker_output(
            worker_name="financing_dilution_worker",
            document_type=DocumentType.FINANCING_DILUTION,
            status=ProcessingStatus.SUCCESS,
            summary="Financing looks opportunistic rather than distressed and extends runway into the data window.",
            sentiment_label=SentimentLabel.POSITIVE,
            sentiment_score=0.28,
            confidence=0.74,
            key_points=["Runway extends beyond expected catalyst timing.", "Use of proceeds is tied to trial and CMC execution."],
            evidence_specs=[
                {"document_id": "aligned_financing_001", "chunk_id": "financing_chunk_1", "rationale": "Runway extension reduces near-term financing pressure.", "snippet_text": "Cash runway extends into the third quarter of 2027.", "interpretation": EvidenceInterpretation.POSITIVE, "source_url": "https://example.com/aligned/financing"},
                {"document_id": "aligned_financing_001", "chunk_id": "financing_chunk_2", "rationale": "Use of proceeds is disclosed concretely.", "snippet_text": "Proceeds support enrollment, assay work, and process validation.", "interpretation": EvidenceInterpretation.CLARIFYING, "source_url": "https://example.com/aligned/financing"},
            ],
            fogging_score=0.18,
            hedging_score=0.15,
            promotional_score=0.18,
            tone_confidence=0.68,
        ),
        make_demo_worker_output(
            worker_name="investor_communication_worker",
            document_type=DocumentType.INVESTOR_COMMUNICATION,
            status=ProcessingStatus.SUCCESS,
            summary="Management narrative matches the harder documents reasonably well and keeps remaining risks visible.",
            sentiment_label=SentimentLabel.POSITIVE,
            sentiment_score=0.35,
            confidence=0.63,
            key_points=["Management emphasizes execution milestones already disclosed elsewhere.", "Risk section still mentions enrollment and manufacturing dependencies."],
            caveats=["Narrative remains somewhat polished."],
            evidence_specs=[
                {"document_id": "aligned_investor_001", "chunk_id": "investor_chunk_1", "rationale": "Investor communication repeats the same core milestones.", "snippet_text": "Top-line data remains expected in the fourth quarter of 2026.", "interpretation": EvidenceInterpretation.POSITIVE, "source_url": "https://example.com/aligned/investor"},
                {"document_id": "aligned_investor_001", "chunk_id": "investor_chunk_2", "rationale": "Management still discloses key execution risks.", "snippet_text": "Enrollment consistency and manufacturing comparability remain important risks.", "interpretation": EvidenceInterpretation.CLARIFYING, "source_url": "https://example.com/aligned/investor"},
            ],
            fogging_score=0.32,
            hedging_score=0.30,
            promotional_score=0.38,
            tone_confidence=0.60,
        ),
    ],
    "contradictory": [
        make_demo_worker_output(
            worker_name="material_event_worker",
            document_type=DocumentType.MATERIAL_EVENT,
            status=ProcessingStatus.SUCCESS,
            summary="Press release claims the program remains on track and highlights upcoming milestones.",
            sentiment_label=SentimentLabel.POSITIVE,
            sentiment_score=0.60,
            confidence=0.72,
            key_points=["Management presents timing as intact."],
            evidence_specs=[{"document_id": "contradict_material_001", "chunk_id": "material_chunk_1", "rationale": "Issuer framing remains favorable.", "snippet_text": "The company remains confident in the regulatory path.", "interpretation": EvidenceInterpretation.POSITIVE, "source_url": "https://example.com/contradict/material"}],
            fogging_score=0.35,
            hedging_score=0.25,
            promotional_score=0.52,
            tone_confidence=0.66,
        ),
        make_demo_worker_output(
            worker_name="clinical_trial_update_worker",
            document_type=DocumentType.CLINICAL_TRIAL_UPDATE,
            status=ProcessingStatus.SUCCESS,
            summary="Registry update shows slower enrollment and pushes out primary completion timing.",
            sentiment_label=SentimentLabel.NEGATIVE,
            sentiment_score=-0.55,
            confidence=0.85,
            key_points=["Enrollment target was reduced.", "Primary completion moved out by two quarters."],
            evidence_specs=[
                {"document_id": "contradict_clinical_001", "chunk_id": "clinical_chunk_1", "rationale": "Registry timing slipped materially.", "snippet_text": "Estimated primary completion moved from Q4 2026 to Q2 2027.", "interpretation": EvidenceInterpretation.NEGATIVE, "source_url": "https://example.com/contradict/clinical"},
                {"document_id": "contradict_clinical_001", "chunk_id": "clinical_chunk_2", "rationale": "Enrollment assumptions weakened.", "snippet_text": "Target enrollment was revised downward after slower site activation.", "interpretation": EvidenceInterpretation.NEGATIVE, "source_url": "https://example.com/contradict/clinical"},
            ],
            fogging_score=0.12,
            hedging_score=0.20,
            promotional_score=0.08,
            tone_confidence=0.80,
        ),
        make_demo_worker_output(
            worker_name="fda_review_worker",
            document_type=DocumentType.FDA_REVIEW,
            status=ProcessingStatus.SUCCESS,
            summary="Regulatory materials identify unresolved safety and CMC issues that could constrain timing and label scope.",
            sentiment_label=SentimentLabel.NEGATIVE,
            sentiment_score=-0.70,
            confidence=0.88,
            key_points=["Safety signal remains under review.", "CMC comparability package is incomplete."],
            caveats=["Benefit-risk is not yet clearly resolved."],
            evidence_specs=[
                {"document_id": "contradict_fda_001", "chunk_id": "fda_chunk_1", "rationale": "Review documents highlight unresolved risk.", "snippet_text": "Outstanding safety concerns require additional analysis before approval.", "interpretation": EvidenceInterpretation.NEGATIVE, "source_url": "https://example.com/contradict/fda"},
                {"document_id": "contradict_fda_001", "chunk_id": "fda_chunk_2", "rationale": "Manufacturing package is not yet complete.", "snippet_text": "CMC comparability remains an approval-critical deficiency.", "interpretation": EvidenceInterpretation.NEGATIVE, "source_url": "https://example.com/contradict/fda"},
            ],
            fogging_score=0.10,
            hedging_score=0.22,
            promotional_score=0.05,
            tone_confidence=0.82,
        ),
        make_demo_worker_output(
            worker_name="financing_dilution_worker",
            document_type=DocumentType.FINANCING_DILUTION,
            status=ProcessingStatus.SUCCESS,
            summary="Financing looks urgent, highly dilutive, and priced from a weak bargaining position.",
            sentiment_label=SentimentLabel.NEGATIVE,
            sentiment_score=-0.60,
            confidence=0.79,
            key_points=["Discount is deep relative to prior trading levels.", "Warrant coverage increases future overhang."],
            evidence_specs=[
                {"document_id": "contradict_financing_001", "chunk_id": "financing_chunk_1", "rationale": "Financing terms are punitive.", "snippet_text": "Offering priced at a 22 percent discount with full warrant coverage.", "interpretation": EvidenceInterpretation.NEGATIVE, "source_url": "https://example.com/contradict/financing"},
                {"document_id": "contradict_financing_001", "chunk_id": "financing_chunk_2", "rationale": "Use of proceeds prioritizes near-term liquidity rather than optionality.", "snippet_text": "Proceeds are needed primarily to maintain ongoing operations.", "interpretation": EvidenceInterpretation.NEGATIVE, "source_url": "https://example.com/contradict/financing"},
            ],
            fogging_score=0.18,
            hedging_score=0.12,
            promotional_score=0.14,
            tone_confidence=0.74,
        ),
        make_demo_worker_output(
            worker_name="investor_communication_worker",
            document_type=DocumentType.INVESTOR_COMMUNICATION,
            status=ProcessingStatus.SUCCESS,
            summary="Management narrative stays strongly upbeat and emphasizes upside despite pressure in harder disclosures.",
            sentiment_label=SentimentLabel.POSITIVE,
            sentiment_score=0.70,
            confidence=0.68,
            key_points=["Management emphasizes catalyst potential and de-emphasizes timing risk.", "Presentation highlights broad market opportunity rather than current constraints."],
            evidence_specs=[{"document_id": "contradict_investor_001", "chunk_id": "investor_chunk_1", "rationale": "Investor narrative remains strongly promotional.", "snippet_text": "We remain uniquely positioned for a best-in-class launch trajectory.", "interpretation": EvidenceInterpretation.PROMOTIONAL, "source_url": "https://example.com/contradict/investor"}],
            fogging_score=0.62,
            hedging_score=0.40,
            promotional_score=0.78,
            tone_confidence=0.67,
        ),
    ],
    "sparse_missing": [
        make_demo_worker_output(worker_name="material_event_worker", document_type=DocumentType.MATERIAL_EVENT, status=ProcessingStatus.NO_DOCUMENT, summary=None, sentiment_label=None, confidence=None, key_points=[], caveats=["No current material-event document was retrieved."]),
        make_demo_worker_output(
            worker_name="financing_dilution_worker",
            document_type=DocumentType.FINANCING_DILUTION,
            status=ProcessingStatus.PARTIAL,
            summary="Financing disclosure exists, but terms and use-of-proceeds detail are thin.",
            sentiment_label=SentimentLabel.INSUFFICIENT_EVIDENCE,
            sentiment_score=None,
            confidence=0.38,
            key_points=["Runway benefit cannot be sized confidently from the available text."],
            caveats=["Material financing terms are incomplete in the available excerpt."],
            evidence_specs=[{"document_id": "sparse_financing_001", "chunk_id": "financing_chunk_1", "rationale": "Only limited financing detail is available.", "snippet_text": "The company completed a registered direct offering.", "interpretation": EvidenceInterpretation.UNCERTAINTY, "source_url": "https://example.com/sparse/financing"}],
            fogging_score=0.35,
            hedging_score=0.45,
            promotional_score=0.20,
            tone_confidence=0.40,
        ),
        make_demo_worker_output(
            worker_name="investor_communication_worker",
            document_type=DocumentType.INVESTOR_COMMUNICATION,
            status=ProcessingStatus.SUCCESS,
            summary="Management remains optimistic, but the communication is light on inspectable support.",
            sentiment_label=SentimentLabel.POSITIVE,
            sentiment_score=0.20,
            confidence=0.31,
            key_points=["Narrative reiterates milestones without adding concrete support."],
            evidence_specs=[{"document_id": "sparse_investor_001", "chunk_id": "investor_chunk_1", "rationale": "Narrative remains optimistic with limited detail.", "snippet_text": "We see multiple value-inflection opportunities ahead.", "interpretation": EvidenceInterpretation.PROMOTIONAL, "source_url": "https://example.com/sparse/investor"}],
            fogging_score=0.58,
            hedging_score=0.42,
            promotional_score=0.72,
            tone_confidence=0.36,
        ),
    ],
}


# ---------------------------------------------------------------------------
# Cell 93 – pre-built arbiter outputs for each demo case
# ---------------------------------------------------------------------------

def _build_demo_arbiter_outputs() -> dict[str, ArbiterOutput]:
    """Run the cross-document arbiter over each demo case at import time."""
    cross_document_arbiter = CrossDocumentArbiter()
    outputs: dict[str, ArbiterOutput] = {}
    for case_name, worker_outputs in ARBITER_DEMO_CASES.items():
        arbiter_input = ArbiterInput(
            run_id=f"prompt7_demo_{case_name}",
            ticker="FAKE",
            worker_outputs=worker_outputs,
            retrieval_results=[],
            config=PIPELINE_CONFIG,
        )
        arbiter_output = cross_document_arbiter.arbitrate(arbiter_input)
        outputs[case_name] = arbiter_output
    return outputs


DEMO_ARBITER_OUTPUTS: dict[str, ArbiterOutput] = _build_demo_arbiter_outputs()
