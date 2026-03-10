"""Section detection and document selection helpers."""

from __future__ import annotations

import re
from typing import Sequence

from pipeline.config import PROCESSING_CONFIG, CHUNKING_CONFIG, logger
from pipeline.enums import ProcessingNoteSeverity, SectionKind
from pipeline.models import (
    DocumentMetadata,
    DocumentSection,
    ProcessedDocument,
    ProcessingNote,
    RetrievalCandidate,
    RetrievalResult,
    SelectedDocument,
)
from pipeline.processing.normalization import (
    BULLET_START_PATTERN,
    HEADER_PREFIX_PATTERN,
    SPEAKER_LABEL_PATTERN,
    approximate_word_count,
    is_probable_header_line,
    make_processing_note,
    safe_text_cleanup,
)
from pipeline.processing.chunking import (
    build_document_chunks,
)
from pipeline.retrieval import (
    build_document_metadata_from_candidate,
)

__all__ = [
    "infer_section_level",
    "extract_reference_label",
    "classify_section_kind",
    "detect_document_sections",
    "selected_document_from_candidate",
    "selected_document_from_retrieval_result",
    "process_selected_document",
]


# ---------------------------------------------------------------------------
# Section-level helpers
# ---------------------------------------------------------------------------


def infer_section_level(title: str) -> int:
    """Infer a lightweight header depth from numbering patterns when present."""
    stripped = title.strip()
    numbered_match = re.match(r"^(\d+(?:\.\d+)*)", stripped)
    if numbered_match:
        return min(4, numbered_match.group(1).count(".") + 1)
    if HEADER_PREFIX_PATTERN.match(stripped):
        return 1
    return 1


def extract_reference_label(title: str) -> str | None:
    """Extract a stable reference label from a section title when possible."""
    stripped = title.strip()
    for pattern in [
        r"^(Item\s+[A-Za-z0-9.]+)",
        r"^(Section\s+[A-Za-z0-9.]+)",
        r"^(\d+(?:\.\d+)*)",
    ]:
        match = re.match(pattern, stripped, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def classify_section_kind(section_text: str) -> SectionKind:
    """Classify a section using explicit, shallow structural heuristics."""
    nonblank_lines = [
        line.strip() for line in section_text.splitlines() if line.strip()
    ]
    if not nonblank_lines:
        return SectionKind.UNKNOWN

    bullet_lines = sum(1 for line in nonblank_lines if BULLET_START_PATTERN.match(line))
    table_lines = sum(1 for line in nonblank_lines if " | " in line)
    speaker_lines = sum(
        1 for line in nonblank_lines if SPEAKER_LABEL_PATTERN.match(line)
    )
    line_count = len(nonblank_lines)

    ratios = {
        SectionKind.LIST: bullet_lines / line_count,
        SectionKind.TABLE: table_lines / line_count,
        SectionKind.TRANSCRIPT: speaker_lines / line_count,
    }
    dominant_kind = max(ratios, key=ratios.get)  # type: ignore[arg-type]
    dominant_ratio = ratios[dominant_kind]

    if dominant_ratio >= 0.45:
        return dominant_kind
    if sum(1 for ratio in ratios.values() if ratio >= 0.20) >= 2:
        return SectionKind.MIXED
    return SectionKind.NARRATIVE


def detect_document_sections(
    selected_document: SelectedDocument,
    cleaned_text: str,
) -> tuple[list[DocumentSection], list[ProcessingNote]]:
    """Detect lightweight sections from cleaned text using explicit header heuristics."""
    notes: list[ProcessingNote] = []
    if not cleaned_text.strip():
        return [], [
            make_processing_note(
                "no_clean_text",
                "No cleaned text was available for section detection.",
                severity=ProcessingNoteSeverity.WARNING,
                document_id=selected_document.document_id,
            )
        ]

    lines = cleaned_text.splitlines()
    header_indices: list[int] = []
    for index, line in enumerate(lines):
        previous_blank = index == 0 or not lines[index - 1].strip()
        next_blank = index == len(lines) - 1 or not lines[index + 1].strip()
        if is_probable_header_line(
            line, previous_blank=previous_blank, next_blank=next_blank
        ):
            header_indices.append(index)

    sections: list[DocumentSection] = []
    if not header_indices:
        notes.append(
            make_processing_note(
                "weak_section_structure",
                "No reliable section headers were detected; a fallback untitled section was created.",
                severity=ProcessingNoteSeverity.WARNING,
                document_id=selected_document.document_id,
            )
        )
        fallback_text = cleaned_text.strip()
        sections.append(
            DocumentSection(
                section_id=f"{selected_document.document_id}::section_001",
                document_id=selected_document.document_id,
                section_index=0,
                title=selected_document.title or "Untitled Document Body",
                section_kind=classify_section_kind(fallback_text),
                level=1,
                reference_label=None,
                line_start=0,
                line_end=max(0, len(lines) - 1),
                raw_text=fallback_text,
                cleaned_text=fallback_text,
                char_count=len(fallback_text),
                word_count=approximate_word_count(fallback_text),
                header_detected=False,
                parent_section_id=None,
            )
        )
        return sections, notes

    section_boundaries = header_indices + [len(lines)]
    section_stack: list[tuple[int, str]] = []
    for section_index, start_index in enumerate(header_indices):
        end_index = section_boundaries[section_index + 1]
        title = lines[start_index].strip()
        body_lines = lines[start_index + 1 : end_index]
        section_text = "\n".join([title] + body_lines).strip()
        level = infer_section_level(title)
        while section_stack and section_stack[-1][0] >= level:
            section_stack.pop()
        parent_section_id = section_stack[-1][1] if section_stack else None
        section_id = f"{selected_document.document_id}::section_{section_index + 1:03d}"
        section = DocumentSection(
            section_id=section_id,
            document_id=selected_document.document_id,
            section_index=section_index,
            title=title,
            section_kind=classify_section_kind(section_text),
            level=level,
            reference_label=extract_reference_label(title),
            line_start=start_index,
            line_end=max(start_index, end_index - 1),
            raw_text=section_text,
            cleaned_text=section_text,
            char_count=len(section_text),
            word_count=approximate_word_count(section_text),
            header_detected=True,
            parent_section_id=parent_section_id,
        )
        sections.append(section)
        section_stack.append((level, section_id))

    notes.append(
        make_processing_note(
            "section_detection_complete",
            f"Detected {len(sections)} section(s) in the cleaned document.",
            severity=ProcessingNoteSeverity.INFO,
            document_id=selected_document.document_id,
            metadata={"section_count": len(sections)},
        )
    )
    return sections, notes


# ---------------------------------------------------------------------------
# Document selection and processing
# ---------------------------------------------------------------------------


def selected_document_from_candidate(candidate: RetrievalCandidate) -> SelectedDocument:
    """Convert a selected retrieval candidate into the shared SelectedDocument schema."""
    metadata = build_document_metadata_from_candidate(candidate)
    return SelectedDocument(
        document_id=candidate.candidate_id,
        document_type=candidate.document_type,
        ticker=candidate.ticker or metadata.ticker,
        title=candidate.title,
        raw_text=candidate.document_text or "",
        source_name=candidate.source_name,
        source_url=candidate.source_url,
        source_identifier=candidate.source_identifier,
        metadata=metadata,
        provenance=candidate.provenance,
        is_mock_data=candidate.is_mock_data,
    )


def selected_document_from_retrieval_result(
    result: RetrievalResult,
) -> SelectedDocument | None:
    """Convert a retrieval result into a SelectedDocument when a winner exists."""
    if result.selected_candidate is None:
        return None
    return selected_document_from_candidate(result.selected_candidate)


def process_selected_document(
    selected_document: SelectedDocument,
    processing_config=None,
    chunking_config=None,
) -> ProcessedDocument:
    """Run deterministic cleanup, section parsing, and chunk building for one selected document."""
    processing_config = processing_config or PROCESSING_CONFIG
    chunking_config = chunking_config or CHUNKING_CONFIG

    cleaned_text, cleanup_notes = safe_text_cleanup(
        selected_document.raw_text, document_id=selected_document.document_id
    )
    sections, section_notes = detect_document_sections(selected_document, cleaned_text)
    chunks, chunk_notes, used_fallback_chunking = build_document_chunks(
        selected_document,
        cleaned_text,
        sections,
        chunking_config,
    )
    processing_notes = cleanup_notes + section_notes + chunk_notes
    if not cleaned_text:
        processing_notes.append(
            make_processing_note(
                "no_clean_text_after_processing",
                "The document remained empty after processing and cannot be chunked reliably.",
                severity=ProcessingNoteSeverity.WARNING,
                document_id=selected_document.document_id,
            )
        )
    return ProcessedDocument(
        document=selected_document,
        cleaned_text=cleaned_text,
        sections=list(sections),
        chunks=chunks,
        processing_notes=processing_notes,
        raw_char_count=len(selected_document.raw_text),
        cleaned_char_count=len(cleaned_text),
        cleaned_word_count=approximate_word_count(cleaned_text),
        section_count=len(sections),
        chunk_count=len(chunks),
        used_fallback_chunking=used_fallback_chunking,
    )
