"""Chunk building functions for document processing."""

from __future__ import annotations

import re
from typing import Sequence

from pipeline.config import CHUNKING_CONFIG, logger
from pipeline.enums import ProcessingNoteSeverity, SectionKind
from pipeline.models import (
    ChunkingConfig,
    DocumentChunk,
    DocumentSection,
    ProcessingNote,
    SelectedDocument,
)
from pipeline.processing.normalization import (
    approximate_word_count,
    make_processing_note,
)

__all__ = [
    "split_section_into_blocks",
    "slice_text_with_overlap",
    "summarize_chunk_context",
    "assign_chunk_ids",
    "build_chunks_from_sections",
    "build_fallback_chunks",
    "build_document_chunks",
]


# ---------------------------------------------------------------------------
# Block splitting
# ---------------------------------------------------------------------------


def split_section_into_blocks(section: DocumentSection) -> list[str]:
    """Split a section into paragraph-like blocks while preserving list/table segments."""
    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n", section.cleaned_text)
        if block.strip()
    ]
    if blocks:
        return blocks
    return [line.strip() for line in section.cleaned_text.splitlines() if line.strip()]


def slice_text_with_overlap(
    text: str, max_chars: int, overlap_chars: int
) -> list[str]:
    """Fallback slicer used when a single block exceeds the preferred chunk size."""
    if len(text) <= max_chars:
        return [text]
    slices: list[str] = []
    start_index = 0
    step_size = max(1, max_chars - overlap_chars)
    while start_index < len(text):
        slice_text = text[start_index : start_index + max_chars].strip()
        if slice_text:
            slices.append(slice_text)
        if start_index + max_chars >= len(text):
            break
        start_index += step_size
    return slices


def summarize_chunk_context(
    chunk_text: str,
    *,
    section_title: str | None = None,
    previous_text: str | None = None,
    next_text: str | None = None,
) -> str:
    """Build a deterministic local-context summary without any model calls."""
    excerpt = chunk_text.replace("\n", " ").strip()[:140]
    parts: list[str] = []
    if section_title:
        parts.append(f"section={section_title}")
    if previous_text:
        parts.append(f"prev={previous_text.replace(chr(10), ' ').strip()[:50]}")
    parts.append(f"excerpt={excerpt}")
    if next_text:
        parts.append(f"next={next_text.replace(chr(10), ' ').strip()[:50]}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Chunk ID assignment
# ---------------------------------------------------------------------------


def assign_chunk_ids(document_id: str, chunk_count: int) -> list[str]:
    """Assign stable chunk ids for one document."""
    return [f"{document_id}::chunk_{index + 1:03d}" for index in range(chunk_count)]


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------


def build_chunks_from_sections(
    selected_document: SelectedDocument,
    sections: Sequence[DocumentSection],
    chunking_config: ChunkingConfig | None = None,
) -> tuple[list[DocumentChunk], list[ProcessingNote]]:
    """Build stable chunks while respecting section boundaries where possible."""
    chunking_config = chunking_config or CHUNKING_CONFIG
    chunks: list[DocumentChunk] = []
    notes: list[ProcessingNote] = []

    for section in sections:
        blocks = split_section_into_blocks(section)
        if not blocks:
            continue

        working_blocks = blocks.copy()
        if (
            chunking_config.include_section_titles
            and section.title
            and not working_blocks[0].startswith(section.title)
        ):
            working_blocks.insert(0, section.title)

        section_chunk_texts: list[str] = []
        current_text = ""
        for block in working_blocks:
            candidate_text = (
                f"{current_text}\n\n{block}".strip() if current_text else block
            )
            if (
                len(candidate_text) <= chunking_config.max_chunk_chars
                or len(current_text) < chunking_config.min_chunk_chars
            ):
                current_text = candidate_text
                continue
            if current_text:
                section_chunk_texts.append(current_text)
            overlap_prefix = (
                current_text[-chunking_config.overlap_chars :].strip()
                if current_text
                else ""
            )
            current_text = (
                f"{overlap_prefix}\n\n{block}".strip() if overlap_prefix else block
            )
            if len(current_text) > chunking_config.max_chunk_chars:
                section_chunk_texts.extend(
                    slice_text_with_overlap(
                        current_text,
                        chunking_config.max_chunk_chars,
                        chunking_config.overlap_chars,
                    )
                )
                current_text = ""
        if current_text:
            section_chunk_texts.append(current_text)

        for order_in_section, chunk_text in enumerate(section_chunk_texts):
            chunks.append(
                DocumentChunk(
                    chunk_id="pending",
                    document_id=selected_document.document_id,
                    document_type=selected_document.document_type,
                    chunk_index=len(chunks),
                    order_in_section=order_in_section,
                    parent_section_id=section.section_id,
                    parent_section_title=section.title,
                    section_kind=section.section_kind,
                    text=chunk_text,
                    char_count=len(chunk_text),
                    word_count=approximate_word_count(chunk_text),
                )
            )

    chunk_ids = assign_chunk_ids(selected_document.document_id, len(chunks))
    for index, chunk in enumerate(chunks):
        chunk.chunk_id = chunk_ids[index]
        chunk.chunk_index = index

    for index, chunk in enumerate(chunks):
        previous_chunk = chunks[index - 1] if index > 0 else None
        next_chunk = chunks[index + 1] if index < len(chunks) - 1 else None
        chunk.previous_chunk_id = previous_chunk.chunk_id if previous_chunk else None
        chunk.next_chunk_id = next_chunk.chunk_id if next_chunk else None
        chunk.context_before = (
            previous_chunk.text[-chunking_config.context_window_chars :]
            if previous_chunk
            else None
        )
        chunk.context_after = (
            next_chunk.text[: chunking_config.context_window_chars]
            if next_chunk
            else None
        )
        chunk.local_context_summary = summarize_chunk_context(
            chunk.text,
            section_title=chunk.parent_section_title,
            previous_text=chunk.context_before,
            next_text=chunk.context_after,
        )

    if chunks:
        notes.append(
            make_processing_note(
                "chunk_build_complete",
                f"Built {len(chunks)} chunk(s) from detected sections.",
                severity=ProcessingNoteSeverity.INFO,
                document_id=selected_document.document_id,
                metadata={"chunk_count": len(chunks)},
            )
        )
    return chunks, notes


def build_fallback_chunks(
    selected_document: SelectedDocument,
    cleaned_text: str,
    chunking_config: ChunkingConfig | None = None,
) -> tuple[list[DocumentChunk], list[ProcessingNote]]:
    """Fallback chunking used when reliable sections are unavailable."""
    chunking_config = chunking_config or CHUNKING_CONFIG
    notes = [
        make_processing_note(
            "fallback_chunking_used",
            "Fallback chunking was used because the document lacked strong section structure.",
            severity=ProcessingNoteSeverity.WARNING,
            document_id=selected_document.document_id,
        )
    ]
    slices = slice_text_with_overlap(
        cleaned_text,
        chunking_config.fallback_chunk_chars,
        chunking_config.overlap_chars,
    )
    synthetic_section_id = f"{selected_document.document_id}::section_fallback"
    chunks: list[DocumentChunk] = []
    for index, chunk_text in enumerate(slices):
        chunks.append(
            DocumentChunk(
                chunk_id=f"{selected_document.document_id}::chunk_{index + 1:03d}",
                document_id=selected_document.document_id,
                document_type=selected_document.document_type,
                chunk_index=index,
                order_in_section=index,
                parent_section_id=synthetic_section_id,
                parent_section_title=selected_document.title or "Fallback Section",
                section_kind=SectionKind.NARRATIVE,
                text=chunk_text,
                char_count=len(chunk_text),
                word_count=approximate_word_count(chunk_text),
            )
        )
    for index, chunk in enumerate(chunks):
        previous_chunk = chunks[index - 1] if index > 0 else None
        next_chunk = chunks[index + 1] if index < len(chunks) - 1 else None
        chunk.previous_chunk_id = previous_chunk.chunk_id if previous_chunk else None
        chunk.next_chunk_id = next_chunk.chunk_id if next_chunk else None
        chunk.context_before = (
            previous_chunk.text[-chunking_config.context_window_chars :]
            if previous_chunk
            else None
        )
        chunk.context_after = (
            next_chunk.text[: chunking_config.context_window_chars]
            if next_chunk
            else None
        )
        chunk.local_context_summary = summarize_chunk_context(
            chunk.text,
            section_title=chunk.parent_section_title,
            previous_text=chunk.context_before,
            next_text=chunk.context_after,
        )
    return chunks, notes


def build_document_chunks(
    selected_document: SelectedDocument,
    cleaned_text: str,
    sections: Sequence[DocumentSection],
    chunking_config: ChunkingConfig | None = None,
) -> tuple[list[DocumentChunk], list[ProcessingNote], bool]:
    """Notebook-wide chunk builder that prefers section-aware chunking before fallback windows."""
    chunking_config = chunking_config or CHUNKING_CONFIG
    if sections:
        chunks, notes = build_chunks_from_sections(
            selected_document, sections, chunking_config
        )
        if chunks:
            return chunks, notes, False
    fallback_chunks, fallback_notes = build_fallback_chunks(
        selected_document, cleaned_text, chunking_config
    )
    return fallback_chunks, fallback_notes, True
