"""Text normalization functions for document processing."""

from __future__ import annotations

import re
from typing import Any

from pipeline.config import PROCESSING_CONFIG, logger
from pipeline.enums import ProcessingNoteSeverity
from pipeline.models import ProcessingNote

__all__ = [
    "HEADER_PREFIX_PATTERN",
    "NUMBERED_HEADER_PATTERN",
    "SPEAKER_LABEL_PATTERN",
    "BULLET_START_PATTERN",
    "TABLE_SPLIT_PATTERN",
    "PAGE_MARKER_PATTERN",
    "SEPARATOR_LINE_PATTERN",
    "make_processing_note",
    "approximate_word_count",
    "normalize_line_breaks",
    "normalize_whitespace",
    "normalize_bullets_and_lists",
    "normalize_table_like_text",
    "strip_noise_blocks",
    "is_probable_header_line",
    "preserve_meaningful_headers",
    "safe_text_cleanup",
    "normalize_document_text",
]

# ---------------------------------------------------------------------------
# Regex pattern constants
# ---------------------------------------------------------------------------

HEADER_PREFIX_PATTERN = re.compile(
    r"^(item|section|part|article)\s+[A-Za-z0-9IVXivx.\-]+", re.IGNORECASE
)
NUMBERED_HEADER_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+){0,3}|[A-Z]\.|[IVXLC]+\.)\s+\S"
)
SPEAKER_LABEL_PATTERN = re.compile(r"^[A-Z][A-Za-z .&/\-]{1,40}:\s")
BULLET_START_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)]|[•◦▪●])\s+")
TABLE_SPLIT_PATTERN = re.compile(r"(?:\t+|\s{2,})")
PAGE_MARKER_PATTERN = re.compile(
    r"^(page|slide)\s+\d+(?:\s+of\s+\d+)?$", re.IGNORECASE
)
SEPARATOR_LINE_PATTERN = re.compile(r"^[\-=*_]{3,}$")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_processing_note(
    stage: str,
    message: str,
    *,
    severity: ProcessingNoteSeverity,
    document_id: str | None = None,
    chunk_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProcessingNote:
    """Factory for creating ProcessingNote instances with sensible defaults."""
    effective_metadata = metadata or {}
    if chunk_id is not None:
        effective_metadata.setdefault("chunk_id", chunk_id)
    return ProcessingNote(
        stage=stage,
        message=message,
        severity=severity,
        document_id=document_id,
        metadata=effective_metadata,
    )


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def approximate_word_count(text: str) -> int:
    """Approximate word count using a simple alphanumeric token regex."""
    return len(re.findall(r"\b\w+\b", text))


def normalize_line_breaks(text: str) -> str:
    """Standardize line breaks without removing paragraph boundaries."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def normalize_whitespace(text: str) -> str:
    """Trim trailing spaces and collapse excessive blank lines conservatively."""
    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.rstrip()
        if " | " in stripped:
            cleaned_lines.append(stripped)
        else:
            cleaned_lines.append(re.sub(r" {2,}", " ", stripped))
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def normalize_bullets_and_lists(text: str) -> str:
    """Standardize common bullet characters while preserving list intent."""
    normalized_lines: list[str] = []
    for line in text.split("\n"):
        updated = re.sub(r"^\s*[•◦▪●]\s*", "- ", line)
        updated = re.sub(r"^\s*(\d+)\)\s+", r"\1. ", updated)
        normalized_lines.append(updated)
    return "\n".join(normalized_lines)


def normalize_table_like_text(text: str) -> str:
    """Preserve simple table-like rows by converting wide spacing into pipe separators."""
    normalized_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            normalized_lines.append("")
            continue
        if " | " in stripped or BULLET_START_PATTERN.match(stripped):
            normalized_lines.append(stripped)
            continue
        candidate_parts = [
            part.strip() for part in TABLE_SPLIT_PATTERN.split(stripped) if part.strip()
        ]
        if len(candidate_parts) >= 3 and len(stripped) <= 180 and not stripped.endswith("."):
            normalized_lines.append(" | ".join(candidate_parts))
        else:
            normalized_lines.append(stripped)
    return "\n".join(normalized_lines)


def strip_noise_blocks(
    text: str, document_id: str | None = None
) -> tuple[str, list[ProcessingNote]]:
    """Remove generic layout noise such as page markers and repeated separators."""
    kept_lines: list[str] = []
    notes: list[ProcessingNote] = []
    removed_count = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        if PAGE_MARKER_PATTERN.match(stripped) or SEPARATOR_LINE_PATTERN.match(stripped):
            removed_count += 1
            continue
        if stripped.isdigit() and len(stripped) <= 3:
            removed_count += 1
            continue
        kept_lines.append(line)
    if removed_count:
        notes.append(
            make_processing_note(
                "noise_lines_removed",
                f"Removed {removed_count} generic layout/noise line(s).",
                severity=ProcessingNoteSeverity.INFO,
                document_id=document_id,
                metadata={"removed_count": removed_count},
            )
        )
    return "\n".join(kept_lines), notes


def is_probable_header_line(
    line: str, *, previous_blank: bool = True, next_blank: bool = True
) -> bool:
    """Detect likely section headers using simple, explicit heuristics."""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > PROCESSING_CONFIG.max_header_chars:
        return False
    word_count = approximate_word_count(stripped)
    if word_count == 0 or word_count > PROCESSING_CONFIG.max_header_words:
        return False
    if HEADER_PREFIX_PATTERN.match(stripped) or NUMBERED_HEADER_PATTERN.match(stripped):
        return True
    if stripped.endswith(":") and word_count <= PROCESSING_CONFIG.max_header_words:
        return True
    uppercase_ratio = sum(1 for char in stripped if char.isupper()) / max(
        1, sum(1 for char in stripped if char.isalpha())
    )
    if uppercase_ratio >= 0.65 and previous_blank:
        return True
    if previous_blank and next_blank and word_count <= 8 and not stripped.endswith("."):
        return True
    return False


def preserve_meaningful_headers(text: str) -> str:
    """Ensure probable headers stay visually separated from surrounding paragraphs."""
    lines = text.split("\n")
    output_lines: list[str] = []
    for index, line in enumerate(lines):
        previous_blank = index == 0 or not lines[index - 1].strip()
        next_blank = index == len(lines) - 1 or not lines[index + 1].strip()
        if is_probable_header_line(
            line, previous_blank=previous_blank, next_blank=next_blank
        ):
            if output_lines and output_lines[-1].strip():
                output_lines.append("")
            output_lines.append(line.strip())
            if not next_blank:
                output_lines.append("")
        else:
            output_lines.append(line)
    return "\n".join(output_lines)


def safe_text_cleanup(
    text: str, document_id: str | None = None
) -> tuple[str, list[ProcessingNote]]:
    """Run conservative, inspectable text cleanup without destroying structure."""
    notes: list[ProcessingNote] = []
    if not text or not text.strip():
        return "", [
            make_processing_note(
                "empty_text",
                "Document text is empty after initial inspection.",
                severity=ProcessingNoteSeverity.WARNING,
                document_id=document_id,
            )
        ]

    working_text = normalize_line_breaks(text)
    working_text = normalize_bullets_and_lists(working_text)
    working_text, stripped_notes = strip_noise_blocks(working_text, document_id=document_id)
    notes.extend(stripped_notes)
    if PROCESSING_CONFIG.normalize_tables:
        working_text = normalize_table_like_text(working_text)
    if PROCESSING_CONFIG.preserve_headers:
        working_text = preserve_meaningful_headers(working_text)
    working_text = normalize_whitespace(working_text)

    notes.append(
        make_processing_note(
            "text_cleanup_complete",
            "Applied deterministic text cleanup and structural preservation steps.",
            severity=ProcessingNoteSeverity.INFO,
            document_id=document_id,
            metadata={
                "raw_char_count": len(text),
                "cleaned_char_count": len(working_text),
            },
        )
    )
    return working_text, notes


def normalize_document_text(raw_text: str | None) -> str | None:
    """Notebook-wide text normalization wrapper used by retrieval and processing."""
    if raw_text is None:
        return None
    cleaned_text, _ = safe_text_cleanup(raw_text)
    return cleaned_text or None
