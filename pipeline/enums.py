"""All enumeration types used across the pipeline."""
from __future__ import annotations

from enum import Enum


class DocumentType(str, Enum):
    """Disclosure types tracked by the pipeline."""

    MATERIAL_EVENT = "material_event"
    CLINICAL_TRIAL_UPDATE = "clinical_trial_update"
    FDA_REVIEW = "fda_review"
    FINANCING_DILUTION = "financing_dilution"
    INVESTOR_COMMUNICATION = "investor_communication"


class SentimentLabel(str, Enum):
    """Discrete sentiment categories."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ProcessingStatus(str, Enum):
    """Pipeline processing status values."""

    PENDING = "pending"
    SUCCESS = "success"
    PARTIAL = "partial"
    NO_DOCUMENT = "no_document"
    RETRIEVAL_FAILED = "retrieval_failed"
    SELECTION_FAILED = "selection_failed"
    EXTRACTION_FAILED = "extraction_failed"
    ANALYSIS_FAILED = "analysis_failed"


class SourceFamily(str, Enum):
    """Source provenance categories for retrieval candidates."""

    OFFICIAL_REGULATORY = "official_regulatory"
    ISSUER_PUBLISHED = "issuer_published"
    PERMITTED_SECONDARY = "permitted_secondary"
    UNKNOWN = "unknown"


class ArbiterKind(str, Enum):
    """Supported arbiter responsibilities for the initial node layout."""

    SUMMARY = "summary"
    SENTIMENT = "sentiment"
    CROSS_DOCUMENT = "cross_document"


class AnalysisTier(str, Enum):
    """Controls the compute expenditure tier for a pipeline run."""

    TEXT_ONLY = "text_only"
    VECTOR_GRAPH = "vector_graph"


class ProcessingNoteSeverity(str, Enum):
    """Severity levels for deterministic processing warnings and notes."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class SectionKind(str, Enum):
    """High-level structural section types detected from cleaned text."""

    NARRATIVE = "narrative"
    LIST = "list"
    TABLE = "table"
    TRANSCRIPT = "transcript"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers for the notebook prototype."""

    VOYAGE = "voyage"


class EmbeddingStatus(str, Enum):
    """Embedding generation status values."""

    SUCCESS = "success"
    MOCK = "mock"
    SKIPPED = "skipped"
    FAILED = "failed"


class GraphNodeType(str, Enum):
    """Node types used by the lightweight single-document graph."""

    DOCUMENT = "document"
    SECTION = "section"
    CHUNK = "chunk"
    ANCHOR = "anchor"


class GraphEdgeType(str, Enum):
    """Edge types used by the lightweight single-document graph."""

    CONTAINS = "contains"
    ADJACENT = "adjacent"
    SAME_SECTION = "same_section"
    SECTION_SEQUENCE = "section_sequence"
    CROSS_REFERENCE = "cross_reference"


class AnalysisDimension(str, Enum):
    """Shared analysis rubric dimensions."""

    SENTIMENT = "sentiment"
    UNCERTAINTY = "uncertainty"
    FOGGING = "fogging"
    HEDGING = "hedging"
    PROMOTIONAL_TONE = "promotional_tone"
    CLARITY = "clarity"
    MATERIALITY = "materiality"
    COMPLETENESS = "completeness"


class EvidenceType(str, Enum):
    """Types of evidence linked to analysis findings."""

    DIRECT_QUOTE = "direct_quote"
    NUMERIC_DETAIL = "numeric_detail"
    STATUS_UPDATE = "status_update"
    CONTEXTUAL_SUMMARY = "contextual_summary"
    CROSS_REFERENCE = "cross_reference"


class EvidenceInterpretation(str, Enum):
    """How one piece of evidence should be read in context."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNCERTAINTY = "uncertainty"
    FOGGING = "fogging"
    HEDGING = "hedging"
    PROMOTIONAL = "promotional"
    CLARIFYING = "clarifying"
    MATERIAL = "material"


class AnalysisIssueSeverity(str, Enum):
    """Severity for analysis-layer issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NormalizedSignalDirection(str, Enum):
    """Directional signal assigned during arbiter normalization."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNCERTAIN = "uncertain"


class ArbiterSignalCategory(str, Enum):
    """Categories for structured arbiter signals."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAINTY = "uncertainty"
    FOGGING = "fogging"
    POSITIVE_SIGNAL = "positive_signal"
    NEGATIVE_SIGNAL = "negative_signal"
    CONFLICTING_SIGNAL = "conflicting_signal"
    UNRESOLVED_UNCERTAINTY = "unresolved_uncertainty"
    FOGGING_FLAG = "fogging_flag"


class CrossDocumentTheme(str, Enum):
    """Cross-document analytical themes tracked by the arbiter."""

    CLINICAL_EXECUTION = "clinical_execution"
    REGULATORY_POSTURE = "regulatory_posture"
    FINANCING_AND_RUNWAY = "financing_and_runway"
    OPERATIONAL_EXECUTION = "operational_execution"
    NARRATIVE_CREDIBILITY = "narrative_credibility"
    REGULATORY_PATHWAY = "regulatory_pathway"
    CLINICAL_EVIDENCE = "clinical_evidence"
    COMMERCIAL_STRATEGY = "commercial_strategy"
    CAPITAL_STRUCTURE = "capital_structure"
    PIPELINE_EXECUTION = "pipeline_execution"
    MANAGEMENT_CREDIBILITY = "management_credibility"


class ArbiterDecisionType(str, Enum):
    """Decision categories for cross-document theme judgments."""

    ALIGNED_SIGNAL = "aligned_signal"
    CONTRADICTORY_SIGNAL = "contradictory_signal"
    CROSS_DOCUMENT_UNCERTAINTY = "cross_document_uncertainty"
    UNRESOLVED_AMBIGUITY = "unresolved_ambiguity"
    MATERIAL_CONCERN = "material_concern"
    MATERIAL_POSITIVE = "material_positive"
    STORY_SUBSTANCE_MISMATCH = "story_substance_mismatch"
