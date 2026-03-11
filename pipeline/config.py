"""Pipeline configuration, logging, environment loading, and shared constants."""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, time, timezone
from pathlib import Path

import dotenv

from pipeline.enums import (
    DocumentType,
    EmbeddingProvider,
    SourceFamily,
)


# --- Logger ---
LOGGER_NAME = "biotech_disclosure_pipeline"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(LOGGER_NAME)

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Global config ---
_config_path = PROJECT_ROOT / "config" / "pipeline_config.json"
if _config_path.exists():
    with open(_config_path, encoding="utf-8") as _f:
        GLOBAL_CONFIG: dict = json.load(_f)
else:
    GLOBAL_CONFIG = {
        "project_name": "AI and Financial Information V2",
        "notebook_version": "0.2.0",
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "data_root": str(PROJECT_ROOT / "data"),
            "raw_documents": str(PROJECT_ROOT / "data" / "raw"),
            "processed_documents": str(PROJECT_ROOT / "data" / "processed"),
            "cache": str(PROJECT_ROOT / "data" / "cache"),
        },
        "disclosure_types": [dt.value for dt in DocumentType],
        "scoring": {
            "sentiment_min": -1.0,
            "sentiment_max": 1.0,
            "tone_min": 0.0,
            "tone_max": 1.0,
        },
        "retrieval_defaults": {
            "max_candidates": 5,
            "min_usable_text_chars": 200,
            "minimum_relevance_score": 40.0,
            "freshness_bonus_max": 20.0,
            "freshness_rank_decay": 4.0,
            "default_source_preferences": [sf.value for sf in SourceFamily],
        },
        "feature_flags": {
            "enable_retrieval": True,
            "enable_embeddings": False,
            "enable_graph_context": False,
            "enable_analysis": True,
            "enable_langchain": False,
            "enable_langgraph": False,
        },
        "future_model_config": {
            "worker_model_name": "moonshot-v1-128k",
            "arbiter_model_name": "moonshot-v1-128k",
            "embedding_model_name": "voyage-3-large",
        },
        "future_source_adapters": {},
    }

# --- Environment variables ---
dotenv.load_dotenv(dotenv.find_dotenv())

SEC_EDGAR_USER_AGENT = os.environ.get("SEC_EDGAR_USER_AGENT", "")
MOONSHOT_API_KEY = os.environ.get("MOONSHOT_API_KEY", "")
OPENFDA_API_KEY = os.environ.get("OPENFDA_API_KEY", "")
VOYAGE_AI_API_KEY = os.environ.get("VOYAGE_AI_API_KEY", "")

# --- Disclosure type registry ---
DISCLOSURE_TYPE_REGISTRY: list[dict[str, str]] = [
    {
        "key": "material_event",
        "label": "Material Event (8-K / Press Release)",
        "description": "SEC 8-K filings, Reg FD press releases, and other material-event disclosures.",
    },
    {
        "key": "clinical_trial_update",
        "label": "Clinical Trial Update (ClinicalTrials.gov)",
        "description": "Registry updates from ClinicalTrials.gov covering study status, design, enrollment, and results.",
    },
    {
        "key": "fda_review",
        "label": "FDA Review Materials",
        "description": "FDA review documents including approval letters, drug labels, and regulatory review packages.",
    },
    {
        "key": "financing_dilution",
        "label": "Financing / Dilution Disclosure",
        "description": "SEC filings related to capital raises, shelf registrations, and dilution-bearing events.",
    },
    {
        "key": "investor_communication",
        "label": "Investor Communication",
        "description": "Investor presentations, earnings call transcripts, shareholder letters, and corporate updates.",
    },
]
DISCLOSURE_TYPE_KEYS = [item["key"] for item in DISCLOSURE_TYPE_REGISTRY]
DISCLOSURE_TYPE_LABELS: dict[str, str] = {item["key"]: item["label"] for item in DISCLOSURE_TYPE_REGISTRY}

# --- Scoring constants ---
SENTIMENT_SCORE_MIN = float(GLOBAL_CONFIG["scoring"]["sentiment_min"])
SENTIMENT_SCORE_MAX = float(GLOBAL_CONFIG["scoring"]["sentiment_max"])
TONE_SCORE_MIN = float(GLOBAL_CONFIG["scoring"]["tone_min"])
TONE_SCORE_MAX = float(GLOBAL_CONFIG["scoring"]["tone_max"])

# --- Retrieval defaults ---
DEFAULT_LANGUAGE = "en"
DEFAULT_SOURCE_FAMILY_ORDER = [
    SourceFamily.OFFICIAL_REGULATORY,
    SourceFamily.ISSUER_PUBLISHED,
    SourceFamily.PERMITTED_SECONDARY,
    SourceFamily.UNKNOWN,
]
DEFAULT_MAX_CANDIDATES = int(GLOBAL_CONFIG["retrieval_defaults"]["max_candidates"])
DEFAULT_MIN_USABLE_TEXT_CHARS = int(GLOBAL_CONFIG["retrieval_defaults"]["min_usable_text_chars"])
DEFAULT_MINIMUM_RELEVANCE_SCORE = float(GLOBAL_CONFIG["retrieval_defaults"]["minimum_relevance_score"])
DEFAULT_FRESHNESS_BONUS_MAX = float(GLOBAL_CONFIG["retrieval_defaults"]["freshness_bonus_max"])
DEFAULT_FRESHNESS_RANK_DECAY = float(GLOBAL_CONFIG["retrieval_defaults"]["freshness_rank_decay"])

MIN_DATETIME_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


def now_utc() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


# --- Processing defaults ---
GLOBAL_CONFIG.setdefault(
    "processing_defaults",
    {
        "max_header_chars": 120,
        "max_header_words": 14,
        "strip_page_markers": True,
        "normalize_tables": True,
        "preserve_headers": True,
        "collapse_blank_lines": True,
    },
)
GLOBAL_CONFIG.setdefault(
    "chunking_defaults",
    {
        "target_chunk_chars": 700,
        "max_chunk_chars": 900,
        "overlap_chars": 120,
        "min_chunk_chars": 160,
        "context_window_chars": 90,
        "include_section_titles": True,
        "fallback_chunk_chars": 650,
    },
)
GLOBAL_CONFIG.setdefault(
    "embedding_defaults",
    {
        "provider": "voyage",
        "enabled": True,
        "model_name": "voyage-3-large",
        "base_url": "https://api.voyageai.com",
        "api_path": "/v1/embeddings",
        "batch_size": 8,
        "request_timeout_seconds": 15.0,
        "normalize_vectors": True,
    },
)
GLOBAL_CONFIG.setdefault(
    "graph_retrieval_defaults",
    {
        "top_k": 3,
        "candidate_pool_size": 6,
        "neighbor_hops": 1,
        "context_expansion_hops": 1,
        "min_similarity_threshold": 0.0,
        "max_graph_bonus": 0.18,
        "section_title_weight": 0.08,
        "neighbor_similarity_weight": 0.06,
        "cross_reference_bonus": 0.03,
    },
)


def _build_pipeline_config():
    """Build PIPELINE_CONFIG lazily to avoid circular imports."""
    from pipeline.models import (
        PipelineConfig,
        ChunkingConfig,
        ProcessingConfig,
        EmbeddingConfig,
        GraphRetrievalConfig,
    )

    pipeline_config = PipelineConfig(
        project_name=GLOBAL_CONFIG["project_name"],
        notebook_version=GLOBAL_CONFIG["notebook_version"],
        project_root=GLOBAL_CONFIG["paths"]["project_root"],
        data_root=GLOBAL_CONFIG["paths"]["data_root"],
        raw_document_dir=GLOBAL_CONFIG["paths"]["raw_documents"],
        processed_document_dir=GLOBAL_CONFIG["paths"]["processed_documents"],
        cache_dir=GLOBAL_CONFIG["paths"]["cache"],
        disclosure_types=[DocumentType(item) for item in GLOBAL_CONFIG["disclosure_types"]],
        sentiment_score_min=GLOBAL_CONFIG["scoring"]["sentiment_min"],
        sentiment_score_max=GLOBAL_CONFIG["scoring"]["sentiment_max"],
        tone_score_min=GLOBAL_CONFIG["scoring"]["tone_min"],
        tone_score_max=GLOBAL_CONFIG["scoring"]["tone_max"],
        default_max_candidates=GLOBAL_CONFIG["retrieval_defaults"]["max_candidates"],
        min_usable_text_chars=GLOBAL_CONFIG["retrieval_defaults"]["min_usable_text_chars"],
        minimum_relevance_score=GLOBAL_CONFIG["retrieval_defaults"]["minimum_relevance_score"],
        freshness_bonus_max=GLOBAL_CONFIG["retrieval_defaults"]["freshness_bonus_max"],
        freshness_rank_decay=GLOBAL_CONFIG["retrieval_defaults"]["freshness_rank_decay"],
        default_source_preferences=[SourceFamily(item) for item in GLOBAL_CONFIG["retrieval_defaults"]["default_source_preferences"]],
        enable_retrieval=GLOBAL_CONFIG["feature_flags"]["enable_retrieval"],
        enable_embeddings=GLOBAL_CONFIG["feature_flags"]["enable_embeddings"],
        enable_graph_context=GLOBAL_CONFIG["feature_flags"]["enable_graph_context"],
        enable_analysis=GLOBAL_CONFIG["feature_flags"]["enable_analysis"],
        enable_langchain=GLOBAL_CONFIG["feature_flags"]["enable_langchain"],
        enable_langgraph=GLOBAL_CONFIG["feature_flags"]["enable_langgraph"],
        worker_model_name=GLOBAL_CONFIG["future_model_config"]["worker_model_name"],
        arbiter_model_name=GLOBAL_CONFIG["future_model_config"]["arbiter_model_name"],
        embedding_model_name=GLOBAL_CONFIG["future_model_config"]["embedding_model_name"],
        source_adapter_placeholders=GLOBAL_CONFIG.get("future_source_adapters", {}),
    )

    chunking_config = ChunkingConfig(**GLOBAL_CONFIG["chunking_defaults"])
    processing_config = ProcessingConfig(**GLOBAL_CONFIG["processing_defaults"])
    embedding_config = EmbeddingConfig(
        provider=EmbeddingProvider(GLOBAL_CONFIG["embedding_defaults"]["provider"]),
        enabled=GLOBAL_CONFIG["embedding_defaults"]["enabled"],
        model_name=GLOBAL_CONFIG["embedding_defaults"]["model_name"],
        base_url=GLOBAL_CONFIG["embedding_defaults"]["base_url"],
        api_path=GLOBAL_CONFIG["embedding_defaults"]["api_path"],
        batch_size=GLOBAL_CONFIG["embedding_defaults"]["batch_size"],
        request_timeout_seconds=GLOBAL_CONFIG["embedding_defaults"]["request_timeout_seconds"],
        normalize_vectors=GLOBAL_CONFIG["embedding_defaults"]["normalize_vectors"],
    )
    graph_retrieval_config = GraphRetrievalConfig(**GLOBAL_CONFIG["graph_retrieval_defaults"])

    return pipeline_config, chunking_config, processing_config, embedding_config, graph_retrieval_config


class _LazyConfigs:
    """Lazy-loaded config instances to avoid circular imports with models.py."""

    def __init__(self):
        self._loaded = False
        self._pipeline_config = None
        self._chunking_config = None
        self._processing_config = None
        self._embedding_config = None
        self._graph_retrieval_config = None

    def _load(self):
        if not self._loaded:
            (
                self._pipeline_config,
                self._chunking_config,
                self._processing_config,
                self._embedding_config,
                self._graph_retrieval_config,
            ) = _build_pipeline_config()
            self._loaded = True

    @property
    def PIPELINE_CONFIG(self):
        self._load()
        return self._pipeline_config

    @property
    def CHUNKING_CONFIG(self):
        self._load()
        return self._chunking_config

    @property
    def PROCESSING_CONFIG(self):
        self._load()
        return self._processing_config

    @property
    def EMBEDDING_CONFIG(self):
        self._load()
        return self._embedding_config

    @property
    def GRAPH_RETRIEVAL_CONFIG(self):
        self._load()
        return self._graph_retrieval_config


_configs = _LazyConfigs()


def __getattr__(name):
    """Module-level __getattr__ for lazy config access."""
    if name in ("PIPELINE_CONFIG", "CHUNKING_CONFIG", "PROCESSING_CONFIG", "EMBEDDING_CONFIG", "GRAPH_RETRIEVAL_CONFIG"):
        return getattr(_configs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
