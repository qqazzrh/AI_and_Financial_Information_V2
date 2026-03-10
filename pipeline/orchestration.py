"""
Pipeline orchestration: resolve_company_from_ticker, build_retrieval_requests,
run_retrieval, run_full_pipeline, and run_tiered_pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pipeline.models import (
    ContractModel,
    PipelineConfig,
    RetrievalRequest,
    RetrievalResult,
    WorkerInput,
    WorkerOutput,
    ArbiterInput,
    ArbiterOutput,
    MasterInput,
    FinalUIPayload,
    SourcePreferencePolicy,
    ProcessedDocument,
    EmbeddingConfig,
)
from pipeline.enums import (
    DocumentType,
    ProcessingStatus,
    AnalysisTier,
    EmbeddingStatus,
)
from pipeline.config import logger, GLOBAL_CONFIG
# Lazy configs accessed via module-level __getattr__ in pipeline.config
from pipeline import config as _config_mod
from pipeline.retrieval import EdgarClient, RETRIEVAL_ADAPTER_REGISTRY
from pipeline.retrieval.base import make_retrieval_error, RetrievalErrorCode
from pipeline.analysis.client import build_analysis_client, BaseAnalysisClient
from pipeline.arbiter.models import EXPECTED_DOCUMENT_TYPES
from pipeline.analysis.registry import WORKER_REGISTRY
from pipeline.arbiter.cross_document import CrossDocumentArbiter
from pipeline.master import IntegratedMasterNode
from pipeline.processing.sections import (
    selected_document_from_retrieval_result,
    process_selected_document,
)
from pipeline.processing.embeddings import (
    VoyageAIEmbeddingClient,
    build_document_graph,
    cosine_similarity,
)
from pipeline.analysis.rubrics import build_generic_analysis_query


__all__ = [
    "resolve_company_from_ticker",
    "build_retrieval_requests",
    "run_retrieval",
    "run_full_pipeline",
    "TieredPipelineRequest",
    "set_database_handles",
    "run_tiered_pipeline",
]


# Database imports — optional, may not be available in all environments
try:
    from db import (
        init_databases,
        get_or_create_user,
        find_cached_run,
        find_any_cached_run,
        save_document_run,
        record_query,
        get_user_history,
        save_chunks_to_lancedb,
        get_chunks_from_lancedb,
    )
except ImportError:
    init_databases = None
    get_or_create_user = None
    find_cached_run = None
    find_any_cached_run = None
    save_document_run = None
    record_query = None
    get_user_history = None
    save_chunks_to_lancedb = None
    get_chunks_from_lancedb = None


# ---------------------------------------------------------------------------
# Lazy config accessors
# ---------------------------------------------------------------------------

def _get_pipeline_config() -> PipelineConfig:
    return _config_mod.PIPELINE_CONFIG


def _get_embedding_config() -> EmbeddingConfig:
    return _config_mod.EMBEDDING_CONFIG


# ---------------------------------------------------------------------------
# Helpers: resolve_company_from_ticker, build_retrieval_requests, run_retrieval
# ---------------------------------------------------------------------------

def resolve_company_from_ticker(ticker: str) -> tuple[str, list[str], str | None]:
    """Resolve ticker -> (company_name, aliases, cik) via SEC EDGAR company_tickers.json.

    Uses the authoritative ticker->CIK->name mapping. Returns CIK for downstream
    EDGAR searches. Falls back to (ticker, [ticker], None) on failure.
    """
    ticker_upper = ticker.strip().upper()
    try:
        client = EdgarClient()
        client._rate_limit()
        resp = client._session.get(
            "https://www.sec.gov/files/company_tickers.json", timeout=15
        )
        resp.raise_for_status()
        for entry in resp.json().values():
            if entry.get("ticker") == ticker_upper:
                company_name = entry["title"]
                cik = str(entry["cik_str"])
                aliases = list(dict.fromkeys([company_name, ticker_upper]))
                logger.info(f"Resolved {ticker_upper} -> {company_name} (CIK {cik})")
                return company_name, aliases, cik
    except Exception as exc:
        logger.warning(f"EDGAR company_tickers.json lookup failed for {ticker_upper}: {exc}")
    return ticker_upper, [ticker_upper], None


def build_retrieval_requests(
    ticker: str,
    company_name: str,
    company_aliases: list[str] | None = None,
    *,
    cik: str | None = None,
    config: PipelineConfig | None = None,
    run_id: str = "pipeline",
) -> list[RetrievalRequest]:
    """Build one RetrievalRequest per DocumentType, parameterized by ticker/company."""
    if config is None:
        config = _get_pipeline_config()
    aliases = company_aliases or [company_name, ticker]
    filters: dict[str, Any] = {}
    if cik:
        filters["cik"] = cik
    return [
        RetrievalRequest(
            request_id=f"{run_id}_{document_type.value}",
            ticker=ticker,
            company_name=company_name,
            company_aliases=aliases,
            document_type=document_type,
            max_candidates=config.default_max_candidates,
            minimum_text_chars=config.min_usable_text_chars,
            minimum_relevance_score=config.minimum_relevance_score,
            freshness_bonus_max=config.freshness_bonus_max,
            freshness_rank_decay=config.freshness_rank_decay,
            source_preferences=SourcePreferencePolicy(
                preferred_source_families=config.default_source_preferences,
                prefer_structured_source=True,
                allow_secondary_sources=True,
                allow_unknown_sources=True,
                allow_mock_data=False,
            ),
            retrieval_notes=[f"Live pipeline request for {ticker}."],
            is_mock_request=False,
            filters=filters,
        )
        for document_type in DocumentType
    ]


def run_retrieval(
    requests: list[RetrievalRequest],
) -> dict[DocumentType, RetrievalResult]:
    """Execute retrieval for all requests with per-adapter error isolation.

    If one adapter crashes, its result gets status=RETRIEVAL_FAILED instead of
    killing the whole pipeline. The arbiter already handles missing_document_types.
    """
    results: dict[DocumentType, RetrievalResult] = {}
    for request in requests:
        try:
            adapter_class = RETRIEVAL_ADAPTER_REGISTRY[request.document_type]
            adapter = adapter_class()
            results[request.document_type] = adapter.retrieve(request)
        except Exception as exc:
            logger.error(f"Retrieval adapter failed for {request.document_type.value}: {exc}")
            results[request.document_type] = RetrievalResult(
                request=request,
                adapter_name=RETRIEVAL_ADAPTER_REGISTRY[request.document_type].__name__,
                status=ProcessingStatus.RETRIEVAL_FAILED,
                issues=[
                    make_retrieval_error(
                        RetrievalErrorCode.ADAPTER_FAILURE,
                        message=f"Adapter crashed: {exc}",
                        stage="retrieval",
                        document_type=request.document_type,
                        adapter_name=RETRIEVAL_ADAPTER_REGISTRY[request.document_type].__name__,
                        recoverable=False,
                    )
                ],
            )
    return results


# ---------------------------------------------------------------------------
# Main entry point: run_full_pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    ticker: str,
    company_name: str | None = None,
    *,
    run_id: str | None = None,
    config: PipelineConfig | None = None,
    analysis_client: BaseAnalysisClient | None = None,
) -> dict[str, Any]:
    """True end-to-end pipeline: ticker -> retrieval -> analysis -> arbitration -> FinalUIPayload.

    Args:
        ticker: Stock ticker symbol (e.g. "GILD", "MRNA").
        company_name: Override company name. Auto-resolved via EDGAR if None.
        run_id: Pipeline run identifier. Auto-generated if None.
        config: Pipeline configuration. Defaults to PIPELINE_CONFIG.
        analysis_client: Override analysis client. Uses build_analysis_client(config) if None.
    """
    if config is None:
        config = _get_pipeline_config()
    effective_run_id = run_id or f"pipeline_{ticker.upper()}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"[run_full_pipeline] Starting pipeline for {ticker} (run_id={effective_run_id})")

    # 1. Resolve company name and CIK
    if company_name is None:
        resolved_name, aliases, cik = resolve_company_from_ticker(ticker)
    else:
        resolved_name = company_name
        aliases = [company_name, ticker]
        cik = None
    logger.info(f"[run_full_pipeline] Company: {resolved_name}" + (f" (CIK {cik})" if cik else ""))

    # 2. Build retrieval requests
    requests = build_retrieval_requests(
        ticker=ticker,
        company_name=resolved_name,
        company_aliases=aliases,
        cik=cik,
        config=config,
        run_id=effective_run_id,
    )

    # 3. Run retrieval with per-adapter error isolation
    logger.info(f"[run_full_pipeline] Running retrieval for {len(requests)} document types...")
    retrieval_results = run_retrieval(requests)
    success_count = sum(1 for r in retrieval_results.values() if r.status == ProcessingStatus.SUCCESS)
    logger.info(f"[run_full_pipeline] Retrieval complete: {success_count}/{len(requests)} succeeded")

    # 4. Build analysis client (never silently mock)
    effective_client = analysis_client or build_analysis_client(config)
    logger.info(f"[run_full_pipeline] Analysis client: {effective_client.client_name}")

    # 5. Run 5 workers
    logger.info(f"[run_full_pipeline] Running {len(EXPECTED_DOCUMENT_TYPES)} analysis workers...")
    worker_outputs: dict[DocumentType, WorkerOutput] = {}
    for document_type in EXPECTED_DOCUMENT_TYPES:
        worker_class = WORKER_REGISTRY[document_type]
        worker = worker_class(analysis_client=effective_client)
        retrieval_result = retrieval_results[document_type]
        worker_outputs[document_type] = worker.analyze(
            WorkerInput(
                run_id=effective_run_id,
                ticker=ticker,
                document_type=document_type,
                retrieval_result=retrieval_result,
                config=config,
                graph_context={},
            )
        )
    logger.info("[run_full_pipeline] Workers complete")

    # 6. Run arbiter
    logger.info("[run_full_pipeline] Running cross-document arbiter...")
    arbiter = CrossDocumentArbiter()
    arbiter_output = arbiter.arbitrate(
        ArbiterInput(
            run_id=effective_run_id,
            ticker=ticker,
            worker_outputs=list(worker_outputs.values()),
            retrieval_results=list(retrieval_results.values()),
            config=config,
        )
    )

    # 7. Run master node
    logger.info("[run_full_pipeline] Building final payload...")
    master_node = IntegratedMasterNode()
    final_payload = master_node.build_payload(
        MasterInput(
            run_id=effective_run_id,
            ticker=ticker,
            worker_outputs=list(worker_outputs.values()),
            arbiter_outputs=[arbiter_output],
            retrieval_results=list(retrieval_results.values()),
            config=config,
        )
    )

    logger.info(f"[run_full_pipeline] Done. Final status: {final_payload.status.value}")
    return {
        "ticker": ticker,
        "company_name": resolved_name,
        "run_id": effective_run_id,
        "retrieval_results": retrieval_results,
        "worker_outputs": worker_outputs,
        "arbiter_output": arbiter_output,
        "final_payload": final_payload,
    }


# ---------------------------------------------------------------------------
# TieredPipelineRequest model
# ---------------------------------------------------------------------------

class TieredPipelineRequest(ContractModel):
    """Request model for the tiered pipeline entry point."""

    ticker: str
    company_name: str | None = None
    user_id: str = "anonymous"
    enable_graph_context: bool = False
    run_id: str | None = None


# ---------------------------------------------------------------------------
# Module-level database handles (initialised by api_server or caller)
# ---------------------------------------------------------------------------

_sqlite_conn = None
_lance_db = None


def set_database_handles(sqlite_conn, lance_db) -> None:
    """Inject database connections from the API server layer."""
    global _sqlite_conn, _lance_db
    _sqlite_conn = sqlite_conn
    _lance_db = lance_db


def _ensure_databases():
    """Lazy-init databases if not already set."""
    global _sqlite_conn, _lance_db
    if _sqlite_conn is None or _lance_db is None:
        if init_databases is None:
            raise RuntimeError("Database module (db) is not available. Install it or call set_database_handles() first.")
        _sqlite_conn, _lance_db = init_databases()


# ---------------------------------------------------------------------------
# Tiered pipeline helpers
# ---------------------------------------------------------------------------

def _extract_release_date(retrieval_result: RetrievalResult) -> str | None:
    """Extract the SEC release / publication date from a retrieval result.

    Uses the selected candidate's published_at or event_date fields,
    falling back to None so the cache treats undated documents as a distinct group.
    """
    if retrieval_result.status != ProcessingStatus.SUCCESS:
        return None
    candidate = retrieval_result.selected_candidate
    if candidate is None:
        return None
    if candidate.published_at:
        return candidate.published_at.date().isoformat()
    if candidate.event_date:
        return candidate.event_date.isoformat()
    return None


def _build_graph_context_for_document(
    processed_document: ProcessedDocument,
    embedding_config: EmbeddingConfig,
    analysis_query: str | None = None,
) -> dict[str, Any]:
    """Run Tier 2 (Premium) enrichment: embed -> graph -> 1-hop expansion.

    Returns a graph_context dict suitable for WorkerInput.graph_context.
    """
    # 1. Embed all chunk texts
    chunk_texts = [chunk.text for chunk in processed_document.chunks]
    embedding_client = VoyageAIEmbeddingClient()
    embed_result = embedding_client.embed_texts(chunk_texts, embedding_config)

    vectors = embed_result.vectors if embed_result.status == EmbeddingStatus.SUCCESS else []

    # 2. Persist chunks + embeddings to LanceDB
    chunk_dicts = [chunk.model_dump(mode="json") for chunk in processed_document.chunks]
    save_chunks_to_lancedb(
        _lance_db,
        document_id=processed_document.document.document_id,
        chunks=chunk_dicts,
        embeddings=vectors if vectors else None,
    )

    # 3. Build the document graph
    graph_index = build_document_graph(processed_document)

    # 4. Semantic retrieval + 1-hop graph expansion (if we have vectors)
    expanded_context: dict[str, Any] = {
        "graph_node_count": len(graph_index.nodes),
        "graph_edge_count": len(graph_index.edges),
        "embedding_status": embed_result.status.value,
        "enriched_chunks": [],
    }

    if vectors and analysis_query:
        # Embed the query
        query_embed_result = embedding_client.embed_texts([analysis_query], embedding_config)
        if query_embed_result.status == EmbeddingStatus.SUCCESS and query_embed_result.vectors:
            query_vector = query_embed_result.vectors[0]

            # Score all chunks by cosine similarity
            scored = []
            for idx, chunk in enumerate(processed_document.chunks):
                if idx < len(vectors):
                    sim = cosine_similarity(query_vector, vectors[idx])
                    scored.append((sim, idx, chunk))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_k = scored[:5]  # top-5 most relevant chunks

            # 1-hop expansion from graph neighbours
            for sim, idx, chunk in top_k:
                neighbour_ids = graph_index.chunk_neighbors.get(chunk.chunk_id, [])
                neighbour_texts = []
                for nc in processed_document.chunks:
                    if nc.chunk_id in neighbour_ids:
                        neighbour_texts.append(nc.text)

                expanded_context["enriched_chunks"].append({
                    "chunk_id": chunk.chunk_id,
                    "similarity": round(sim, 4),
                    "text": chunk.text,
                    "neighbour_chunk_ids": neighbour_ids,
                    "neighbour_texts": neighbour_texts,
                    "section_title": chunk.parent_section_title,
                })

    return expanded_context


# ---------------------------------------------------------------------------
# run_tiered_pipeline
# ---------------------------------------------------------------------------

def run_tiered_pipeline(
    request: TieredPipelineRequest,
) -> dict[str, Any]:
    """Tiered pipeline with smart caching.

    1. Resolve ticker -> company/CIK
    2. Run retrieval to get documents and SEC release dates
    3. For each document type, check the cache:
       - Exact hit -> serve from SQLite (zero compute)
       - Upsell (has text_only, wants vector_graph) -> embed existing chunks
       - Miss -> run from scratch
    4. Run workers on the (possibly cached) processed documents
    5. Run arbiter
    6. Record query history
    7. Return FinalUIPayload
    """
    _ensure_databases()

    PIPELINE_CONFIG = _get_pipeline_config()
    EMBEDDING_CONFIG = _get_embedding_config()

    analysis_tier = AnalysisTier.VECTOR_GRAPH if request.enable_graph_context else AnalysisTier.TEXT_ONLY
    effective_run_id = request.run_id or f"tiered_{request.ticker.upper()}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"[run_tiered_pipeline] Starting {analysis_tier.value} pipeline for {request.ticker} (run_id={effective_run_id})")

    # Ensure user exists
    get_or_create_user(_sqlite_conn, request.user_id)

    # 1. Resolve company
    if request.company_name is None:
        resolved_name, aliases, cik = resolve_company_from_ticker(request.ticker)
    else:
        resolved_name = request.company_name
        aliases = [request.company_name, request.ticker]
        cik = None

    # 2. Build and run retrieval (needed to discover release dates)
    requests_list = build_retrieval_requests(
        ticker=request.ticker,
        company_name=resolved_name,
        company_aliases=aliases,
        cik=cik,
        config=PIPELINE_CONFIG,
        run_id=effective_run_id,
    )

    logger.info(f"[run_tiered_pipeline] Running retrieval for {len(requests_list)} document types...")
    retrieval_results = run_retrieval(requests_list)

    # 3. Build analysis client
    effective_client = build_analysis_client(PIPELINE_CONFIG)
    logger.info(f"[run_tiered_pipeline] Analysis client: {effective_client.client_name}")

    # 4. Process each document type with cache-aware logic
    embedding_config = EMBEDDING_CONFIG
    worker_outputs: dict[DocumentType, WorkerOutput] = {}
    cache_hits = 0
    cache_misses = 0

    for document_type in EXPECTED_DOCUMENT_TYPES:
        retrieval_result = retrieval_results[document_type]
        release_date = _extract_release_date(retrieval_result)

        # --- Cache check ---
        cached = find_cached_run(
            _sqlite_conn,
            ticker=request.ticker,
            document_type=document_type.value,
            release_date=release_date,
            analysis_tier=analysis_tier.value,
        )

        if cached and cached.get("worker_payload"):
            # Scenario A: exact cache hit
            logger.info(f"[run_tiered_pipeline] Cache HIT for {document_type.value} ({analysis_tier.value})")
            cache_hits += 1
            worker_outputs[document_type] = WorkerOutput(**cached["worker_payload"])
            continue

        # Scenario B: upsell path — user wants vector_graph but we only have text_only
        existing_lower = None
        if analysis_tier == AnalysisTier.VECTOR_GRAPH:
            existing_lower = find_cached_run(
                _sqlite_conn,
                ticker=request.ticker,
                document_type=document_type.value,
                release_date=release_date,
                analysis_tier=AnalysisTier.TEXT_ONLY.value,
            )

        # Scenario C (or B): run worker
        cache_misses += 1
        logger.info(f"[run_tiered_pipeline] Cache MISS for {document_type.value} — running {analysis_tier.value} worker")

        worker_class = WORKER_REGISTRY[document_type]
        worker = worker_class(analysis_client=effective_client)

        graph_context: dict[str, Any] = {}

        if analysis_tier == AnalysisTier.VECTOR_GRAPH and retrieval_result.status == ProcessingStatus.SUCCESS:
            # Build processed document for embedding + graph
            selected = selected_document_from_retrieval_result(retrieval_result)
            if selected:
                processed = process_selected_document(selected)
                analysis_query = build_generic_analysis_query(processed)
                graph_context = _build_graph_context_for_document(
                    processed, embedding_config, analysis_query
                )
        elif analysis_tier == AnalysisTier.TEXT_ONLY and retrieval_result.status == ProcessingStatus.SUCCESS:
            # Tier 1: still save raw chunks to LanceDB for potential future upsell
            selected = selected_document_from_retrieval_result(retrieval_result)
            if selected:
                processed = process_selected_document(selected)
                chunk_dicts = [chunk.model_dump(mode="json") for chunk in processed.chunks]
                save_chunks_to_lancedb(
                    _lance_db,
                    document_id=processed.document.document_id,
                    chunks=chunk_dicts,
                )

        worker_output = worker.analyze(
            WorkerInput(
                run_id=effective_run_id,
                ticker=request.ticker,
                document_type=document_type,
                retrieval_result=retrieval_result,
                config=PIPELINE_CONFIG,
                graph_context=graph_context,
            )
        )
        worker_outputs[document_type] = worker_output

        # Persist to cache ledger
        doc_run_id = f"{effective_run_id}_{document_type.value}"
        save_document_run(
            _sqlite_conn,
            run_id=doc_run_id,
            ticker=request.ticker,
            document_type=document_type.value,
            release_date=release_date,
            analysis_tier=analysis_tier.value,
            worker_payload=worker_output.model_dump(mode="json"),
        )

    logger.info(f"[run_tiered_pipeline] Workers complete: {cache_hits} cache hits, {cache_misses} computed")

    # 5. Run arbiter
    logger.info("[run_tiered_pipeline] Running cross-document arbiter...")
    arbiter = CrossDocumentArbiter()
    arbiter_output = arbiter.arbitrate(
        ArbiterInput(
            run_id=effective_run_id,
            ticker=request.ticker,
            worker_outputs=list(worker_outputs.values()),
            retrieval_results=list(retrieval_results.values()),
            config=PIPELINE_CONFIG,
        )
    )

    # 6. Build final payload
    logger.info("[run_tiered_pipeline] Building final payload...")
    master_node = IntegratedMasterNode()
    final_payload = master_node.build_payload(
        MasterInput(
            run_id=effective_run_id,
            ticker=request.ticker,
            worker_outputs=list(worker_outputs.values()),
            arbiter_outputs=[arbiter_output],
            retrieval_results=list(retrieval_results.values()),
            config=PIPELINE_CONFIG,
        )
    )

    # 7. Record query history (use the first doc_run_id as the linkage)
    first_doc_type = EXPECTED_DOCUMENT_TYPES[0]
    primary_run_id = f"{effective_run_id}_{first_doc_type.value}"
    arbiter_response_json = arbiter_output.model_dump(mode="json")

    record_query(
        _sqlite_conn,
        user_id=request.user_id,
        run_id=primary_run_id,
        ticker=request.ticker,
        arbiter_response=arbiter_response_json,
    )

    logger.info(f"[run_tiered_pipeline] Pipeline complete (tier={analysis_tier.value})")

    return {
        "run_id": effective_run_id,
        "analysis_tier": analysis_tier.value,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "final_payload": final_payload,
        "arbiter_output": arbiter_output,
        "worker_outputs": worker_outputs,
        "retrieval_results": retrieval_results,
    }
