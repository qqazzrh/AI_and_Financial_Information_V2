"""Embedding client, graph building, and retrieval functions."""

from __future__ import annotations

import json
import math
import os
import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pipeline.config import (
    EMBEDDING_CONFIG,
    GRAPH_RETRIEVAL_CONFIG,
    logger,
)

__all__ = [
    "tokenize_for_embedding",
    "l2_normalize",
    "cosine_similarity",
    "BaseEmbeddingClient",
    "VoyageAIEmbeddingClient",
    "build_embedding_client",
    "batch_embed_texts",
    "build_chunk_embedding_records",
    "build_document_graph",
    "get_neighbor_chunks",
    "expand_chunk_context",
    "score_chunk_with_graph_context",
    "build_chunk_embedding_lookup",
    "make_text_excerpt",
    "retrieve_relevant_chunks",
]
from pipeline.enums import (
    EmbeddingProvider,
    EmbeddingStatus,
    GraphEdgeType,
    GraphNodeType,
    ProcessingNoteSeverity,
)
from pipeline.models import (
    ChunkEmbeddingRecord,
    ChunkRetrievalHit,
    ChunkRetrievalResult,
    DocumentChunk,
    DocumentSection,
    EmbeddingBatchResult,
    EmbeddingConfig,
    GraphDocumentIndex,
    GraphEdge,
    GraphNode,
    GraphRetrievalConfig,
    ProcessedDocument,
    ProcessingNote,
)
from pipeline.processing.normalization import make_processing_note
from pipeline.retrieval import normalize_token


# ---------------------------------------------------------------------------
# Tokenization and vector helpers
# ---------------------------------------------------------------------------


def tokenize_for_embedding(text: str) -> list[str]:
    """Tokenize text for deterministic lexical overlap checks."""
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Return an L2-normalized copy of a vector."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def cosine_similarity(
    vector_a: Sequence[float], vector_b: Sequence[float]
) -> float:
    """Compute cosine similarity for two vectors of equal length."""
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return 0.0
    numerator = sum(left * right for left, right in zip(vector_a, vector_b))
    left_norm = math.sqrt(sum(value * value for value in vector_a))
    right_norm = math.sqrt(sum(value * value for value in vector_b))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


# ---------------------------------------------------------------------------
# Embedding clients
# ---------------------------------------------------------------------------


class BaseEmbeddingClient(ABC):
    """Base embedding client interface."""

    provider_name: ClassVar[str] = "base"

    @abstractmethod
    def embed_texts(
        self, texts: Sequence[str], config: EmbeddingConfig
    ) -> EmbeddingBatchResult:
        """Embed a batch of texts using the configured provider."""


class VoyageAIEmbeddingClient(BaseEmbeddingClient):
    """Voyage AI embedding client using the REST API."""

    provider_name = EmbeddingProvider.VOYAGE.value

    def embed_texts(
        self, texts: Sequence[str], config: EmbeddingConfig
    ) -> EmbeddingBatchResult:
        if not texts:
            logger.info(
                "VoyageAI embed_texts called with empty text list — skipping."
            )
            return EmbeddingBatchResult(
                provider_name=self.provider_name,
                model_name=config.model_name,
                status=EmbeddingStatus.SKIPPED,
                vectors=[],
                notes=[
                    make_processing_note(
                        "no_texts_to_embed",
                        "No texts were provided to the embedding client.",
                        severity=ProcessingNoteSeverity.INFO,
                    )
                ],
            )

        if not config.enabled:
            logger.warning(
                "Embedding generation is disabled in config — skipping Voyage AI API call."
            )
            return EmbeddingBatchResult(
                provider_name=self.provider_name,
                model_name=config.model_name,
                status=EmbeddingStatus.SKIPPED,
                vectors=[],
                notes=[
                    make_processing_note(
                        "embeddings_disabled",
                        "Embedding generation is disabled in the current configuration.",
                        severity=ProcessingNoteSeverity.WARNING,
                    )
                ],
            )

        api_key = os.environ.get("VOYAGE_AI_API_KEY", "")
        if not api_key:
            logger.error(
                "VOYAGE_AI_API_KEY is not set — cannot call Voyage AI embedding API. "
                "Set VOYAGE_AI_API_KEY in your .env file to enable embeddings."
            )
            return EmbeddingBatchResult(
                provider_name=self.provider_name,
                model_name=config.model_name,
                status=EmbeddingStatus.FAILED,
                vectors=[],
                notes=[
                    make_processing_note(
                        "voyage_api_key_missing",
                        "VOYAGE_AI_API_KEY environment variable is not set.",
                        severity=ProcessingNoteSeverity.ERROR,
                    )
                ],
            )

        url = config.base_url.rstrip("/") + config.api_path
        logger.info(
            f"Calling Voyage AI embedding API: {len(texts)} text(s), model={config.model_name}"
        )
        payload = json.dumps(
            {
                "input": list(texts),
                "model": config.model_name,
                "input_type": "document",
            }
        ).encode("utf-8")
        request = Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=config.request_timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            logger.error(f"Voyage AI embedding request FAILED: {exc}")
            return EmbeddingBatchResult(
                provider_name=self.provider_name,
                model_name=config.model_name,
                status=EmbeddingStatus.FAILED,
                vectors=[],
                notes=[
                    make_processing_note(
                        "voyage_request_failed",
                        f"Voyage AI embedding request failed: {exc}",
                        severity=ProcessingNoteSeverity.ERROR,
                    )
                ],
            )

        data = response_payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            logger.error(
                f"Voyage AI response invalid: expected {len(texts)} embeddings, "
                f"got {len(data) if isinstance(data, list) else type(data).__name__}"
            )
            return EmbeddingBatchResult(
                provider_name=self.provider_name,
                model_name=config.model_name,
                status=EmbeddingStatus.FAILED,
                vectors=[],
                notes=[
                    make_processing_note(
                        "voyage_response_invalid",
                        "Voyage AI response did not contain a data array matching the request batch size.",
                        severity=ProcessingNoteSeverity.ERROR,
                    )
                ],
            )

        raw_vectors = [item["embedding"] for item in data]
        normalized_vectors = [
            l2_normalize(v) if config.normalize_vectors else v for v in raw_vectors
        ]
        logger.info(
            f"Voyage AI embedding SUCCESS: {len(normalized_vectors)} vector(s), "
            f"{len(normalized_vectors[0])} dims each"
        )
        return EmbeddingBatchResult(
            provider_name=self.provider_name,
            model_name=config.model_name,
            status=EmbeddingStatus.SUCCESS,
            vectors=normalized_vectors,
            is_mock_embedding=False,
            notes=[
                make_processing_note(
                    "voyage_embeddings_complete",
                    f"Generated {len(normalized_vectors)} embedding vector(s) via Voyage AI.",
                    severity=ProcessingNoteSeverity.INFO,
                )
            ],
        )


# ---------------------------------------------------------------------------
# Client factory and batch helpers
# ---------------------------------------------------------------------------


def build_embedding_client(
    config: EmbeddingConfig | None = None,
) -> BaseEmbeddingClient:
    """Build the preferred embedding client from configuration."""
    return VoyageAIEmbeddingClient()


def batch_embed_texts(
    texts: Sequence[str],
    config: EmbeddingConfig | None = None,
    preferred_client: BaseEmbeddingClient | None = None,
) -> EmbeddingBatchResult:
    """Embed text batches via the configured provider. Fails loudly if the API call cannot be made."""
    config = config or EMBEDDING_CONFIG
    client = preferred_client or build_embedding_client(config)
    batch_result = client.embed_texts(texts, config)
    if batch_result.status == EmbeddingStatus.FAILED:
        logger.error(
            f"Embedding batch FAILED for {len(texts)} text(s). "
            f"Reason: {batch_result.notes[0].message if batch_result.notes else 'unknown'}"
        )
    return batch_result


def build_chunk_embedding_records(
    processed_document: ProcessedDocument,
    config: EmbeddingConfig | None = None,
    preferred_client: BaseEmbeddingClient | None = None,
) -> tuple[list[ChunkEmbeddingRecord], list[ProcessingNote]]:
    """Build chunk embedding records for a processed document via live API calls."""
    config = config or EMBEDDING_CONFIG
    if not processed_document.chunks:
        notes = [
            make_processing_note(
                "no_chunks_to_embed",
                "Processed document contains no chunks to embed.",
                severity=ProcessingNoteSeverity.WARNING,
                document_id=processed_document.document.document_id,
            )
        ]
        logger.warning(
            f"No chunks to embed for document {processed_document.document.document_id}"
        )
        return [], notes

    logger.info(
        f"Embedding {len(processed_document.chunks)} chunks for document "
        f"{processed_document.document.document_id}"
    )
    batch_result = batch_embed_texts(
        [chunk.text for chunk in processed_document.chunks],
        config=config,
        preferred_client=preferred_client,
    )
    if batch_result.status == EmbeddingStatus.FAILED:
        logger.error(
            f"Chunk embedding FAILED for document {processed_document.document.document_id} — "
            f"retrieval will have no vectors."
        )
        return [], batch_result.notes

    records: list[ChunkEmbeddingRecord] = []
    for chunk, vector in zip(processed_document.chunks, batch_result.vectors):
        records.append(
            ChunkEmbeddingRecord(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                provider_name=batch_result.provider_name,
                model_name=batch_result.model_name,
                status=batch_result.status,
                embedding=vector,
                vector_dimension=len(vector),
                is_mock_embedding=batch_result.is_mock_embedding,
                notes=batch_result.notes,
            )
        )
    return records, batch_result.notes


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------


def build_document_graph(
    processed_document: ProcessedDocument,
) -> GraphDocumentIndex:
    """Build a lightweight graph index from one processed document."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    notes: list[ProcessingNote] = []
    section_to_chunk_ids: dict[str, list[str]] = {}
    chunk_neighbors: dict[str, list[str]] = {
        chunk.chunk_id: [] for chunk in processed_document.chunks
    }
    chunk_to_section_id: dict[str, str] = {}
    cross_reference_targets: dict[str, list[str]] = {}

    document_node_id = f"document::{processed_document.document.document_id}"
    nodes.append(
        GraphNode(
            node_id=document_node_id,
            document_id=processed_document.document.document_id,
            node_type=GraphNodeType.DOCUMENT,
            label=processed_document.document.title
            or processed_document.document.document_id,
            ref_id=processed_document.document.document_id,
        )
    )

    section_lookup: dict[str, DocumentSection] = {
        section.section_id: section for section in processed_document.sections
    }
    section_reference_lookup: dict[str, str] = {}

    for section in processed_document.sections:
        section_node_id = f"section::{section.section_id}"
        nodes.append(
            GraphNode(
                node_id=section_node_id,
                document_id=processed_document.document.document_id,
                node_type=GraphNodeType.SECTION,
                label=section.title,
                ref_id=section.section_id,
                metadata={
                    "section_kind": section.section_kind.value,
                    "level": section.level,
                },
            )
        )
        edges.append(
            GraphEdge(
                edge_id=f"edge::{document_node_id}::{section_node_id}",
                document_id=processed_document.document.document_id,
                source_node_id=document_node_id,
                target_node_id=section_node_id,
                edge_type=GraphEdgeType.CONTAINS,
            )
        )
        if section.reference_label:
            section_reference_lookup[
                normalize_token(section.reference_label)
            ] = section.section_id

    for previous_section, next_section in zip(
        processed_document.sections, processed_document.sections[1:]
    ):
        edges.append(
            GraphEdge(
                edge_id=f"edge::section_sequence::{previous_section.section_id}::{next_section.section_id}",
                document_id=processed_document.document.document_id,
                source_node_id=f"section::{previous_section.section_id}",
                target_node_id=f"section::{next_section.section_id}",
                edge_type=GraphEdgeType.SECTION_SEQUENCE,
                weight=0.5,
            )
        )

    for chunk in processed_document.chunks:
        chunk_node_id = f"chunk::{chunk.chunk_id}"
        nodes.append(
            GraphNode(
                node_id=chunk_node_id,
                document_id=processed_document.document.document_id,
                node_type=GraphNodeType.CHUNK,
                label=chunk.parent_section_title or chunk.chunk_id,
                ref_id=chunk.chunk_id,
                metadata={
                    "chunk_index": chunk.chunk_index,
                    "section_kind": chunk.section_kind.value,
                },
            )
        )
        if chunk.parent_section_id:
            section_to_chunk_ids.setdefault(chunk.parent_section_id, []).append(
                chunk.chunk_id
            )
            chunk_to_section_id[chunk.chunk_id] = chunk.parent_section_id
            edges.append(
                GraphEdge(
                    edge_id=f"edge::contains::{chunk.parent_section_id}::{chunk.chunk_id}",
                    document_id=processed_document.document.document_id,
                    source_node_id=f"section::{chunk.parent_section_id}",
                    target_node_id=chunk_node_id,
                    edge_type=GraphEdgeType.CONTAINS,
                )
            )

        if chunk.previous_chunk_id:
            chunk_neighbors[chunk.chunk_id].append(chunk.previous_chunk_id)
            edges.append(
                GraphEdge(
                    edge_id=f"edge::adjacent::{chunk.previous_chunk_id}::{chunk.chunk_id}",
                    document_id=processed_document.document.document_id,
                    source_node_id=f"chunk::{chunk.previous_chunk_id}",
                    target_node_id=chunk_node_id,
                    edge_type=GraphEdgeType.ADJACENT,
                )
            )
        if chunk.next_chunk_id:
            chunk_neighbors[chunk.chunk_id].append(chunk.next_chunk_id)

    for section_id, chunk_ids in section_to_chunk_ids.items():
        for first_chunk_id, second_chunk_id in zip(chunk_ids, chunk_ids[1:]):
            chunk_neighbors.setdefault(first_chunk_id, []).append(second_chunk_id)
            chunk_neighbors.setdefault(second_chunk_id, []).append(first_chunk_id)
            edges.append(
                GraphEdge(
                    edge_id=f"edge::same_section::{first_chunk_id}::{second_chunk_id}",
                    document_id=processed_document.document.document_id,
                    source_node_id=f"chunk::{first_chunk_id}",
                    target_node_id=f"chunk::{second_chunk_id}",
                    edge_type=GraphEdgeType.SAME_SECTION,
                    weight=0.75,
                )
            )

    for chunk in processed_document.chunks:
        normalized_chunk_text = normalize_token(chunk.text)
        for reference_label, target_section_id in section_reference_lookup.items():
            if (
                reference_label
                and target_section_id != chunk.parent_section_id
                and f"section {reference_label}" in normalized_chunk_text
            ):
                cross_reference_targets.setdefault(chunk.chunk_id, []).append(
                    target_section_id
                )
                edges.append(
                    GraphEdge(
                        edge_id=f"edge::cross_reference::{chunk.chunk_id}::{target_section_id}",
                        document_id=processed_document.document.document_id,
                        source_node_id=f"chunk::{chunk.chunk_id}",
                        target_node_id=f"section::{target_section_id}",
                        edge_type=GraphEdgeType.CROSS_REFERENCE,
                        weight=0.5,
                    )
                )

    if len(processed_document.sections) <= 1:
        notes.append(
            make_processing_note(
                "graph_structure_minimal",
                "Graph structure is minimal because the document has one or fewer sections.",
                severity=ProcessingNoteSeverity.WARNING,
                document_id=processed_document.document.document_id,
            )
        )
    else:
        notes.append(
            make_processing_note(
                "graph_index_complete",
                f"Built graph index with {len(nodes)} nodes and {len(edges)} edges.",
                severity=ProcessingNoteSeverity.INFO,
                document_id=processed_document.document.document_id,
            )
        )

    for chunk_id, neighbors in chunk_neighbors.items():
        chunk_neighbors[chunk_id] = sorted(set(neighbors))

    return GraphDocumentIndex(
        document_id=processed_document.document.document_id,
        nodes=nodes,
        edges=edges,
        section_to_chunk_ids=section_to_chunk_ids,
        chunk_neighbors=chunk_neighbors,
        chunk_to_section_id=chunk_to_section_id,
        cross_reference_targets=cross_reference_targets,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Graph traversal helpers
# ---------------------------------------------------------------------------


def get_neighbor_chunks(
    graph_index: GraphDocumentIndex,
    chunk_id: str,
    *,
    max_hops: int = 1,
) -> list[str]:
    """Return neighboring chunk ids using adjacency and same-section links."""
    visited = {chunk_id}
    frontier = {chunk_id}
    neighbors: list[str] = []
    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for current_chunk_id in frontier:
            for neighbor_id in graph_index.chunk_neighbors.get(
                current_chunk_id, []
            ):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    next_frontier.add(neighbor_id)
                    neighbors.append(neighbor_id)
        frontier = next_frontier
        if not frontier:
            break
    return neighbors


def expand_chunk_context(
    processed_document: ProcessedDocument,
    graph_index: GraphDocumentIndex,
    chunk_id: str,
    *,
    max_hops: int = 1,
) -> list[DocumentChunk]:
    """Expand one chunk into a small neighborhood for richer retrieval context."""
    chunk_lookup = {chunk.chunk_id: chunk for chunk in processed_document.chunks}
    related_ids = [chunk_id] + get_neighbor_chunks(
        graph_index, chunk_id, max_hops=max_hops
    )
    related_chunks = [
        chunk_lookup[candidate_id]
        for candidate_id in related_ids
        if candidate_id in chunk_lookup
    ]
    return sorted(related_chunks, key=lambda chunk: chunk.chunk_index)


def score_chunk_with_graph_context(
    *,
    query_text: str,
    chunk: DocumentChunk,
    processed_document: ProcessedDocument,
    graph_index: GraphDocumentIndex,
    base_similarity: float,
    similarity_lookup: dict[str, float],
    retrieval_config: GraphRetrievalConfig | None = None,
) -> tuple[float, float, list[ProcessingNote]]:
    """Apply a lightweight graph bonus using section titles and neighbor support."""
    retrieval_config = retrieval_config or GRAPH_RETRIEVAL_CONFIG
    notes: list[ProcessingNote] = []
    query_tokens = set(tokenize_for_embedding(query_text))
    section_title_tokens = set(
        tokenize_for_embedding(chunk.parent_section_title or "")
    )
    section_overlap_ratio = 0.0
    if query_tokens and section_title_tokens:
        section_overlap_ratio = len(query_tokens & section_title_tokens) / len(
            query_tokens | section_title_tokens
        )

    neighbor_ids = get_neighbor_chunks(
        graph_index, chunk.chunk_id, max_hops=retrieval_config.neighbor_hops
    )
    neighbor_similarity = max(
        (similarity_lookup.get(neighbor_id, 0.0) for neighbor_id in neighbor_ids),
        default=0.0,
    )
    cross_reference_bonus = (
        retrieval_config.cross_reference_bonus
        if graph_index.cross_reference_targets.get(chunk.chunk_id)
        else 0.0
    )

    graph_bonus = min(
        retrieval_config.max_graph_bonus,
        (section_overlap_ratio * retrieval_config.section_title_weight)
        + (neighbor_similarity * retrieval_config.neighbor_similarity_weight)
        + cross_reference_bonus,
    )
    adjusted_score = base_similarity + graph_bonus

    if graph_bonus:
        notes.append(
            make_processing_note(
                "graph_bonus_applied",
                "Applied graph-aware retrieval bonus using section title overlap and neighbor support.",
                severity=ProcessingNoteSeverity.INFO,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                metadata={
                    "section_overlap_ratio": round(section_overlap_ratio, 4),
                    "neighbor_similarity": round(neighbor_similarity, 4),
                    "cross_reference_bonus": round(cross_reference_bonus, 4),
                },
            )
        )
    return adjusted_score, graph_bonus, notes


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def build_chunk_embedding_lookup(
    records: Sequence[ChunkEmbeddingRecord],
) -> dict[str, list[float]]:
    """Build a chunk-id to embedding vector mapping."""
    return {
        record.chunk_id: record.embedding
        for record in records
        if record.embedding is not None
    }


def make_text_excerpt(text: str, limit: int = 180) -> str:
    """Build a short, single-line excerpt for notebook display."""
    single_line = " ".join(text.split())
    if len(single_line) <= limit:
        return single_line
    return single_line[: limit - 3] + "..."


def retrieve_relevant_chunks(
    query_text: str,
    processed_document: ProcessedDocument,
    chunk_embeddings: Sequence[ChunkEmbeddingRecord],
    graph_index: GraphDocumentIndex,
    embedding_config: EmbeddingConfig | None = None,
    retrieval_config: GraphRetrievalConfig | None = None,
    preferred_client: BaseEmbeddingClient | None = None,
) -> ChunkRetrievalResult:
    """Retrieve relevant chunks from one processed document using embeddings plus graph context."""
    embedding_config = embedding_config or EMBEDDING_CONFIG
    retrieval_config = retrieval_config or GRAPH_RETRIEVAL_CONFIG
    notes: list[ProcessingNote] = []
    if not processed_document.chunks:
        notes.append(
            make_processing_note(
                "no_chunks_available",
                "No chunks are available for retrieval from this processed document.",
                severity=ProcessingNoteSeverity.WARNING,
                document_id=processed_document.document.document_id,
            )
        )
        return ChunkRetrievalResult(
            document_id=processed_document.document.document_id,
            query_text=query_text,
            embedding_provider=embedding_config.provider.value,
            embedding_model_name=embedding_config.model_name,
            query_embedding_status=EmbeddingStatus.SKIPPED,
            hits=[],
            notes=notes,
        )

    query_embedding_batch = batch_embed_texts(
        [query_text], config=embedding_config, preferred_client=preferred_client
    )
    notes.extend(query_embedding_batch.notes)
    if not query_embedding_batch.vectors:
        notes.append(
            make_processing_note(
                "query_embedding_unavailable",
                "Query embedding could not be generated, so no chunk retrieval was performed.",
                severity=ProcessingNoteSeverity.WARNING,
                document_id=processed_document.document.document_id,
            )
        )
        return ChunkRetrievalResult(
            document_id=processed_document.document.document_id,
            query_text=query_text,
            embedding_provider=query_embedding_batch.provider_name,
            embedding_model_name=query_embedding_batch.model_name,
            query_embedding_status=query_embedding_batch.status,
            hits=[],
            notes=notes,
        )

    query_vector = query_embedding_batch.vectors[0]
    chunk_lookup = {chunk.chunk_id: chunk for chunk in processed_document.chunks}
    embedding_lookup = build_chunk_embedding_lookup(chunk_embeddings)
    similarity_lookup: dict[str, float] = {}
    for chunk in processed_document.chunks:
        vector = embedding_lookup.get(chunk.chunk_id)
        if vector is None:
            continue
        similarity_lookup[chunk.chunk_id] = cosine_similarity(query_vector, vector)

    candidate_ids = [
        chunk_id
        for chunk_id, similarity in sorted(
            similarity_lookup.items(), key=lambda item: (-item[1], item[0])
        )
        if similarity >= retrieval_config.min_similarity_threshold
    ]

    if not candidate_ids:
        notes.append(
            make_processing_note(
                "no_chunk_matches",
                "Chunk retrieval returned no candidates above the similarity threshold.",
                severity=ProcessingNoteSeverity.WARNING,
                document_id=processed_document.document.document_id,
            )
        )
        return ChunkRetrievalResult(
            document_id=processed_document.document.document_id,
            query_text=query_text,
            embedding_provider=query_embedding_batch.provider_name,
            embedding_model_name=query_embedding_batch.model_name,
            query_embedding_status=query_embedding_batch.status,
            hits=[],
            notes=notes,
        )

    candidate_pool_size = max(
        retrieval_config.top_k, retrieval_config.candidate_pool_size
    )
    candidate_pool_ids = candidate_ids[:candidate_pool_size]
    hits: list[ChunkRetrievalHit] = []
    for chunk_id in candidate_pool_ids:
        chunk = chunk_lookup[chunk_id]
        base_similarity = similarity_lookup.get(chunk_id, 0.0)
        adjusted_score, graph_bonus, graph_notes = score_chunk_with_graph_context(
            query_text=query_text,
            chunk=chunk,
            processed_document=processed_document,
            graph_index=graph_index,
            base_similarity=base_similarity,
            similarity_lookup=similarity_lookup,
            retrieval_config=retrieval_config,
        )
        expanded_chunks = expand_chunk_context(
            processed_document,
            graph_index,
            chunk_id,
            max_hops=retrieval_config.context_expansion_hops,
        )
        hits.append(
            ChunkRetrievalHit(
                rank=0,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                section_id=chunk.parent_section_id,
                section_title=chunk.parent_section_title,
                similarity_score=base_similarity,
                graph_bonus=graph_bonus,
                adjusted_score=adjusted_score,
                text_excerpt=make_text_excerpt(chunk.text),
                expanded_context_chunk_ids=[
                    expanded_chunk.chunk_id for expanded_chunk in expanded_chunks
                ],
                expanded_context_preview=make_text_excerpt(
                    "\n\n".join(
                        expanded_chunk.text for expanded_chunk in expanded_chunks
                    ),
                    220,
                ),
                notes=graph_notes,
            )
        )

    ranked_hits = sorted(
        hits, key=lambda hit: (-hit.adjusted_score, hit.chunk_id)
    )[: retrieval_config.top_k]
    for rank, hit in enumerate(ranked_hits, start=1):
        hit.rank = rank

    if len(ranked_hits) < retrieval_config.top_k:
        notes.append(
            make_processing_note(
                "too_few_retrieval_hits",
                f"Only {len(ranked_hits)} chunk hit(s) were available for top_k={retrieval_config.top_k}.",
                severity=ProcessingNoteSeverity.WARNING,
                document_id=processed_document.document.document_id,
            )
        )

    return ChunkRetrievalResult(
        document_id=processed_document.document.document_id,
        query_text=query_text,
        embedding_provider=query_embedding_batch.provider_name,
        embedding_model_name=query_embedding_batch.model_name,
        query_embedding_status=query_embedding_batch.status,
        used_graph_context=True,
        hits=ranked_hits,
        notes=notes,
    )
