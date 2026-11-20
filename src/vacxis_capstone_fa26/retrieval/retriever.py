"""End-to-end Retrieval Pipeline for VacxiS Capstone."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
import logging
from typing import Any, Sequence, TYPE_CHECKING

import torch
if TYPE_CHECKING:
    from FlagEmbedding import BGEM3FlagModel

from vacxis_capstone_fa26.indexing.table_summarizer import merge_table_fragments
from vacxis_capstone_fa26.retrieval.hybrid_searcher import HybridSearcher
from vacxis_capstone_fa26.retrieval.reranker import BGEReranker
from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig, RetrievedChunk

logger = logging.getLogger("vacxis.retrieval.retriever")


class VacxisRetriever:
    """End-to-end retriever orchestrating BGE-M3 encoding, Hybrid Qdrant search, RRF, and Reranking."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        embed_model: BGEM3FlagModel | None = None,
        reranker: BGEReranker | None = None,
        hybrid_searcher: HybridSearcher | None = None,
        lazy_load: bool = True,
    ) -> None:
        """Initialize retrieval pipeline components.

        Args:
            config: Pipeline configuration settings.
            embed_model: Optional pre-loaded BGE-M3 model.
            reranker: Optional pre-loaded BGE-reranker.
            hybrid_searcher: Optional pre-configured HybridSearcher.
            lazy_load: If True, models are loaded upon first call to retrieve.
        """
        self.config = config or RetrievalConfig()
        self._embed_model = embed_model
        self._reranker = reranker
        self._searcher = hybrid_searcher

        if not lazy_load:
            self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Ensure all required components and models are loaded."""
        if self._embed_model is None:
            use_fp16 = torch.cuda.is_available()
            logger.info(
                "Loading BGE-M3 embedding model from '%s' (fp16=%s)...",
                self.config.embed_model_path,
                use_fp16,
            )
            from FlagEmbedding import BGEM3FlagModel
            self._embed_model = BGEM3FlagModel(
                str(self.config.embed_model_path),
                use_fp16=use_fp16,
            )
            logger.info("BGE-M3 model loaded successfully.")

        if self._searcher is None:
            self._searcher = HybridSearcher(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                collection_name=self.config.collection_name,
                rrf_k=self.config.rrf_k,
            )

    @property
    def embed_model(self) -> BGEM3FlagModel:
        """Get or initialize embedding model."""
        self._ensure_initialized()
        assert self._embed_model is not None
        return self._embed_model

    @property
    def searcher(self) -> HybridSearcher:
        """Get or initialize hybrid searcher."""
        self._ensure_initialized()
        assert self._searcher is not None
        return self._searcher

    @property
    def reranker(self) -> BGEReranker:
        """Get or initialize reranker."""
        self._ensure_initialized()
        if self._reranker is None:
            self._reranker = BGEReranker(model_path=self.config.reranker_model_path)
        return self._reranker

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        rrf_top_k: int | None = None,
        skip_rerank: bool | None = None,
        query_max_length: int = 512,
        *,
        merge_table_chunks: bool | None = None,
        on_rerank: Callable[[], None] | None = None,
    ) -> list[RetrievedChunk]:
        """Execute hybrid search, RRF fusion, and reranking for a single query.

        Args:
            query: User question or search text.
            top_k: Number of final results to return.
            top_k_dense: Dense search candidate count.
            top_k_sparse: Sparse search candidate count.
            rrf_top_k: RRF candidate count.
            skip_rerank: If True, bypass cross-encoder reranking.
            query_max_length: Token length truncation for query encoding.
            merge_table_chunks: If True, merge split table chunks sharing (doc_id, header) before reranking.

        Returns:
            List of top-k RetrievedChunk items.
        """
        if not query or not query.strip():
            logger.warning("Empty or whitespace-only query provided to retrieve().")
            return []

        k = top_k if top_k is not None else self.config.top_k
        k_dense = top_k_dense if top_k_dense is not None else self.config.top_k_dense
        k_sparse = top_k_sparse if top_k_sparse is not None else self.config.top_k_sparse
        k_rrf = rrf_top_k if rrf_top_k is not None else self.config.rrf_top_k
        should_skip_rerank = skip_rerank if skip_rerank is not None else self.config.skip_rerank
        should_merge_tables = (
            merge_table_chunks
            if merge_table_chunks is not None
            else self.config.merge_table_chunks
        )

        effective_rrf_top_k = max(k_rrf, k)
        if k_rrf < k:
            logger.warning(
                "rrf_top_k (%d) is smaller than top_k (%d). Automatically adjusting rrf_top_k to %d.",
                k_rrf,
                k,
                effective_rrf_top_k,
            )

        # 1. Encode query with BGE-M3
        embed_output = self.embed_model.encode(
            [query],
            batch_size=1,
            max_length=query_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vec = embed_output["dense_vecs"][0].tolist()
        sparse_weights = embed_output["lexical_weights"][0]

        # 2. Hybrid search on Qdrant + RRF
        candidates = self.searcher.search(
            dense_vec=dense_vec,
            sparse_weights=sparse_weights,
            top_k_dense=k_dense,
            top_k_sparse=k_sparse,
            rrf_top_k=effective_rrf_top_k,
        )

        if not candidates:
            return []

        if should_merge_tables:
            candidates = merge_retrieved_table_chunks(candidates)

        # 3. Rerank
        if not should_skip_rerank:
            if on_rerank is not None:
                on_rerank()
            return self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=k,
            )

        # If skipping rerank, take top-k from RRF directly
        return candidates[:k]

    def batch_retrieve(
        self,
        queries: Sequence[str],
        top_k: int | None = None,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        rrf_top_k: int | None = None,
        skip_rerank: bool | None = None,
        batch_size: int = 16,
        query_max_length: int = 512,
        *,
        merge_table_chunks: bool | None = None,
        on_rerank: Callable[[], None] | None = None,
    ) -> list[list[RetrievedChunk]]:
        """Execute hybrid search, RRF, and reranking across multiple queries.

        Args:
            queries: Sequence of query strings.
            top_k: Number of final results to return per query.
            top_k_dense: Dense search candidate count.
            top_k_sparse: Sparse search candidate count.
            rrf_top_k: RRF candidate count.
            skip_rerank: If True, bypass cross-encoder reranking.
            batch_size: Batch size for BGE-M3 query encoding.
            query_max_length: Token length truncation for query encoding.
            merge_table_chunks: If True, merge split table chunks sharing (doc_id, header) before reranking.

        Returns:
            List of top-k RetrievedChunk lists corresponding to input queries.
        """
        if not queries:
            return []

        query_list = list(queries)
        if any(not isinstance(q, str) or not q.strip() for q in query_list):
            raise ValueError("Batch queries must be nonblank strings.")
        embed_output = self.embed_model.encode(
            query_list,
            batch_size=batch_size,
            max_length=query_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        if len(embed_output["dense_vecs"]) != len(query_list) or len(embed_output["lexical_weights"]) != len(query_list):
            raise ValueError("Expected one embedding per query.")
        dense_vecs = [v.tolist() for v in embed_output["dense_vecs"]]
        sparse_weights_list = list(embed_output["lexical_weights"])

        k = top_k if top_k is not None else self.config.top_k
        k_dense = top_k_dense if top_k_dense is not None else self.config.top_k_dense
        k_sparse = top_k_sparse if top_k_sparse is not None else self.config.top_k_sparse
        k_rrf = rrf_top_k if rrf_top_k is not None else self.config.rrf_top_k
        should_skip_rerank = skip_rerank if skip_rerank is not None else self.config.skip_rerank
        should_merge_tables = (
            merge_table_chunks
            if merge_table_chunks is not None
            else self.config.merge_table_chunks
        )

        effective_rrf_top_k = max(k_rrf, k)
        if k_rrf < k:
            logger.warning(
                "rrf_top_k (%d) is smaller than top_k (%d). Automatically adjusting rrf_top_k to %d.",
                k_rrf,
                k,
                effective_rrf_top_k,
            )

        # 1. Batched hybrid search across all queries in single Qdrant batch
        all_candidates = self.searcher.batch_search(
            dense_vecs=dense_vecs,
            sparse_weights_list=sparse_weights_list,
            top_k_dense=k_dense,
            top_k_sparse=k_sparse,
            rrf_top_k=effective_rrf_top_k,
        )

        if len(all_candidates) != len(query_list):
            raise ValueError("Expected one candidate group per query.")

        if should_merge_tables:
            all_candidates = [merge_retrieved_table_chunks(cands) for cands in all_candidates]

        # 2. Batched cross-encoder reranking across all candidate pairs
        if not should_skip_rerank:
            if on_rerank is not None and any(all_candidates):
                on_rerank()
            return self.reranker.batch_rerank(
                queries=query_list,
                candidates_list=all_candidates,
                top_k=k,
            )

        return [cands[:k] for cands in all_candidates]

    def close(self) -> None:
        """Release resources and connections."""
        if self._searcher is not None:
            self._searcher.close()

    def __enter__(self) -> VacxisRetriever:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def merge_retrieved_table_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Merge retrieved table sub-chunks sharing (doc_id, header) and deduplicate overlap."""
    if len(chunks) <= 1:
        return chunks

    table_groups: dict[tuple[str, str], list[RetrievedChunk]] = defaultdict(list)
    for chunk in chunks:
        if (chunk.metadata or {}).get("has_table"):
            table_groups[(chunk.doc_id, chunk.header)].append(chunk)

    emitted_keys: set[tuple[str, str]] = set()
    merged: list[RetrievedChunk] = []
    for chunk in chunks:
        if not (chunk.metadata or {}).get("has_table"):
            merged.append(chunk)
            continue

        key = (chunk.doc_id, chunk.header)
        if key in emitted_keys:
            continue
        emitted_keys.add(key)

        group = sorted(table_groups[key], key=lambda c: c.chunk_id)
        if len(group) == 1:
            merged.append(chunk)
            continue

        merged_text = merge_table_fragments([g.text for g in group])
        merged_meta = dict(chunk.metadata or {})
        merged_ids: list[str] = []
        for g in group:
            existing = (g.metadata or {}).get("merged_chunk_ids")
            if isinstance(existing, list) and existing:
                for cid in existing:
                    cid_str = str(cid)
                    if cid_str not in merged_ids:
                        merged_ids.append(cid_str)
            elif g.chunk_id not in merged_ids:
                merged_ids.append(g.chunk_id)
        merged_meta["merged_chunk_ids"] = merged_ids
        merged.append(replace(chunk, text=merged_text, metadata=merged_meta))

    return merged
