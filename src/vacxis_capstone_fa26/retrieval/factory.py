"""Factory for instantiating configured VacxiS retrievers lazily."""
from __future__ import annotations

from typing import Any

from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig


def build_retriever(
    config: RetrievalConfig | None = None,
    *,
    chunk_texts: dict[str, str] | None = None,
    **kwargs: Any,
) -> Any:
    """Build a retriever instance according to ``config.mode``.

    Supported modes:
    - ``naive``: lexical BM25 over ``chunk_texts`` (or ``config.chunks_parquet_path``).
    - ``advanced``: single-query BGE-M3 dense search (``DenseRetriever``).
    - ``hybrid``: BGE-M3 dense+sparse hybrid search + RRF + BGE reranker (``VacxisRetriever``).
    - ``lightrag``: LightRAG Knowledge Graph retriever via ``aquery_data`` (``LightRAGRetriever``).
    - ``hybrid_lightrag``: combined Hybrid + LightRAG retriever with shared BGE-M3 (``HybridLightRAGRetriever``).
    """
    cfg = config or RetrievalConfig()
    mode = cfg.mode

    if mode == "naive":
        from vacxis_capstone_fa26.evaluation.datasets import load_chunk_texts
        from vacxis_capstone_fa26.retrieval import bm25 as bm25_mod

        texts = (
            chunk_texts
            if chunk_texts is not None
            else load_chunk_texts(cfg.chunks_parquet_path)
        )
        return bm25_mod.BM25Retriever(texts)

    if mode == "advanced":
        from vacxis_capstone_fa26.retrieval import dense as dense_mod

        return dense_mod.DenseRetriever(config=cfg, **kwargs)

    if mode == "hybrid":
        from vacxis_capstone_fa26.retrieval import retriever as retriever_mod

        return retriever_mod.VacxisRetriever(config=cfg, **kwargs)

    if mode == "lightrag":
        from vacxis_capstone_fa26.retrieval import lightrag_retriever as lightrag_mod

        return lightrag_mod.LightRAGRetriever(config=cfg, **kwargs)

    if mode == "hybrid_lightrag":
        from vacxis_capstone_fa26.retrieval import hybrid_lightrag as hybrid_lr_mod

        return hybrid_lr_mod.HybridLightRAGRetriever(config=cfg, **kwargs)

    raise ValueError(
        f"Unsupported retriever mode: {mode!r}. "
        "Expected one of: 'naive', 'advanced', 'hybrid', 'lightrag', 'hybrid_lightrag'."
    )
