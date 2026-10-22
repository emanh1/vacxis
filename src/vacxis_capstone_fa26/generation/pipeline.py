"""End-to-End VacxiS RAG Pipeline coupling Retrieval with Medical Generation."""
from __future__ import annotations

import logging
from types import TracebackType
from typing import Any, Sequence

from vacxis_capstone_fa26.generation.generator import MedicalGenerator
from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer, GenerationConfig

logger = logging.getLogger("vacxis.generation.pipeline")


class VacxisRAGPipeline:
    """End-to-end RAG orchestrator for Vietnamese Vaccine Information."""

    def __init__(
        self,
        retriever: Any | None = None,
        generator: MedicalGenerator | None = None,
        retrieval_config: Any | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        if retriever is None:
            # Lazy import so generation does not require torch unless retriever is initialized
            from vacxis_capstone_fa26.retrieval.retriever import VacxisRetriever
            self.retriever = VacxisRetriever(config=retrieval_config)
        else:
            self.retriever = retriever

        self.generator = generator or MedicalGenerator(config=generation_config)

    def __enter__(self) -> VacxisRAGPipeline:
        if hasattr(self.retriever, "__enter__"):
            self.retriever.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if hasattr(self.retriever, "__exit__"):
            self.retriever.__exit__(exc_type, exc_val, exc_tb)

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        override_chunks: Sequence[Any] | None = None,
        skip_rerank: bool | None = None,
    ) -> GeneratedAnswer:
        """Answer a single user query.

        Args:
            query: Medical question text in Vietnamese.
            top_k: Optional number of context chunks to retrieve.
            override_chunks: Optional pre-retrieved chunks (bypasses retrieval).
            skip_rerank: If True, bypass cross-encoder reranking during retrieval.

        Returns:
            GeneratedAnswer with synthesized text, verified citations, and metadata.
        """
        if override_chunks is not None:
            chunks = list(override_chunks)
        else:
            chunks = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                skip_rerank=skip_rerank,
            )

        return self.generator.generate(query=query, chunks=chunks)

    def answer_batch(
        self,
        queries: Sequence[str],
        top_k: int | None = None,
        override_chunks_per_query: Sequence[Sequence[Any]] | None = None,
        skip_rerank: bool | None = None,
    ) -> list[GeneratedAnswer]:
        """Answer a batch of queries efficiently using batched retrieval and generation."""
        if override_chunks_per_query is not None:
            chunks_batch = list(override_chunks_per_query)
        else:
            chunks_batch = self.retriever.batch_retrieve(
                queries=queries,
                top_k=top_k,
                skip_rerank=skip_rerank,
            )

        return self.generator.generate_batch(queries=queries, chunks_per_query=chunks_batch)
