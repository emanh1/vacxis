"""End-to-End VacxiS RAG Pipeline coupling Retrieval with Medical Generation."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from itertools import zip_longest
from time import perf_counter
from types import TracebackType
from typing import Any, Sequence

from vacxis_capstone_fa26.generation.generator import MedicalGenerator
from vacxis_capstone_fa26.generation.prompts import get_chunk_field
from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer, GenerationConfig
from vacxis_capstone_fa26.query import (
    Clarification, ClarifyPlan, ConversationMemoryConfig, ConversationMessage, QueryPlanner,
)
from vacxis_capstone_fa26.query.memory import limit_history


class VacxisRAGPipeline:
    """End-to-end RAG orchestrator for Vietnamese Vaccine Information."""

    def __init__(
        self,
        retriever: Any | None = None,
        generator: MedicalGenerator | None = None,
        retrieval_config: Any | None = None,
        generation_config: GenerationConfig | None = None,
        *,
        query_planner: QueryPlanner | None = None,
    ) -> None:
        if retriever is None:
            # Lazy import so generation does not require torch unless retriever is initialized
            from vacxis_capstone_fa26.retrieval.factory import build_retriever
            self.retriever = build_retriever(retrieval_config)
        else:
            self.retriever = retriever

        self.generator = generator or MedicalGenerator(config=generation_config)
        self.query_planner = query_planner

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
        try:
            if hasattr(self.retriever, "__exit__"):
                self.retriever.__exit__(exc_type, exc_val, exc_tb)
        finally:
            if hasattr(self.generator, "close"):
                self.generator.close()

    def run_turn(
        self,
        original_query: str,
        *,
        clarifications: Sequence[Clarification] = (),
        history: Sequence[ConversationMessage] = (),
        use_memory: bool = True,
        top_k: int | None = None,
        skip_rerank: bool | None = None,
        on_progress: Callable[[str, int | None], None] | None = None,
    ) -> GeneratedAnswer | ClarifyPlan:
        """Plan and execute one turn, or return a question for the caller to ask."""
        if self.query_planner is None:
            raise ValueError("run_turn requires a query planner.")
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError("top_k must be a positive integer.")
        if type(use_memory) is not bool:
            raise TypeError("use_memory must be a boolean.")
        memory_config = getattr(self.query_planner, "memory_config", None)
        if not isinstance(memory_config, ConversationMemoryConfig):
            memory_config = ConversationMemoryConfig()
        selected = limit_history(history, memory_config) if use_memory else ()

        if on_progress is not None:
            on_progress("planning", None)
        started = perf_counter()
        plan = self.query_planner.plan(original_query, clarifications, history=selected)
        planner_latency = perf_counter() - started
        if isinstance(plan, ClarifyPlan):
            return plan

        retrieval_options = {}
        if on_progress is not None:
            on_progress("searching", None)
            retrieval_options["on_rerank"] = lambda: on_progress("reranking", None)

        if len(plan.search_queries) == 1:
            chunks = self.retriever.retrieve(
                query=plan.resolved_query, top_k=top_k, skip_rerank=skip_rerank, **retrieval_options,
            )
        else:
            limit = top_k if top_k is not None else self.retriever.config.top_k
            groups = self.retriever.batch_retrieve(
                queries=plan.search_queries, top_k=limit, skip_rerank=skip_rerank, **retrieval_options,
            )
            if len(groups) != len(plan.search_queries):
                raise ValueError("Expected one retrieval group per search query.")
            chunks = self._merge_query_results(groups, limit)

        if on_progress is not None:
            on_progress("generating", len(chunks))
        answer = self.generator.generate(query=plan.resolved_query, chunks=chunks)
        return replace(answer, query=original_query, query_context={
            "resolved_query": plan.resolved_query,
            "search_queries": [plan.resolved_query] if len(plan.search_queries) == 1 else list(plan.search_queries),
            "use_memory": use_memory,
            "history_messages_supplied": len(history),
            "history_messages_included": len(selected),
            "history_reduced": use_memory and len(selected) < len(history),
            "memory_limits": asdict(memory_config),
            "planner_model": self.query_planner.model,
            "planner_latency_seconds": planner_latency,
        })

    @staticmethod
    def _merge_query_results(groups: Sequence[Sequence[Any]], limit: int) -> list[Any]:
        chunks: dict[str, Any] = {}
        for rank in zip_longest(*groups):
            for chunk in rank:
                if chunk is not None:
                    chunks.setdefault(str(get_chunk_field(chunk, "chunk_id", "")), chunk)
                    if len(chunks) >= limit:
                        return list(chunks.values())
        return list(chunks.values())

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

        if len(chunks_batch) != len(queries):
            raise ValueError("Expected one chunk group per query.")
        answers = self.generator.generate_batch(queries=queries, chunks_per_query=chunks_batch)
        if len(answers) != len(queries):
            raise ValueError("Expected one generated answer per query.")
        return answers
