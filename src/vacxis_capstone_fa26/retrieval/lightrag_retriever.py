"""LightRAG Knowledge Graph retriever integrated with VacxiS RetrievalConfig."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any, TypeVar

from dotenv import load_dotenv
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from vacxis_capstone_fa26.indexing.table_summarizer import group_and_merge_document_chunks
from vacxis_capstone_fa26.retrieval.lightrag_embedder import (
    DENSE_DIM,
    LocalBGEM3Embedder,
    is_local_bge_m3_model,
)
from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig, RetrievedChunk

logger = logging.getLogger("vacxis.retrieval.lightrag_retriever")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KG_CONTEXT_CHUNK_ID = "lightrag_kg_context"
KG_CONTENT_TYPE = "knowledge_graph"
_GRAPH_FIELD_SEP = "<SEP>"

_T = TypeVar("_T")


def _import_lightrag() -> tuple[Any, Any, Any]:
    """Import LightRAG symbols, adding the local ``LightRAG`` directory to sys.path if needed."""
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.utils import EmbeddingFunc

        return LightRAG, QueryParam, EmbeddingFunc
    except ImportError:
        candidates = [PROJECT_ROOT / "LightRAG", Path.cwd() / "LightRAG"]
        for repo_dir in candidates:
            if repo_dir.is_dir():
                repo_str = str(repo_dir)
                if repo_str not in sys.path:
                    sys.path.insert(0, repo_str)
                try:
                    from lightrag import LightRAG, QueryParam
                    from lightrag.utils import EmbeddingFunc

                    return LightRAG, QueryParam, EmbeddingFunc
                except ImportError:
                    continue
        raise ImportError(
            "Unable to import 'lightrag'. Ensure the local LightRAG directory exists "
            f"at '{PROJECT_ROOT / 'LightRAG'}' or install lightrag in the environment."
        )


@dataclass
class _FallbackQueryParam:
    """Minimal QueryParam substitute used only when LightRAG is mocked without the package."""

    mode: str = "hybrid"
    only_need_context: bool = True
    top_k: int = 20
    chunk_top_k: int = 5
    max_entity_tokens: int = 1500
    max_relation_tokens: int = 1500
    enable_rerank: bool = False


class _AsyncLoopRunner:
    """Dedicated background event-loop thread for synchronous LightRAG async calls."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="vacxis-lightrag-loop",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()

    def run(self, coro: Coroutine[Any, Any, _T], timeout: float | None = None) -> _T:
        """Execute a coroutine on the background event loop and wait for its result."""
        with self._lock:
            if self._closed:
                coro.close()
                raise RuntimeError("LightRAG async event loop runner is closed.")
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def close(self) -> None:
        """Stop the background event loop and join its thread idempotently."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=5.0)


def _clean_kg_text(value: Any) -> str:
    """Normalize LightRAG field strings by replacing <SEP> delimiters and trimming whitespace."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if _GRAPH_FIELD_SEP in text:
        parts = [part.strip() for part in text.split(_GRAPH_FIELD_SEP) if part.strip()]
        return "; ".join(parts)
    return text


def format_kg_context(
    entities: Sequence[dict[str, Any]],
    relationships: Sequence[dict[str, Any]],
) -> str:
    """Format LightRAG entities and relationships into a structured Vietnamese KG context block."""
    entity_lines: list[str] = []
    for item in entities:
        if not isinstance(item, dict):
            continue
        name = _clean_kg_text(item.get("entity_name"))
        if not name:
            continue
        entity_type = _clean_kg_text(item.get("entity_type"))
        desc = _clean_kg_text(item.get("description"))
        label = f"{name} ({entity_type})" if entity_type else name
        if desc:
            entity_lines.append(f"- {label}: {desc}")
        else:
            entity_lines.append(f"- {label}")

    relation_lines: list[str] = []
    for item in relationships:
        if not isinstance(item, dict):
            continue
        src = _clean_kg_text(item.get("src_id"))
        tgt = _clean_kg_text(item.get("tgt_id"))
        if not src or not tgt:
            continue
        keywords = _clean_kg_text(item.get("keywords"))
        desc = _clean_kg_text(item.get("description"))
        edge = f"{src} -> {tgt}"
        if keywords:
            edge = f"{edge} [{keywords}]"
        if desc:
            relation_lines.append(f"- {edge}: {desc}")
        else:
            relation_lines.append(f"- {edge}")

    if not entity_lines and not relation_lines:
        return ""

    sections = ["[ĐỒ THỊ TRI THỨC Y KHOA (LIGHTRAG)]"]
    if entity_lines:
        sections.append("Thực thể liên quan:\n" + "\n".join(entity_lines))
    if relation_lines:
        sections.append("Mối quan hệ:\n" + "\n".join(relation_lines))
    return "\n\n".join(sections)


def build_lightrag_chunk_id_map(
    kv_store_path: str | Path,
    chunks_parquet_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Build deterministic reverse mapping from LightRAG ``chunk-<hash>`` to Parquet metadata.

    Replays the exact document grouping and table-chunk merging order used during
    LightRAG ingestion so every ``(full_doc_id, chunk_order_index)`` maps cleanly
    back to its original Parquet ``chunk_id`` and metadata.
    """
    kv_path = Path(kv_store_path)
    parquet_path = Path(chunks_parquet_path)
    if not kv_path.is_file() or not parquet_path.is_file():
        logger.warning(
            "Skipping LightRAG chunk reverse map (kv_exists=%s, parquet_exists=%s).",
            kv_path.is_file(),
            parquet_path.is_file(),
        )
        return {}

    try:
        kv_data = json.loads(kv_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load LightRAG KV chunk store '%s': %s", kv_path, exc)
        return {}

    if not isinstance(kv_data, dict):
        return {}

    schema = pq.read_schema(parquet_path)
    desired_columns = [
        "chunk_id",
        "doc_id",
        "doc_title",
        "category",
        "parent_header",
        "header",
        "content_type",
        "text",
        "metadata",
    ]
    available_columns = [col for col in desired_columns if col in schema.names]
    if not {"chunk_id", "doc_id", "text"}.issubset(available_columns):
        logger.warning(
            "Parquet file '%s' is missing required columns for LightRAG reverse mapping.",
            parquet_path,
        )
        return {}

    table = pq.read_table(parquet_path, columns=available_columns)
    table = table.filter(pc.is_valid(table["text"]))

    doc_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table.to_pylist():
        text = (row.get("text") or "").strip()
        if not text:
            continue
        row["text"] = text
        doc_chunks[str(row["doc_id"])].append(row)

    order_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for doc_id, rows in doc_chunks.items():
        rows.sort(key=lambda r: str(r.get("chunk_id") or ""))
        merged_rows = group_and_merge_document_chunks(rows)
        for idx, rec in enumerate(merged_rows):
            meta = dict(rec.get("metadata") or {})
            merged_ids = rec.get("merged_chunk_ids")
            if isinstance(merged_ids, list) and merged_ids:
                meta["merged_chunk_ids"] = [str(cid) for cid in merged_ids]
            order_lookup[(doc_id, idx)] = {
                "chunk_id": str(rec.get("chunk_id") or ""),
                "doc_id": str(rec.get("doc_id") or doc_id),
                "doc_title": str(rec.get("doc_title") or ""),
                "category": str(rec.get("category") or ""),
                "parent_header": str(rec.get("parent_header") or ""),
                "header": str(rec.get("header") or ""),
                "content_type": str(rec.get("content_type") or "text"),
                "metadata": meta,
            }

    mapping: dict[str, dict[str, Any]] = {}
    for lr_chunk_id, entry in kv_data.items():
        if not isinstance(entry, dict):
            continue
        full_doc_id = entry.get("full_doc_id")
        order_idx = entry.get("chunk_order_index")
        if full_doc_id is None or order_idx is None:
            continue
        try:
            key = (str(full_doc_id), int(order_idx))
        except (TypeError, ValueError):
            continue
        matched = order_lookup.get(key)
        if matched is not None:
            mapping[str(lr_chunk_id)] = matched

    logger.debug(
        "Built LightRAG chunk reverse map: %d/%d entries matched.",
        len(mapping),
        len(kv_data),
    )
    return mapping


class LightRAGRetriever:
    """Knowledge-graph retriever querying LightRAG via ``aquery_data`` (retrieval only)."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        *,
        embed_model: Any | None = None,
        rag: Any | None = None,
        chunk_id_map: dict[str, dict[str, Any]] | None = None,
        lazy_load: bool = True,
    ) -> None:
        self.config = config or RetrievalConfig()
        self._embed_model = embed_model
        self._rag = rag
        self._owns_rag = rag is None
        self._initialized = rag is not None
        self._chunk_id_map = chunk_id_map
        self._loop_runner: _AsyncLoopRunner | None = None
        self._init_lock = threading.Lock()
        self.last_query_metadata: dict[str, Any] = {}

        if not lazy_load:
            self._ensure_initialized()

    def _get_loop_runner(self) -> _AsyncLoopRunner:
        if self._loop_runner is None:
            self._loop_runner = _AsyncLoopRunner()
        return self._loop_runner

    def _resolve_kv_store_path(self) -> Path:
        working_dir = Path(self.config.lightrag_working_dir)
        workspace = (self.config.lightrag_workspace or "").strip()
        if workspace:
            ws_path = working_dir / workspace / "kv_store_text_chunks.json"
            if ws_path.is_file():
                return ws_path
        return working_dir / "kv_store_text_chunks.json"

    def _load_chunk_id_map(self) -> dict[str, dict[str, Any]]:
        if self._chunk_id_map is None:
            self._chunk_id_map = build_lightrag_chunk_id_map(
                self._resolve_kv_store_path(),
                self.config.chunks_parquet_path,
            )
        return self._chunk_id_map

    def _ensure_initialized(self) -> None:
        with self._init_lock:
            if self._initialized and self._rag is not None:
                return

            load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
            LightRAG, _, EmbeddingFunc = _import_lightrag()
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed

            api_key = (
                os.getenv("LIGHTRAG_API_KEY")
                or os.getenv("OPENROUTER_API_KEY")
                or ""
            ).strip()
            llm_model = os.getenv("LIGHTRAG_LLM", "qwen/qwen-2.5-7b-instruct").strip()
            embed_model_name = os.getenv("LIGHTRAG_EMBED_MODEL", "baai/bge-m3").strip()
            base_url = os.getenv("LIGHTRAG_BASE_URL", "https://openrouter.ai/api/v1").strip()
            reasoning_effort = os.getenv("OPENAI_LLM_REASONING_EFFORT", "").strip()

            default_qdrant_url = f"http://{self.config.qdrant_host}:{self.config.qdrant_port}"
            os.environ.setdefault("QDRANT_URL", default_qdrant_url)
            os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
            os.environ.setdefault("NEO4J_USERNAME", "neo4j")
            os.environ.setdefault("NEO4J_PASSWORD", "neo4j")
            os.environ.setdefault("NEO4J_DATABASE", "neo4j")
            os.environ.setdefault("SUMMARY_LANGUAGE", "Vietnamese")
            os.environ.setdefault("ENTITY_EXTRACTION_USE_JSON", "true")

            prompt_dir = PROJECT_ROOT / "prompts"
            if (prompt_dir / "entity_type" / "entity_type_prompt.yml").is_file():
                os.environ.setdefault("PROMPT_DIR", str(prompt_dir))
                os.environ.setdefault("ENTITY_TYPE_PROMPT_FILE", "entity_type_prompt.yml")

            async def custom_llm_model_func(
                prompt: str,
                system_prompt: str | None = None,
                history_messages: list[dict[str, Any]] | None = None,
                keyword_extraction: bool = False,
                **kwargs: Any,
            ) -> str:
                if not api_key:
                    raise ValueError(
                        "Missing API key for LightRAG keyword extraction. "
                        "Configure LIGHTRAG_API_KEY or OPENROUTER_API_KEY."
                    )
                call_kwargs = dict(kwargs)
                if reasoning_effort and "reasoning_effort" not in call_kwargs:
                    call_kwargs["reasoning_effort"] = reasoning_effort

                return await openai_complete_if_cache(
                    model=llm_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages or [],
                    base_url=base_url,
                    api_key=api_key,
                    keyword_extraction=keyword_extraction,
                    **call_kwargs,
                )

            env_dim = os.getenv("LIGHTRAG_EMBED_DIM", "").strip()
            embed_dim = int(env_dim) if env_dim.isdigit() else DENSE_DIM

            if is_local_bge_m3_model(embed_model_name):
                local_embedder = LocalBGEM3Embedder(
                    model_path=self.config.embed_model_path,
                    batch_size=32,
                    max_length=8192,
                    embed_model=self._embed_model,
                )
                canonical_embed_model = "baai/bge-m3"
                embed_callable: Any = local_embedder
            else:
                canonical_embed_model = embed_model_name

                async def remote_embed_func(
                    texts: list[str], *args: Any, **kwargs: Any
                ) -> np.ndarray:
                    if not api_key:
                        raise ValueError(
                            "Missing API key for remote LightRAG embeddings. "
                            "Configure LIGHTRAG_API_KEY or OPENROUTER_API_KEY."
                        )
                    return await openai_embed.func(
                        texts=texts,
                        model=embed_model_name,
                        base_url=base_url,
                        api_key=api_key,
                        embedding_dim=embed_dim,
                        max_token_size=8192,
                        *args,
                        **kwargs,
                    )

                embed_callable = remote_embed_func

            embedding_wrapper = EmbeddingFunc(
                embedding_dim=embed_dim,
                max_token_size=8192,
                func=embed_callable,
                model_name=canonical_embed_model,
                send_dimensions=False,
            )

            working_dir = Path(self.config.lightrag_working_dir)
            workspace = self.config.lightrag_workspace
            if workspace and not (working_dir / workspace).exists():
                logger.warning(
                    "LightRAG workspace directory '%s' does not exist yet.",
                    working_dir / workspace,
                )
            working_dir.mkdir(parents=True, exist_ok=True)

            rag = LightRAG(
                working_dir=str(working_dir),
                workspace=workspace,
                vector_storage="QdrantVectorDBStorage",
                graph_storage="Neo4JStorage",
                llm_model_func=custom_llm_model_func,
                llm_model_name=llm_model,
                embedding_func=embedding_wrapper,
                vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.2},
                addon_params={
                    "language": "Vietnamese",
                    "entity_type_prompt_file": "entity_type_prompt.yml",
                },
            )
            self._get_loop_runner().run(rag.initialize_storages())
            self._rag = rag
            self._initialized = True

    def _build_query_param(
        self,
        *,
        mode: str,
        top_k: int,
        chunk_top_k: int,
    ) -> Any:
        try:
            _, QueryParam, _ = _import_lightrag()
        except ImportError:
            QueryParam = _FallbackQueryParam

        return QueryParam(
            mode=mode,
            only_need_context=True,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            max_entity_tokens=self.config.lightrag_max_entity_tokens,
            max_relation_tokens=self.config.lightrag_max_relation_tokens,
            enable_rerank=False,
        )

    def query_raw(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
    ) -> dict[str, Any]:
        """Execute LightRAG ``aquery_data`` and return the raw structured response."""
        self._ensure_initialized()
        assert self._rag is not None

        active_mode = mode or self.config.lightrag_mode
        active_top_k = top_k if top_k is not None else self.config.lightrag_top_k
        active_chunk_top_k = (
            chunk_top_k if chunk_top_k is not None else self.config.lightrag_chunk_top_k
        )
        param = self._build_query_param(
            mode=active_mode,
            top_k=active_top_k,
            chunk_top_k=active_chunk_top_k,
        )
        response = self._get_loop_runner().run(self._rag.aquery_data(query, param))
        if isinstance(response, dict):
            self.last_query_metadata = dict(response.get("metadata") or {})
            return response
        self.last_query_metadata = {}
        return {}

    def _parse_response(
        self,
        response: dict[str, Any],
        *,
        include_chunks: bool,
        max_doc_chunks: int | None,
        mode: str,
    ) -> tuple[list[RetrievedChunk], RetrievedChunk | None]:
        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, dict):
            return [], None

        entities = data.get("entities") or []
        relationships = data.get("relationships") or []
        raw_chunks = data.get("chunks") or []

        kg_text = format_kg_context(entities, relationships)
        kg_chunk: RetrievedChunk | None = None
        if kg_text:
            kg_chunk = RetrievedChunk(
                chunk_id=KG_CONTEXT_CHUNK_ID,
                doc_id="lightrag_kg",
                doc_title="Đồ thị tri thức LightRAG",
                category="knowledge_graph",
                parent_header="",
                header=f"LightRAG KG ({mode})",
                content_type=KG_CONTENT_TYPE,
                text=kg_text,
                score=1.0,
                rrf_score=0.0,
                rerank_score=None,
                point_id=None,
                metadata={
                    "source": "lightrag_kg",
                    "entity_count": len(entities),
                    "relationship_count": len(relationships),
                    "lightrag_mode": mode,
                },
            )

        doc_chunks: list[RetrievedChunk] = []
        if include_chunks and raw_chunks:
            chunk_map = self._load_chunk_id_map()
            seen_ids: set[str] = set()
            for rank, item in enumerate(raw_chunks, start=1):
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content") or "").strip()
                if not content:
                    continue
                lr_cid = str(item.get("chunk_id") or "")
                mapped = chunk_map.get(lr_cid)
                if mapped is not None:
                    orig_cid = mapped["chunk_id"]
                    doc_id = mapped["doc_id"]
                    doc_title = mapped["doc_title"]
                    category = mapped["category"]
                    parent_header = mapped["parent_header"]
                    header = mapped["header"]
                    content_type = mapped["content_type"]
                    meta = dict(mapped.get("metadata") or {})
                else:
                    orig_cid = lr_cid or f"lightrag_chunk_{rank}"
                    doc_id = str(item.get("file_path") or "")
                    doc_title = ""
                    category = ""
                    parent_header = ""
                    header = ""
                    content_type = "text"
                    meta = {}

                if orig_cid in seen_ids:
                    continue
                seen_ids.add(orig_cid)

                meta["source"] = "lightrag"
                meta["lightrag_chunk_id"] = lr_cid
                meta["lightrag_rank"] = rank
                rank_score = 1.0 / (60.0 + rank)
                doc_chunks.append(
                    RetrievedChunk(
                        chunk_id=orig_cid,
                        doc_id=doc_id,
                        doc_title=doc_title,
                        category=category,
                        parent_header=parent_header,
                        header=header,
                        content_type=content_type,
                        text=content,
                        score=rank_score,
                        rrf_score=rank_score,
                        rerank_score=None,
                        point_id=None,
                        metadata=meta,
                    )
                )
                if max_doc_chunks is not None and len(doc_chunks) >= max_doc_chunks:
                    break

        return doc_chunks, kg_chunk

    def retrieve_components(
        self,
        query: str,
        *,
        top_k: int | None = None,
        include_chunks: bool | None = None,
        mode: str | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievedChunk | None]:
        """Return ``(document_chunks, kg_context_chunk)`` separately for fusion pipelines."""
        if not query or not query.strip():
            logger.warning("Empty or whitespace-only query provided to LightRAGRetriever.")
            return [], None

        active_mode = mode or self.config.lightrag_mode
        should_include_chunks = (
            include_chunks
            if include_chunks is not None
            else self.config.lightrag_include_chunks
        )
        doc_limit = top_k if top_k is not None else self.config.lightrag_chunk_top_k
        request_chunk_top_k = max(self.config.lightrag_chunk_top_k, doc_limit) if should_include_chunks else self.config.lightrag_chunk_top_k

        response = self.query_raw(
            query.strip(),
            mode=active_mode,
            top_k=self.config.lightrag_top_k,
            chunk_top_k=request_chunk_top_k,
        )
        return self._parse_response(
            response,
            include_chunks=should_include_chunks,
            max_doc_chunks=doc_limit if should_include_chunks else 0,
            mode=active_mode,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        skip_rerank: bool | None = None,
        **kwargs: Any,
    ) -> list[RetrievedChunk]:
        """Retrieve document chunks and formatted Knowledge Graph context for a single query."""
        del skip_rerank, kwargs
        doc_chunks, kg_chunk = self.retrieve_components(query, top_k=top_k)
        if kg_chunk is not None:
            return [*doc_chunks, kg_chunk]
        return doc_chunks

    def batch_retrieve(
        self,
        queries: Sequence[str],
        top_k: int | None = None,
        skip_rerank: bool | None = None,
        **kwargs: Any,
    ) -> list[list[RetrievedChunk]]:
        """Retrieve LightRAG chunks sequentially for a sequence of queries."""
        if not queries:
            return []
        query_list = list(queries)
        if any(not isinstance(q, str) or not q.strip() for q in query_list):
            raise ValueError("Batch queries must be nonblank strings.")
        return [
            self.retrieve(query=q, top_k=top_k, skip_rerank=skip_rerank, **kwargs)
            for q in query_list
        ]

    def close(self) -> None:
        """Finalize LightRAG storages and shut down the background event-loop thread."""
        try:
            if (
                self._owns_rag
                and self._initialized
                and self._rag is not None
                and self._loop_runner is not None
                and hasattr(self._rag, "finalize_storages")
            ):
                self._loop_runner.run(self._rag.finalize_storages())
        finally:
            if self._owns_rag:
                self._initialized = False
                self._rag = None
            if self._loop_runner is not None:
                self._loop_runner.close()
                self._loop_runner = None

    def __enter__(self) -> LightRAGRetriever:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
