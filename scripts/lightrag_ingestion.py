"""LightRAG ingestion pipeline for pre-chunked vaccine data.

Ingests pre-partitioned chunks from chunks.parquet into LightRAG using:
- QdrantVectorDBStorage for vector collections (chunks, entities, relationships)
- Neo4JStorage for the biomedical knowledge graph
- OpenAI-compatible LLM & Embedding endpoints (OpenRouter, vLLM, etc.)
- Custom entity types from entity_type_prompt.yml / prompts/entity_type/entity_type_prompt.yml
- Reasoning effort configuration for reasoning models (OPENAI_LLM_REASONING_EFFORT)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure LightRAG repository and sibling scripts are discoverable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
LIGHTRAG_REPO_DIR = PROJECT_ROOT / "LightRAG"
for _path in (SCRIPTS_DIR, LIGHTRAG_REPO_DIR):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from dotenv import load_dotenv

from vacxis_capstone_fa26.indexing.table_summarizer import (
    classify_table_kind,
    group_and_merge_document_chunks,
    load_table_summary_cache,
    save_table_summary_cache,
    summarize_table_to_entity_prose,
)

# Load local environment configuration if available
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


def load_parquet_chunks(
    parquet_path: str | Path,
    target_doc_id: str | None = None,
    limit_docs: int | None = None,
    *,
    merge_tables: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Read chunks from Parquet, group by doc_id, and optionally merge split table chunks."""
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    table = pq.read_table(
        path,
        columns=[
            "chunk_id",
            "doc_id",
            "doc_title",
            "category",
            "parent_header",
            "header",
            "content_type",
            "text",
            "metadata",
        ],
    )

    valid_mask = pc.is_valid(table["text"])
    if target_doc_id:
        doc_mask = pc.equal(table["doc_id"], target_doc_id)
        valid_mask = pc.and_(valid_mask, doc_mask)

    table = table.filter(valid_mask)

    doc_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table.to_pylist():
        text = (row.get("text") or "").strip()
        if not text:
            continue
        row["text"] = text
        doc_chunks[row["doc_id"]].append(row)

    for doc_id in doc_chunks:
        doc_chunks[doc_id].sort(key=lambda r: r.get("chunk_id", ""))
        if merge_tables:
            doc_chunks[doc_id] = group_and_merge_document_chunks(doc_chunks[doc_id])

    if limit_docs and limit_docs > 0:
        limited_keys = list(doc_chunks.keys())[:limit_docs]
        doc_chunks = {k: doc_chunks[k] for k in limited_keys}

    return dict(doc_chunks)


def load_ingested_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load the set of doc_ids already ingested into LightRAG."""
    if not checkpoint_path.is_file():
        return set()
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return set(data) if isinstance(data, list) else set()
    except (json.JSONDecodeError, OSError):
        return set()


def save_ingested_checkpoint(checkpoint_path: Path, completed_docs: set[str]) -> None:
    """Persist completed doc_ids atomically to disk for resume support."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(sorted(completed_docs), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(checkpoint_path)


async def check_qdrant_connection(qdrant_url: str) -> bool:
    """Verify connectivity to Qdrant vector database."""
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=qdrant_url, timeout=5)
        collections = client.get_collections()
        col_names = [c.name for c in collections.collections]
        print(f"[OK] Qdrant connected at {qdrant_url}. Existing collections: {col_names}")
        return True
    except Exception as exc:
        print(f"[ERROR] Failed to connect to Qdrant at {qdrant_url}: {exc}")
        return False


async def check_neo4j_connection(
    uri: str, username: str, password: str, database: str = "neo4j"
) -> bool:
    """Verify connectivity and authentication to Neo4j database."""
    from neo4j import AsyncGraphDatabase

    driver = None
    try:
        driver = AsyncGraphDatabase.driver(
            uri,
            auth=(username, password),
            connection_timeout=5.0,
        )
        async with driver.session(database=database) as session:
            result = await session.run("RETURN 1 AS test")
            record = await result.single()
            if record and record["test"] == 1:
                print(f"[OK] Neo4j connected and authenticated at {uri} (database: '{database}')")
                return True
            return False
    except Exception as exc:
        print(f"[ERROR] Failed to connect/authenticate to Neo4j at {uri}: {exc}")
        return False
    finally:
        if driver:
            await driver.close()


def prepare_prompt_directory(entity_prompt_path: Path) -> Path:
    """Ensure prompt directory contains the entity type definition."""
    prompt_dir = PROJECT_ROOT / "prompts" / "entity_type"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    target_path = prompt_dir / "entity_type_prompt.yml"

    if entity_prompt_path.is_file() and entity_prompt_path.resolve() != target_path.resolve():
        shutil.copy(entity_prompt_path, target_path)

    if not target_path.is_file():
        raise FileNotFoundError(f"Entity prompt profile not found at: {target_path}")

    os.environ["PROMPT_DIR"] = str(PROJECT_ROOT / "prompts")
    os.environ["ENTITY_TYPE_PROMPT_FILE"] = "entity_type_prompt.yml"
    return target_path


def is_local_bge_m3_model(embed_model: str) -> bool:
    """Return True when the configured LightRAG embedding model maps to local BGE-M3."""
    return embed_model.strip().lower() in {"baai/bge-m3", "bge-m3", "models/bge-m3"}


def build_embedding_config(
    embed_model: str,
    *,
    embed_model_path: str | Path,
    base_url: str,
    api_key: str,
    embed_dim: int,
) -> tuple[str, Any]:
    """Build (canonical_model_name, async_embed_callable) for LightRAG EmbeddingFunc.

    Uses local ``models/bge-m3`` via ``LocalBGEM3Embedder`` when ``embed_model``
    targets BGE-M3 so LightRAG shares the exact same embedding weights as the
    main pipeline without external API calls.
    """
    if is_local_bge_m3_model(embed_model):
        from reembed_lightrag_qdrant import LocalBGEM3Embedder

        local_embedder = LocalBGEM3Embedder(
            model_path=embed_model_path,
            batch_size=32,
            max_length=8192,
        )
        return "baai/bge-m3", local_embedder

    from lightrag.llm.openai import openai_embed

    async def remote_embed_func(
        texts: list[str], *args: Any, **kwargs: Any
    ) -> np.ndarray:
        return await openai_embed.func(
            texts=texts,
            model=embed_model,
            base_url=base_url,
            api_key=api_key,
            embedding_dim=embed_dim,
            max_token_size=8192,
            *args,
            **kwargs,
        )

    return embed_model, remote_embed_func


async def probe_embedding_dimension(
    base_url: str,
    api_key: str,
    embed_model: str,
) -> int:
    """Probe the embedding endpoint to detect vector dimension."""
    from lightrag.llm.openai import openai_embed

    sample_text = ["Kiểm tra số chiều vector vắc xin"]
    vectors = await openai_embed.func(
        texts=sample_text,
        model=embed_model,
        base_url=base_url,
        api_key=api_key,
    )
    dim = int(vectors.shape[1])
    print(f"[OK] Successfully probed {embed_model}: dimension is {dim}")
    return dim


async def run_pipeline(args: argparse.Namespace) -> None:
    """Main ingestion execution loop."""
    parquet_path = Path(args.parquet)
    entity_prompt_path = Path(args.entity_prompt)

    api_key = (
        os.getenv("LIGHTRAG_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or ""
    ).strip()
    llm_model = os.getenv("LIGHTRAG_LLM", "qwen/qwen-2.5-7b-instruct").strip()
    embed_model = os.getenv("LIGHTRAG_EMBED_MODEL", "baai/bge-m3").strip()
    base_url = os.getenv("LIGHTRAG_BASE_URL", "https://openrouter.ai/api/v1").strip()
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j").strip()
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "neo4j").strip()
    neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j").strip()

    reasoning_effort = (
        args.reasoning_effort
        or os.getenv("OPENAI_LLM_REASONING_EFFORT", "")
    ).strip()

    # Export variables required by LightRAG internal storages and roles
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_USERNAME"] = neo4j_user
    os.environ["NEO4J_PASSWORD"] = neo4j_pass
    os.environ["NEO4J_DATABASE"] = neo4j_db
    os.environ["SUMMARY_LANGUAGE"] = "Vietnamese"
    os.environ["ENTITY_EXTRACTION_USE_JSON"] = "true"
    if reasoning_effort:
        os.environ["OPENAI_LLM_REASONING_EFFORT"] = reasoning_effort

    print("Preflight check for services and data...")
    qdrant_ok = await check_qdrant_connection(qdrant_url)
    neo4j_ok = await check_neo4j_connection(neo4j_uri, neo4j_user, neo4j_pass, neo4j_db)

    merge_tables = args.table_mode != "raw"
    doc_chunks = load_parquet_chunks(
        parquet_path=parquet_path,
        target_doc_id=args.doc_id,
        limit_docs=args.limit_docs,
        merge_tables=merge_tables,
    )
    total_docs = len(doc_chunks)
    total_chunks = sum(len(c) for c in doc_chunks.values())
    table_records = [
        r for recs in doc_chunks.values() for r in recs
        if (r.get("metadata") or {}).get("has_table", False)
    ]
    clinical_tables = sum(
        1 for r in table_records
        if classify_table_kind(r.get("header", ""), r["text"], r.get("category", "")) == "clinical"
    )
    stat_tables = len(table_records) - clinical_tables
    print(
        f"[OK] Parquet loaded ({args.table_mode} mode): {total_docs} documents, "
        f"{total_chunks} chunks ({len(table_records)} tables: "
        f"{clinical_tables} clinical, {stat_tables} statistical)."
    )

    working_dir = Path(args.working_dir)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else working_dir / f"ingested_docs_{args.workspace}.json"
    )
    completed_docs = load_ingested_checkpoint(checkpoint_path) if args.resume else set()
    if args.resume and completed_docs:
        pending_count = sum(1 for d in doc_chunks if d not in completed_docs)
        print(
            f"[OK] Resume enabled ({checkpoint_path}): "
            f"{len(completed_docs)} already ingested, {pending_count} remaining."
        )

    if reasoning_effort:
        print(f"[OK] Reasoning effort configured: '{reasoning_effort}'")

    if not api_key:
        print("[WARNING] LIGHTRAG_API_KEY / OPENROUTER_API_KEY is not set.")

    prepare_prompt_directory(entity_prompt_path)

    if args.check_only:
        print("Check-only mode completed.")
        return

    if not qdrant_ok or not neo4j_ok:
        raise RuntimeError("Preflight check failed: Qdrant or Neo4j connection is not ready.")

    if not api_key:
        raise ValueError("Missing API key. Please configure LIGHTRAG_API_KEY in .env.")

    # Determine embedding dimension
    env_dim = os.getenv("LIGHTRAG_EMBED_DIM")
    if env_dim and env_dim.strip().isdigit():
        embed_dim = int(env_dim.strip())
    elif is_local_bge_m3_model(embed_model):
        embed_dim = 1024
    else:
        print("Probing embedding model dimension...")
        embed_dim = await probe_embedding_dimension(
            base_url=base_url,
            api_key=api_key,
            embed_model=embed_model,
        )

    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc

    async def custom_llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        keyword_extraction: bool = False,
        **kwargs: Any,
    ) -> str:
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

    canonical_embed_model, custom_embed_func = build_embedding_config(
        embed_model,
        embed_model_path=args.embed_model_path,
        base_url=base_url,
        api_key=api_key,
        embed_dim=embed_dim,
    )
    use_local_bge_m3 = is_local_bge_m3_model(embed_model)
    embed_source = args.embed_model_path if use_local_bge_m3 else base_url
    print(
        f"[OK] LightRAG embedding configured: model_name='{canonical_embed_model}', "
        f"local_bge_m3={use_local_bge_m3} ({embed_source})"
    )

    embedding_wrapper = EmbeddingFunc(
        embedding_dim=embed_dim,
        max_token_size=8192,
        func=custom_embed_func,
        model_name=canonical_embed_model,
        send_dimensions=False,
    )

    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing LightRAG with workspace '{args.workspace}'...")
    rag = LightRAG(
        working_dir=str(working_dir),
        workspace=args.workspace,
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

    await rag.initialize_storages()
    print("Storages initialized successfully. Starting custom chunk ingestion...")

    summary_model = (args.table_summary_model or llm_model).strip()
    cache_path = Path(args.table_cache)
    table_cache = load_table_summary_cache(cache_path) if args.table_mode == "summary" else {}
    semaphore = asyncio.Semaphore(max(1, args.table_concurrency))

    async def table_summary_llm_func(prompt: str, system_prompt: str | None = None) -> str:
        async with semaphore:
            return await openai_complete_if_cache(
                model=summary_model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=[],
                base_url=base_url,
                api_key=api_key,
            )

    try:
        processed = 0
        ingested_now = 0
        skipped_now = 0
        for doc_id, records in doc_chunks.items():
            processed += 1
            if args.resume and doc_id in completed_docs:
                skipped_now += 1
                print(f"[{processed}/{total_docs}] [SKIP] Document '{doc_id}' already ingested.")
                continue

            if args.table_mode == "skip":
                prepared_texts = [
                    r["text"] for r in records
                    if not (r.get("metadata") or {}).get("has_table", False)
                ]
            elif args.table_mode == "summary":
                async def _prepare_record(rec: dict[str, Any]) -> str:
                    if not (rec.get("metadata") or {}).get("has_table", False):
                        return rec["text"]
                    return await summarize_table_to_entity_prose(
                        record=rec,
                        llm_func=table_summary_llm_func,
                        cache=table_cache,
                        model_name=summary_model,
                    )

                prepared_texts = list(await asyncio.gather(*(_prepare_record(r) for r in records)))
                save_table_summary_cache(cache_path, table_cache)
            else:
                prepared_texts = [r["text"] for r in records]

            if not prepared_texts:
                continue

            print(
                f"[{processed}/{total_docs}] Ingesting document '{doc_id}' "
                f"({len(prepared_texts)} chunks)..."
            )
            full_text = "\n\n".join(prepared_texts)
            await rag.ainsert_custom_chunks(
                full_text=full_text,
                text_chunks=prepared_texts,
                doc_id=doc_id,
            )
            completed_docs.add(doc_id)
            save_ingested_checkpoint(checkpoint_path, completed_docs)
            ingested_now += 1

        print(
            f"Ingestion completed: {ingested_now} newly ingested, "
            f"{skipped_now} skipped (resume), {len(completed_docs)} total in checkpoint."
        )
    finally:
        if args.table_mode == "summary" and table_cache:
            save_table_summary_cache(cache_path, table_cache)
        print("Finalizing LightRAG storages...")
        await rag.finalize_storages()
        print("Storages finalized successfully.")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest pre-chunked vaccine Parquet data into LightRAG (Qdrant + Neo4j)."
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=str(PROJECT_ROOT / "chunks_storage" / "chunks.parquet"),
        help="Path to chunks.parquet file.",
    )
    parser.add_argument(
        "--entity-prompt",
        type=str,
        default=str(
            PROJECT_ROOT / "entity_type_prompt.yml"
            if (PROJECT_ROOT / "entity_type_prompt.yml").is_file()
            else PROJECT_ROOT / "prompts" / "entity_type" / "entity_type_prompt.yml"
        ),
        help="Path to entity_type_prompt.yml profile.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="vacxis",
        help="Workspace name for Qdrant collection isolation and Neo4j labels.",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=str(PROJECT_ROOT / "lightrag_storage"),
        help="Working directory for LightRAG internal KV cache and locks.",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Limit number of documents to ingest (useful for testing).",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Specific doc_id to ingest.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Reasoning effort level (e.g. low, medium, high). Overrides OPENAI_LLM_REASONING_EFFORT.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run preflight connectivity checks and print data statistics without executing ingestion.",
    )
    parser.add_argument(
        "--table-mode",
        choices=["summary", "merged-raw", "raw", "skip"],
        default="summary",
        help="How to process has_table=True chunks before LightRAG NER (default: summary).",
    )
    parser.add_argument(
        "--table-cache",
        type=str,
        default=str(PROJECT_ROOT / "lightrag_storage" / "table_summaries_cache.json"),
        help="Path to JSON cache for LLM table-to-prose summaries.",
    )
    parser.add_argument(
        "--table-concurrency",
        type=int,
        default=8,
        help="Max concurrent LLM calls when summarizing tables.",
    )
    parser.add_argument(
        "--table-summary-model",
        type=str,
        default=None,
        help="Optional LLM model override for table-to-prose summarization (defaults to LIGHTRAG_LLM).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip documents already recorded in the ingestion checkpoint (use --no-resume to re-ingest all).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to doc_id checkpoint JSON (defaults to <working_dir>/ingested_docs_<workspace>.json).",
    )
    parser.add_argument(
        "--embed-model-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "bge-m3"),
        help="Local BGE-M3 directory used when LIGHTRAG_EMBED_MODEL targets bge-m3.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(run_pipeline(args))
