"""Table-aware preprocessing pipeline for Qdrant hybrid indexing."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
from typing import Any, Literal

from vacxis_capstone_fa26.indexing.table_summarizer import (
    _strip_prefix,
    build_table_summary_prompt,
    classify_table_kind,
    group_and_merge_document_chunks,
    load_table_summary_cache,
    make_table_cache_key,
    save_table_summary_cache,
)

logger = logging.getLogger("vacxis.indexing.table_pipeline")

TableStrategy = Literal["raw", "merge", "summarize", "summarize_dual"]
VALID_TABLE_STRATEGIES: tuple[str, ...] = ("raw", "merge", "summarize", "summarize_dual")
DEFAULT_TABLE_CACHE_PATH = Path("lightrag_storage/table_summaries_cache.json")
DEFAULT_TABLE_SUMMARY_MODEL = "gpt-oss-120b"


def resolve_table_summary_model(summary_model: str | None = None) -> str:
    """Resolve the model name used for table summary cache keys and fallback generation.

    Priority: explicit ``summary_model`` -> ``LIGHTRAG_LLM`` env var -> ``gpt-oss-120b``.
    This is intentionally decoupled from ``VACXIS_GENERATION_MODEL``.
    """
    if summary_model and summary_model.strip():
        return summary_model.strip()
    env_model = os.getenv("LIGHTRAG_LLM", "").strip()
    if env_model:
        return env_model
    return DEFAULT_TABLE_SUMMARY_MODEL


def _record_has_table(record: dict[str, Any]) -> bool:
    """Check whether a chunk record contains a table."""
    meta = record.get("metadata")
    if isinstance(meta, dict) and "has_table" in meta:
        return bool(meta["has_table"])
    return bool(record.get("has_table", False))


def _run_coro_sync(coro: Awaitable[str]) -> str:
    """Run an awaitable synchronously, safely handling an already-running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()  # type: ignore[arg-type]


def prepare_records_for_indexing(
    records: list[dict[str, Any]],
    strategy: TableStrategy = "raw",
    *,
    cache_path: str | Path = DEFAULT_TABLE_CACHE_PATH,
    summary_model: str | None = None,
    llm_func: Callable[..., Awaitable[str]] | None = None,
    allow_llm_fallback: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Transform raw Parquet chunk records according to the selected table strategy.

    Strategies:
    - ``raw``: return shallow copies of original records unchanged.
    - ``merge``: merge consecutive split table chunks sharing ``(doc_id, header)``.
    - ``summarize``: merge split tables and replace text/embed_text with entity-rich prose.
    - ``summarize_dual``: merge split tables, embed entity-rich prose for retrieval
      (``embed_text``), and store prose + original merged table body in ``text`` for generation.

    Returns:
        Tuple of ``(prepared_records, stats)`` with counts of inputs, outputs,
        table chunks, cache hits, cache misses, and LLM calls.
    """
    if strategy not in VALID_TABLE_STRATEGIES:
        raise ValueError(
            f"Unsupported table strategy: {strategy!r}. "
            f"Expected one of {VALID_TABLE_STRATEGIES}."
        )

    stats: dict[str, int] = {
        "total_in": len(records),
        "total_out": len(records),
        "tables_total": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "llm_calls": 0,
    }
    if not records:
        return [], stats

    if strategy == "raw":
        prepared_raw: list[dict[str, Any]] = []
        for rec in records:
            copied = dict(rec)
            if isinstance(rec.get("metadata"), dict):
                copied["metadata"] = dict(rec["metadata"])
            if _record_has_table(copied):
                stats["tables_total"] += 1
            prepared_raw.append(copied)
        stats["total_out"] = len(prepared_raw)
        return prepared_raw, stats

    # Group records by doc_id preserving original document order and sorting chunks by chunk_id
    # so merged HTML/Markdown table text matches table_summaries_cache.json deterministically.
    doc_groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        copied = dict(rec)
        copied["text"] = str(rec.get("text") or "").strip()
        if isinstance(rec.get("metadata"), dict):
            copied["metadata"] = dict(rec["metadata"])
        doc_key = str(copied.get("doc_id", ""))
        doc_groups.setdefault(doc_key, []).append(copied)

    merged_all: list[dict[str, Any]] = []
    for doc_records in doc_groups.values():
        doc_records.sort(key=lambda r: str(r.get("chunk_id", "")))
        merged_all.extend(group_and_merge_document_chunks(doc_records))

    resolved_model = resolve_table_summary_model(summary_model)
    resolved_cache_path = Path(cache_path)
    needs_summary = strategy in ("summarize", "summarize_dual")
    cache: dict[str, str] = load_table_summary_cache(resolved_cache_path) if needs_summary else {}
    cache_dirty = False

    prepared: list[dict[str, Any]] = []
    for rec in merged_all:
        meta = dict(rec.get("metadata") or {})
        meta["table_strategy"] = strategy

        merged_ids = rec.pop("merged_chunk_ids", None)
        is_table = _record_has_table(rec)

        if isinstance(merged_ids, list) and merged_ids:
            meta["merged_chunk_ids"] = [str(cid) for cid in merged_ids]
        elif is_table:
            meta["merged_chunk_ids"] = [str(rec.get("chunk_id", ""))]

        if is_table:
            stats["tables_total"] += 1
            raw_merged_text = str(rec.get("text") or "").strip()
            prefix, merged_raw_body = _strip_prefix(raw_merged_text)
            doc_title = str(rec.get("doc_title") or rec.get("doc_id") or "Vắc xin")
            header = str(rec.get("header") or "Bảng dữ liệu")
            category = str(rec.get("category") or "")
            table_kind = classify_table_kind(
                header=header,
                content=merged_raw_body,
                category=category,
            )
            meta["table_kind"] = table_kind

            if needs_summary:
                cache_key = make_table_cache_key(
                    doc_id=str(rec.get("doc_id", "")),
                    header=header,
                    table_kind=table_kind,
                    merged_text=raw_merged_text,
                    model_name=resolved_model,
                )
                cached_prose = str(cache.get(cache_key) or "").strip()
                prose = ""
                if cached_prose:
                    prose = cached_prose
                    stats["cache_hits"] += 1
                else:
                    stats["cache_misses"] += 1
                    if allow_llm_fallback and llm_func is not None:
                        system_prompt, user_prompt = build_table_summary_prompt(
                            doc_title=doc_title,
                            header=header,
                            table_body=merged_raw_body,
                            table_kind=table_kind,
                        )
                        prose = str(
                            _run_coro_sync(
                                llm_func(prompt=user_prompt, system_prompt=system_prompt)
                            )
                        ).strip()
                        stats["llm_calls"] += 1
                        if prose:
                            cache[cache_key] = prose
                            cache_dirty = True
                    else:
                        logger.warning(
                            "Table summary cache miss for chunk_id=%s (doc_id=%s, header=%r, model=%s); "
                            "falling back to merged table text.",
                            rec.get("chunk_id"),
                            rec.get("doc_id"),
                            header,
                            resolved_model,
                        )

                if prose:
                    effective_prefix = prefix or f"[CHỦ ĐỀ: {doc_title}] > [MỤC: {header}]"
                    prose_with_prefix = f"{effective_prefix}\n{prose}"
                    rec["embed_text"] = prose_with_prefix
                    if strategy == "summarize":
                        rec["text"] = prose_with_prefix
                    else:
                        rec["text"] = (
                            f"{prose_with_prefix}\n\n"
                            f"[BẢNG DỮ LIỆU GỐC]\n{merged_raw_body}"
                        )
                else:
                    rec["embed_text"] = raw_merged_text
                    rec["text"] = raw_merged_text

        rec["metadata"] = meta
        prepared.append(rec)

    if cache_dirty:
        save_table_summary_cache(resolved_cache_path, cache)

    stats["total_out"] = len(prepared)
    return prepared, stats
