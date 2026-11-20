"""Unit tests for table-aware indexing pipeline and Qdrant dual payload upsert."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from vacxis_capstone_fa26.indexing.indexer import DENSE_DIM, ingest_to_qdrant
from vacxis_capstone_fa26.indexing.storage import write_chunks_to_parquet
from vacxis_capstone_fa26.indexing.table_pipeline import (
    DEFAULT_TABLE_SUMMARY_MODEL,
    prepare_records_for_indexing,
    resolve_table_summary_model,
)
from vacxis_capstone_fa26.indexing.table_summarizer import (
    classify_table_kind,
    make_table_cache_key,
    merge_html_table_fragments,
    save_table_summary_cache,
)


def _sample_records() -> list[dict]:
    part1 = (
        "[CHỦ ĐỀ: Vaxneuvance] > [MỤC: Liều dùng]\n"
        "<table><thead><tr><th>Đối tượng</th><th>Liều</th></tr></thead>"
        "<tbody><tr><td>Trẻ 6 tuần</td><td>0.5 ml</td></tr></tbody></table>"
    )
    part2 = (
        "[CHỦ ĐỀ: Vaxneuvance] > [MỤC: Liều dùng]\n"
        "<table><thead><tr><th>Đối tượng</th><th>Liều</th></tr></thead>"
        "<tbody><tr><td>Trẻ 6 tuần</td><td>0.5 ml</td></tr>"
        "<tr><td>Người lớn</td><td>0.5 ml</td></tr></tbody></table>"
    )
    return [
        {
            "chunk_id": "vax_c_0000",
            "doc_id": "vax",
            "doc_title": "Vaxneuvance",
            "category": "chỉ định",
            "parent_header": "CHỈ ĐỊNH",
            "header": "Chỉ định",
            "content_type": "text",
            "text": "[CHỦ ĐỀ: Vaxneuvance] > [MỤC: Chỉ định]\nPhòng bệnh phế cầu.",
            "metadata": {"has_table": False, "source_file": "vax.md"},
        },
        {
            "chunk_id": "vax_c_0001",
            "doc_id": "vax",
            "doc_title": "Vaxneuvance",
            "category": "liều lượng và cách dùng",
            "parent_header": "LIỀU DÙNG",
            "header": "Liều dùng",
            "content_type": "table",
            "text": part1,
            "metadata": {"has_table": True, "source_file": "vax.md"},
        },
        {
            "chunk_id": "vax_c_0002",
            "doc_id": "vax",
            "doc_title": "Vaxneuvance",
            "category": "liều lượng và cách dùng",
            "parent_header": "LIỀU DÙNG",
            "header": "Liều dùng",
            "content_type": "table",
            "text": part2,
            "metadata": {"has_table": True, "source_file": "vax.md"},
        },
    ]


def test_resolve_table_summary_model_precedence(monkeypatch):
    monkeypatch.setenv("VACXIS_GENERATION_MODEL", "unrelated-gen-model")
    monkeypatch.delenv("LIGHTRAG_LLM", raising=False)
    assert resolve_table_summary_model() == DEFAULT_TABLE_SUMMARY_MODEL

    monkeypatch.setenv("LIGHTRAG_LLM", "custom-lightrag-llm")
    assert resolve_table_summary_model() == "custom-lightrag-llm"
    assert resolve_table_summary_model("explicit-model") == "explicit-model"


def test_prepare_records_raw_strategy_preserves_records():
    records = _sample_records()
    prepared, stats = prepare_records_for_indexing(records, strategy="raw")

    assert len(prepared) == 3
    assert stats["total_in"] == 3
    assert stats["total_out"] == 3
    assert stats["tables_total"] == 2
    assert stats["cache_hits"] == 0
    assert stats["llm_calls"] == 0
    assert prepared[1]["text"] == records[1]["text"]
    assert "embed_text" not in prepared[1]


def test_prepare_records_merge_strategy_combines_split_tables():
    records = _sample_records()
    prepared, stats = prepare_records_for_indexing(records, strategy="merge")

    assert len(prepared) == 2
    assert stats["total_in"] == 3
    assert stats["total_out"] == 2
    assert stats["tables_total"] == 1
    table_rec = prepared[1]
    assert table_rec["chunk_id"] == "vax_c_0001"
    assert table_rec["metadata"]["table_strategy"] == "merge"
    assert table_rec["metadata"]["table_kind"] == "clinical"
    assert table_rec["metadata"]["merged_chunk_ids"] == ["vax_c_0001", "vax_c_0002"]
    assert table_rec["text"].count("<tr><td>Trẻ 6 tuần</td><td>0.5 ml</td></tr>") == 1
    assert "<tr><td>Người lớn</td><td>0.5 ml</td></tr>" in table_rec["text"]


def test_prepare_records_summarize_and_summarize_dual_cache_hit(tmp_path: Path):
    records = _sample_records()
    merged_text = merge_html_table_fragments([records[1]["text"], records[2]["text"]])
    table_kind = classify_table_kind(
        header="Liều dùng",
        content=merged_text,
        category="liều lượng và cách dùng",
    )
    cache_key = make_table_cache_key(
        doc_id="vax",
        header="Liều dùng",
        table_kind=table_kind,
        merged_text=merged_text,
        model_name="gpt-oss-120b",
    )
    prose = "Vắc xin Vaxneuvance tiêm liều 0.5 ml cho trẻ từ 6 tuần tuổi và người lớn."
    cache_path = tmp_path / "table_summaries_cache.json"
    save_table_summary_cache(cache_path, {cache_key: prose})

    # summarize strategy
    sum_records, sum_stats = prepare_records_for_indexing(
        records,
        strategy="summarize",
        cache_path=cache_path,
        summary_model="gpt-oss-120b",
    )
    assert sum_stats["cache_hits"] == 1
    assert sum_stats["cache_misses"] == 0
    assert sum_stats["llm_calls"] == 0
    assert sum_records[1]["embed_text"] == f"[CHỦ ĐỀ: Vaxneuvance] > [MỤC: Liều dùng]\n{prose}"
    assert sum_records[1]["text"] == sum_records[1]["embed_text"]
    assert "<table>" not in sum_records[1]["text"]

    # summarize_dual strategy
    dual_records, dual_stats = prepare_records_for_indexing(
        records,
        strategy="summarize_dual",
        cache_path=cache_path,
        summary_model="gpt-oss-120b",
    )
    assert dual_stats["cache_hits"] == 1
    assert dual_stats["cache_misses"] == 0
    assert dual_stats["llm_calls"] == 0
    dual_table = dual_records[1]
    assert dual_table["embed_text"] == f"[CHỦ ĐỀ: Vaxneuvance] > [MỤC: Liều dùng]\n{prose}"
    assert "<table>" not in dual_table["embed_text"]
    assert "[BẢNG DỮ LIỆU GỐC]\n<table>" in dual_table["text"]
    assert prose in dual_table["text"]
    assert "<tr><td>Người lớn</td><td>0.5 ml</td></tr>" in dual_table["text"]
    # Prefix must appear exactly once (no duplication before original table body)
    assert dual_table["text"].count("[CHỦ ĐỀ: Vaxneuvance] > [MỤC: Liều dùng]") == 1
    assert dual_table["metadata"]["table_strategy"] == "summarize_dual"
    assert dual_table["metadata"]["merged_chunk_ids"] == ["vax_c_0001", "vax_c_0002"]


def test_prepare_records_cache_miss_fallback_and_optional_llm(tmp_path: Path):
    records = _sample_records()
    cache_path = tmp_path / "empty_cache.json"

    # Without LLM fallback, cache miss safely falls back to merged raw table text
    fallback_records, fallback_stats = prepare_records_for_indexing(
        records,
        strategy="summarize_dual",
        cache_path=cache_path,
        summary_model="gpt-oss-120b",
        allow_llm_fallback=False,
    )
    assert fallback_stats["cache_misses"] == 1
    assert fallback_stats["llm_calls"] == 0
    assert "<table>" in fallback_records[1]["text"]
    assert fallback_records[1]["embed_text"] == fallback_records[1]["text"]

    # With LLM fallback enabled, generate prose, persist cache, and reuse on second run
    calls = 0

    async def fake_llm(prompt: str, system_prompt: str | None = None) -> str:
        nonlocal calls
        calls += 1
        return "Tóm tắt tự sinh từ LLM."

    gen_records, gen_stats = prepare_records_for_indexing(
        records,
        strategy="summarize_dual",
        cache_path=cache_path,
        summary_model="gpt-oss-120b",
        llm_func=fake_llm,
        allow_llm_fallback=True,
    )
    assert calls == 1
    assert gen_stats["llm_calls"] == 1
    assert "Tóm tắt tự sinh từ LLM." in gen_records[1]["embed_text"]
    assert cache_path.is_file()

    # Second call should hit persisted cache with 0 LLM calls
    cached_records, cached_stats = prepare_records_for_indexing(
        records,
        strategy="summarize_dual",
        cache_path=cache_path,
        summary_model="gpt-oss-120b",
        llm_func=fake_llm,
        allow_llm_fallback=True,
    )
    assert calls == 1
    assert cached_stats["cache_hits"] == 1
    assert cached_stats["llm_calls"] == 0
    assert cached_records[1]["text"] == gen_records[1]["text"]


def test_prepare_records_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="Unsupported table strategy"):
        prepare_records_for_indexing(_sample_records(), strategy="invalid")  # type: ignore[arg-type]


def test_ingest_to_qdrant_summarize_dual_embeds_prose_and_stores_dual_payload(tmp_path: Path):
    records = _sample_records()
    parquet_path = tmp_path / "chunks.parquet"
    write_chunks_to_parquet(records, parquet_path)

    merged_text = merge_html_table_fragments([records[1]["text"], records[2]["text"]])
    table_kind = classify_table_kind("Liều dùng", merged_text, "liều lượng và cách dùng")
    cache_key = make_table_cache_key("vax", "Liều dùng", table_kind, merged_text, "gpt-oss-120b")
    prose = "Vaxneuvance tiêm liều 0.5 ml."
    cache_path = tmp_path / "cache.json"
    save_table_summary_cache(cache_path, {cache_key: prose})

    encoded_batches: list[list[str]] = []

    class RecordingEmbedder:
        def encode(self, texts, **kwargs):
            encoded_batches.append(list(texts))
            return {
                "dense_vecs": np.tile([1.0] + [0.0] * (DENSE_DIM - 1), (len(texts), 1)),
                "lexical_weights": [{"1": 1.0} for _ in texts],
            }

    client = QdrantClient(":memory:")
    try:
        upserted = ingest_to_qdrant(
            parquet_path=parquet_path,
            collection_name="test_table_dual",
            recreate=True,
            table_strategy="summarize_dual",
            table_cache_path=cache_path,
            table_summary_model="gpt-oss-120b",
            client=client,
            embed_model=RecordingEmbedder(),
        )
        assert upserted == 2
        assert len(encoded_batches) == 1
        # Embedded text for table record must be clean prose without HTML table tags
        assert "<table>" not in encoded_batches[0][1]
        assert prose in encoded_batches[0][1]

        points, _ = client.scroll(collection_name="test_table_dual", limit=10, with_payload=True)
        by_id = {p.payload["chunk_id"]: p.payload for p in points}
        assert set(by_id.keys()) == {"vax_c_0000", "vax_c_0001"}
        table_payload = by_id["vax_c_0001"]
        assert prose in table_payload["text"]
        assert "[BẢNG DỮ LIỆU GỐC]" in table_payload["text"]
        assert "<tr><td>Người lớn</td><td>0.5 ml</td></tr>" in table_payload["text"]
        assert table_payload["metadata"]["table_strategy"] == "summarize_dual"
        assert table_payload["metadata"]["table_kind"] == "clinical"
        assert table_payload["metadata"]["merged_chunk_ids"] == ["vax_c_0001", "vax_c_0002"]
    finally:
        client.close()
