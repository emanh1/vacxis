"""Unit tests for table chunk merging, classification, LLM prose conversion, and retrieval merge."""

import asyncio
from pathlib import Path

from vacxis_capstone_fa26.indexing.table_summarizer import (
    classify_table_kind,
    group_and_merge_document_chunks,
    load_table_summary_cache,
    merge_html_table_fragments,
    merge_markdown_table_fragments,
    save_table_summary_cache,
    summarize_table_to_entity_prose,
)
from vacxis_capstone_fa26.retrieval.retriever import merge_retrieved_table_chunks
from vacxis_capstone_fa26.retrieval.schemas import RetrievedChunk


def test_merge_html_table_fragments_deduplicates_thead_and_overlap_rows():
    part1 = (
        "[CHỦ ĐỀ: Vaxneuvance] > [MỤC: Liều dùng]\n"
        "<table>\n<thead>\n<tr><th>Đối tượng</th><th>Liều</th></tr>\n</thead>\n"
        "<tbody>\n<tr><td>Trẻ 6 tuần</td><td>0.5 ml</td></tr>\n"
        "<tr><td>Trẻ 12 tháng</td><td>0.5 ml</td></tr>\n</tbody>\n</table>"
    )
    part2 = (
        "[CHỦ ĐỀ: Vaxneuvance] > [MỤC: Liều dùng]\n"
        "<table>\n<thead>\n<tr><th>Đối tượng</th><th>Liều</th></tr>\n</thead>\n"
        "<tbody>\n<tr><td>Trẻ 12 tháng</td><td>0.5 ml</td></tr>\n"
        "<tr><td>Người lớn</td><td>0.5 ml</td></tr>\n</tbody>\n"
        "<tfoot><tr><td colspan='2'>Tổng kết</td></tr></tfoot>\n</table>\n"
        "<p><sup>a</sup> Tiêm bắp.</p>"
    )

    merged = merge_html_table_fragments([part1, part2])

    assert merged.startswith("[CHỦ ĐỀ: Vaxneuvance] > [MỤC: Liều dùng]\n<table>")
    assert merged.count("<thead>") == 1
    assert merged.count("<tr><td>Trẻ 12 tháng</td><td>0.5 ml</td></tr>") == 1
    assert "<tr><td>Trẻ 6 tuần</td><td>0.5 ml</td></tr>" in merged
    assert "<tr><td>Người lớn</td><td>0.5 ml</td></tr>" in merged
    assert "<tfoot><tr><td colspan='2'>Tổng kết</td></tr></tfoot>" in merged
    assert "<p><sup>a</sup> Tiêm bắp.</p>" in merged


def test_merge_markdown_table_fragments_deduplicates_header_and_overlap():
    part1 = (
        "[CHỦ ĐỀ: Prevenar 20] > [MỤC: Thành phần]\n"
        "| Kháng nguyên | Hàm lượng |\n| --- | --- |\n"
        "| Týp 1 | 2.2 mcg |\n| Týp 3 | 2.2 mcg |"
    )
    part2 = (
        "[CHỦ ĐỀ: Prevenar 20] > [MỤC: Thành phần]\n"
        "| Kháng nguyên | Hàm lượng |\n| --- | --- |\n"
        "| Týp 3 | 2.2 mcg |\n| Týp 4 | 2.2 mcg |"
    )

    merged = merge_markdown_table_fragments([part1, part2])

    assert merged.count("| Kháng nguyên | Hàm lượng |") == 1
    assert merged.count("| Týp 3 | 2.2 mcg |") == 1
    assert "| Týp 1 | 2.2 mcg |" in merged
    assert "| Týp 4 | 2.2 mcg |" in merged


def test_group_and_merge_document_chunks_respects_boundaries():
    records = [
        {"chunk_id": "d_c_0000", "doc_id": "d", "header": "A", "text": "Text A", "metadata": {"has_table": False}},
        {
            "chunk_id": "d_c_0001",
            "doc_id": "d",
            "header": "Bảng 1",
            "text": "<table><thead><tr><th>C</th></tr></thead><tbody><tr><td>1</td></tr></tbody></table>",
            "metadata": {"has_table": True},
        },
        {
            "chunk_id": "d_c_0002",
            "doc_id": "d",
            "header": "Bảng 1",
            "text": "<table><thead><tr><th>C</th></tr></thead><tbody><tr><td>1</td></tr><tr><td>2</td></tr></tbody></table>",
            "metadata": {"has_table": True},
        },
        {
            "chunk_id": "d_c_0003",
            "doc_id": "d",
            "header": "Bảng 2",
            "text": "<table><tbody><tr><td>3</td></tr></tbody></table>",
            "metadata": {"has_table": True},
        },
    ]

    merged = group_and_merge_document_chunks(records)

    assert len(merged) == 3
    assert merged[0]["chunk_id"] == "d_c_0000"
    assert merged[1]["merged_chunk_ids"] == ["d_c_0001", "d_c_0002"]
    assert "<tr><td>2</td></tr>" in merged[1]["text"]
    assert merged[1]["text"].count("<tr><td>1</td></tr>") == 1
    assert merged[2]["chunk_id"] == "d_c_0003"


def test_classify_table_kind_distinguishes_clinical_and_statistical():
    assert classify_table_kind(
        "Bảng 1: Báo cáo các phản ứng bất lợi trong dự kiến trong vòng 7 ngày",
        "<table><tr><td>Sốt</td><td>12,3%</td><td>4,5%</td></tr></table>",
        category="tác dụng không mong muốn",
    ) == "clinical"

    assert classify_table_kind(
        "Thông tin chung",
        "<table><tr><td>Thành phần hoạt chất:</td><td>Virus thủy đậu</td><td>3.800 PFU</td></tr></table>",
        category="thông tin chung",
    ) == "clinical"

    assert classify_table_kind(
        "Bảng 6: GMT OPA đặc hiệu týp huyết thanh lúc 30 ngày sau tiêm (Đề cương 019)",
        "<table><tr><td>1</td><td>598</td><td>256,3</td><td>322,6</td><td>0,79 (0,66, 0,96)</td></tr></table>",
        category="bảng số liệu",
    ) == "statistical"

    assert classify_table_kind(
        "Bảng 1. Đáp ứng kháng thể diệt khuẩn trong huyết thanh sau liều thứ ba Bexsero",
        "<table><tr><td>fHbp</td><td>99%</td><td>100%</td></tr></table>",
        category="quá liều",
    ) == "statistical"


def test_summarize_table_to_entity_prose_uses_cache(tmp_path: Path):
    calls = 0

    async def fake_llm(prompt: str, system_prompt: str | None = None) -> str:
        nonlocal calls
        calls += 1
        assert "Vaxneuvance" in prompt
        return "Vắc xin Vaxneuvance được tiêm bắp liều 0.5 ml cho trẻ từ 6 tuần tuổi."

    record = {
        "doc_id": "539310049125",
        "doc_title": "Vaxneuvance",
        "header": "Liều dùng",
        "category": "liều lượng và cách dùng",
        "text": "[CHỦ ĐỀ: Vaxneuvance] > [MỤC: Liều dùng]\n<table><tr><td>Trẻ em</td><td>0.5 ml</td></tr></table>",
    }
    cache_path = tmp_path / "cache.json"
    cache: dict[str, str] = {}

    out1 = asyncio.run(summarize_table_to_entity_prose(record, fake_llm, cache, "test-model"))
    save_table_summary_cache(cache_path, cache)
    reloaded = load_table_summary_cache(cache_path)
    out2 = asyncio.run(summarize_table_to_entity_prose(record, fake_llm, reloaded, "test-model"))

    assert calls == 1
    assert out1 == out2
    assert out1.startswith("[CHỦ ĐỀ: Vaxneuvance] > [MỤC: Liều dùng]\n")
    assert "<table>" not in out1


def test_merge_retrieved_table_chunks_combines_same_table_fragments():
    c1 = RetrievedChunk(
        chunk_id="doc_c_0002",
        doc_id="doc",
        doc_title="Vaxneuvance",
        category="liều lượng và cách dùng",
        parent_header="",
        header="Liều dùng",
        content_type="table",
        text="[CHỦ ĐỀ: Vaxneuvance] > [MỤC: Liều dùng]\n<table><thead><tr><th>A</th></tr></thead><tbody><tr><td>1</td></tr></tbody></table>",
        score=0.9,
        rrf_score=0.8,
        metadata={"has_table": True},
    )
    c_text = RetrievedChunk(
        chunk_id="doc_c_0001",
        doc_id="doc",
        doc_title="Vaxneuvance",
        category="chỉ định",
        parent_header="",
        header="Chỉ định",
        content_type="text",
        text="Chỉ định phòng bệnh.",
        score=0.85,
        rrf_score=0.75,
        metadata={"has_table": False},
    )
    c2 = RetrievedChunk(
        chunk_id="doc_c_0003",
        doc_id="doc",
        doc_title="Vaxneuvance",
        category="liều lượng và cách dùng",
        parent_header="",
        header="Liều dùng",
        content_type="table",
        text="[CHỦ ĐỀ: Vaxneuvance] > [MỤC: Liều dùng]\n<table><thead><tr><th>A</th></tr></thead><tbody><tr><td>1</td></tr><tr><td>2</td></tr></tbody></table>",
        score=0.8,
        rrf_score=0.7,
        metadata={"has_table": True},
    )

    merged = merge_retrieved_table_chunks([c1, c_text, c2])

    assert len(merged) == 2
    assert merged[0].chunk_id == "doc_c_0002"
    assert merged[0].text.count("<tr><td>1</td></tr>") == 1
    assert "<tr><td>2</td></tr>" in merged[0].text
    assert merged[1].chunk_id == "doc_c_0001"


def test_ingested_checkpoint_roundtrip(tmp_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lightrag_ingestion",
        Path(__file__).resolve().parent.parent / "scripts" / "lightrag_ingestion.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ckpt = tmp_path / "ingested_docs_vacxis.json"
    assert mod.load_ingested_checkpoint(ckpt) == set()

    mod.save_ingested_checkpoint(ckpt, {"doc_b", "doc_a"})
    assert mod.load_ingested_checkpoint(ckpt) == {"doc_a", "doc_b"}
