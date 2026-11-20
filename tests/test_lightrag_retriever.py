"""Unit tests for LightRAG embedder, retriever, and HybridLightRAGRetriever."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vacxis_capstone_fa26.retrieval.hybrid_lightrag import (
    HybridLightRAGRetriever,
    _fuse_hybrid_and_lightrag_candidates,
)
from vacxis_capstone_fa26.retrieval.lightrag_embedder import (
    DENSE_DIM,
    LocalBGEM3Embedder,
    is_local_bge_m3_model,
)
from vacxis_capstone_fa26.retrieval.lightrag_retriever import (
    KG_CONTENT_TYPE,
    KG_CONTEXT_CHUNK_ID,
    LightRAGRetriever,
    build_lightrag_chunk_id_map,
    format_kg_context,
)
from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig, RetrievedChunk


def _chunk(
    chunk_id: str,
    text: str = "Sample text",
    *,
    score: float = 0.5,
    rrf_score: float = 0.02,
    metadata: dict[str, Any] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc_1",
        doc_title="Vaccine A",
        category="INDICATION",
        parent_header="CHỈ ĐỊNH",
        header="Liều dùng",
        content_type="text",
        text=text,
        score=score,
        rrf_score=rrf_score,
        rerank_score=None,
        metadata=metadata or {},
    )


class _FakeEmbedModel:
    def __init__(self, dim: int = DENSE_DIM) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        self.calls.append(list(texts))
        vecs = np.ones((len(texts), self.dim), dtype=np.float32)
        return {"dense_vecs": vecs}


class _FakeLightRAG:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[tuple[str, Any]] = []
        self.finalized = False

    async def aquery_data(self, query: str, param: Any) -> dict[str, Any]:
        self.calls.append((query, param))
        return self.response

    async def finalize_storages(self) -> None:
        self.finalized = True


def test_local_bge_m3_embedder_reuses_shared_model_and_validates_shape():
    assert is_local_bge_m3_model("BAAI/bge-m3") is True
    assert is_local_bge_m3_model("models/bge-m3") is True
    assert is_local_bge_m3_model("openai/text-embedding-3-small") is False

    fake = _FakeEmbedModel()
    embedder = LocalBGEM3Embedder(embed_model=fake, batch_size=8, max_length=256)
    assert embedder.embed_model is fake

    import copy
    from dataclasses import asdict, dataclass

    @dataclass
    class _Holder:
        func: Any

    assert copy.deepcopy(embedder) is embedder
    assert asdict(_Holder(func=embedder))["func"] is embedder


    empty = embedder.encode_sync([])
    assert empty.shape == (0, DENSE_DIM)
    assert fake.calls == []

    vecs = asyncio.run(embedder(["Liều tiêm Verorab", "Phản ứng sau tiêm"]))
    assert vecs.shape == (2, DENSE_DIM)
    assert vecs.dtype == np.float32
    assert len(fake.calls) == 1

    bad_embedder = LocalBGEM3Embedder(embed_model=_FakeEmbedModel(dim=512))
    with pytest.raises(ValueError, match="Expected embedding shape"):
        bad_embedder.encode_sync(["test"])

    with pytest.raises(ValueError, match="positive integers"):
        LocalBGEM3Embedder(batch_size=0)


def test_build_lightrag_chunk_id_map_matches_merged_table_order(tmp_path: Path):
    parquet_path = tmp_path / "chunks.parquet"
    rows = [
        {
            "chunk_id": "vax_a_c_0000",
            "doc_id": "vax_a",
            "doc_title": "Vaccine A",
            "category": "INDICATION",
            "parent_header": "CHỈ ĐỊNH",
            "header": "Chỉ định",
            "content_type": "text",
            "text": "Văn bản mở đầu.",
            "metadata": {"has_table": False},
        },
        {
            "chunk_id": "vax_a_c_0001",
            "doc_id": "vax_a",
            "doc_title": "Vaccine A",
            "category": "DOSAGE",
            "parent_header": "LIỀU DÙNG",
            "header": "Bảng 1",
            "content_type": "table",
            "text": "<table><tr><th>Nhóm</th><th>Liều</th></tr><tr><td>Trẻ em</td><td>0.5ml</td></tr></table>",
            "metadata": {"has_table": True},
        },
        {
            "chunk_id": "vax_a_c_0002",
            "doc_id": "vax_a",
            "doc_title": "Vaccine A",
            "category": "DOSAGE",
            "parent_header": "LIỀU DÙNG",
            "header": "Bảng 1",
            "content_type": "table",
            "text": "<table><tr><th>Nhóm</th><th>Liều</th></tr><tr><td>Người lớn</td><td>1.0ml</td></tr></table>",
            "metadata": {"has_table": True},
        },
        {
            "chunk_id": "vax_a_c_0003",
            "doc_id": "vax_a",
            "doc_title": "Vaccine A",
            "category": "STORAGE",
            "parent_header": "BẢO QUẢN",
            "header": "Nhiệt độ",
            "content_type": "text",
            "text": "Bảo quản từ 2 đến 8 độ C.",
            "metadata": {"has_table": False},
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    # After group_and_merge_document_chunks:
    # index 0 -> vax_a_c_0000
    # index 1 -> merged(vax_a_c_0001 + vax_a_c_0002), primary chunk_id = vax_a_c_0001
    # index 2 -> vax_a_c_0003
    kv_path = tmp_path / "kv_store_text_chunks.json"
    kv_path.write_text(
        json.dumps(
            {
                "chunk-hash-0": {"full_doc_id": "vax_a", "chunk_order_index": 0, "content": "..."},
                "chunk-hash-1": {"full_doc_id": "vax_a", "chunk_order_index": 1, "content": "..."},
                "chunk-hash-2": {"full_doc_id": "vax_a", "chunk_order_index": 2, "content": "..."},
            }
        ),
        encoding="utf-8",
    )

    mapping = build_lightrag_chunk_id_map(kv_path, parquet_path)
    assert mapping["chunk-hash-0"]["chunk_id"] == "vax_a_c_0000"
    assert mapping["chunk-hash-1"]["chunk_id"] == "vax_a_c_0001"
    assert mapping["chunk-hash-1"]["metadata"]["merged_chunk_ids"] == [
        "vax_a_c_0001",
        "vax_a_c_0002",
    ]
    assert mapping["chunk-hash-2"]["chunk_id"] == "vax_a_c_0003"
    assert mapping["chunk-hash-2"]["category"] == "STORAGE"

    # Missing file returns empty map gracefully
    assert build_lightrag_chunk_id_map(tmp_path / "missing.json", parquet_path) == {}


def test_format_kg_context_normalizes_separators_and_handles_empty():
    assert format_kg_context([], []) == ""
    assert format_kg_context([{"entity_name": "   "}], [{"src_id": "A", "tgt_id": ""}]) == ""

    formatted = format_kg_context(
        entities=[
            {
                "entity_name": "Gardasil 9",
                "entity_type": "VACCINE",
                "description": "Phòng 9 chủng HPV<SEP>Tiêm bắp 0.5 ml",
            }
        ],
        relationships=[
            {
                "src_id": "Gardasil 9",
                "tgt_id": "Ung thư cổ tử cung",
                "keywords": "phòng ngừa<SEP>chỉ định",
                "description": "Giúp phòng ngừa ung thư cổ tử cung do HPV.",
            }
        ],
    )
    assert "[ĐỒ THỊ TRI THỨC Y KHOA (LIGHTRAG)]" in formatted
    assert "- Gardasil 9 (VACCINE): Phòng 9 chủng HPV; Tiêm bắp 0.5 ml" in formatted
    assert "- Gardasil 9 -> Ung thư cổ tử cung [phòng ngừa; chỉ định]: Giúp phòng ngừa ung thư cổ tử cung do HPV." in formatted


def test_lightrag_retriever_maps_chunks_and_appends_kg_context():
    fake_rag = _FakeLightRAG(
        response={
            "status": "success",
            "data": {
                "entities": [
                    {
                        "entity_name": "Verorab",
                        "entity_type": "VACCINE",
                        "description": "Vắc-xin phòng dại.",
                    }
                ],
                "relationships": [
                    {
                        "src_id": "Verorab",
                        "tgt_id": "Bệnh dại",
                        "keywords": "phòng bệnh",
                        "description": "Verorab tạo miễn dịch phòng bệnh dại.",
                    }
                ],
                "chunks": [
                    {"chunk_id": "chunk-abc", "content": "Nội dung chunk Verorab."},
                    {"chunk_id": "chunk-unmapped", "content": "Nội dung chưa ánh xạ."},
                ],
            },
            "metadata": {"query_mode": "hybrid", "keywords": {"high_level": ["dại"], "low_level": ["Verorab"]}},
        }
    )
    chunk_map = {
        "chunk-abc": {
            "chunk_id": "verorab_001_c_0001",
            "doc_id": "verorab_001",
            "doc_title": "Verorab",
            "category": "INDICATION",
            "parent_header": "CHỈ ĐỊNH",
            "header": "Phòng dại",
            "content_type": "text",
            "metadata": {"has_table": False},
        }
    }
    config = RetrievalConfig(mode="lightrag", lightrag_mode="hybrid", lightrag_chunk_top_k=5)
    with LightRAGRetriever(config=config, rag=fake_rag, chunk_id_map=chunk_map) as retriever:
        assert retriever.retrieve("   ") == []
        results = retriever.retrieve("Liều tiêm Verorab?", top_k=2)
        batch = retriever.batch_retrieve(["Câu hỏi 1", "Câu hỏi 2"], top_k=1)

    assert len(fake_rag.calls) == 3
    first_query, first_param = fake_rag.calls[0]
    assert first_query == "Liều tiêm Verorab?"
    assert first_param.enable_rerank is False
    assert first_param.only_need_context is True
    assert first_param.mode == "hybrid"

    assert [c.chunk_id for c in results] == [
        "verorab_001_c_0001",
        "chunk-unmapped",
        KG_CONTEXT_CHUNK_ID,
    ]
    assert results[0].doc_title == "Verorab"
    assert results[0].metadata["lightrag_chunk_id"] == "chunk-abc"
    assert results[2].content_type == KG_CONTENT_TYPE
    assert "[ĐỒ THỊ TRI THỨC Y KHOA (LIGHTRAG)]" in results[2].text
    assert len(batch) == 2
    assert [c.chunk_id for c in batch[0]] == ["verorab_001_c_0001", KG_CONTEXT_CHUNK_ID]
    assert fake_rag.finalized is False  # Injected rag is not owned by retriever


def test_hybrid_lightrag_context_append_mode():
    hybrid = Mock()
    hybrid.retrieve.return_value = [_chunk("c1", "Dual table text 1"), _chunk("c2", "Text 2")]
    hybrid.batch_retrieve.return_value = [[_chunk("c1")], [_chunk("c2")]]

    kg_chunk = _chunk(KG_CONTEXT_CHUNK_ID, "[ĐỒ THỊ TRI THỨC Y KHOA (LIGHTRAG)]\n- A: B")
    lightrag = Mock()
    lightrag.retrieve_components.side_effect = [([], kg_chunk), ([], kg_chunk), ([], None)]

    config = RetrievalConfig(mode="hybrid_lightrag", lightrag_fusion="context_append", top_k=2)
    with HybridLightRAGRetriever(config=config, hybrid_retriever=hybrid, lightrag_retriever=lightrag) as ret:
        single = ret.retrieve("Câu hỏi 1", top_k=2)
        batch = ret.batch_retrieve(["Q1", "Q2"], top_k=1)

    assert [c.chunk_id for c in single] == ["c1", "c2", KG_CONTEXT_CHUNK_ID]
    assert [c.chunk_id for c in batch[0]] == ["c1", KG_CONTEXT_CHUNK_ID]
    assert [c.chunk_id for c in batch[1]] == ["c2"]
    hybrid.close.assert_called_once()
    lightrag.close.assert_called_once()


def test_hybrid_lightrag_rrf_rerank_dedupes_merged_tables_and_reranks():
    h_table = _chunk(
        "doc_c_0001",
        "Prose\n\n[BẢNG DỮ LIỆU GỐC]\n<table>...</table>",
        metadata={"has_table": True, "merged_chunk_ids": ["doc_c_0001", "doc_c_0002"]},
    )
    h_other = _chunk("doc_c_0005", "Hybrid only chunk")
    lr_subchunk = _chunk("doc_c_0002", "Prose only from LightRAG")
    lr_new = _chunk("doc_c_0009", "Graph-linked chunk")

    fused = _fuse_hybrid_and_lightrag_candidates(
        [h_table, h_other],
        [lr_subchunk, lr_new],
        rrf_k=60,
    )
    assert [c.chunk_id for c in fused] == ["doc_c_0001", "doc_c_0005", "doc_c_0009"]
    assert "[BẢNG DỮ LIỆU GỐC]" in fused[0].text
    assert fused[0].metadata["lightrag_fused"] is True

    hybrid = Mock()
    hybrid.retrieve.return_value = [h_table, h_other]
    hybrid.reranker.rerank.side_effect = lambda query, candidates, top_k: [
        replace(c, score=0.95 - idx * 0.1, rerank_score=0.95 - idx * 0.1)
        for idx, c in enumerate(candidates[:top_k])
    ]
    kg_chunk = _chunk(KG_CONTEXT_CHUNK_ID, "[ĐỒ THỊ TRI THỨC Y KHOA (LIGHTRAG)]\n- Vax")
    lightrag = Mock()
    lightrag.retrieve_components.return_value = ([lr_subchunk, lr_new], kg_chunk)

    rerank_called = False

    def _on_rerank() -> None:
        nonlocal rerank_called
        rerank_called = True

    config = RetrievalConfig(mode="hybrid_lightrag", lightrag_fusion="rrf_rerank", top_k=2)
    ret = HybridLightRAGRetriever(config=config, hybrid_retriever=hybrid, lightrag_retriever=lightrag)
    results = ret.retrieve("Liều dùng?", top_k=2, on_rerank=_on_rerank)

    assert rerank_called is True
    assert [c.chunk_id for c in results] == ["doc_c_0001", "doc_c_0005", KG_CONTEXT_CHUNK_ID]
    assert hybrid.retrieve.call_args.kwargs["skip_rerank"] is True


def test_hybrid_lightrag_shares_single_embed_model_and_closes_both_on_error():
    shared_model = _FakeEmbedModel()
    config = RetrievalConfig(mode="hybrid_lightrag")
    ret = HybridLightRAGRetriever(config=config, embed_model=shared_model, lazy_load=False)

    assert ret.hybrid_retriever._embed_model is shared_model
    assert ret.lightrag_retriever._embed_model is shared_model

    # Even if hybrid.close() fails, lightrag.close() must still execute in finally
    bad_hybrid = Mock()
    bad_hybrid.close.side_effect = RuntimeError("hybrid close failed")
    good_lightrag = Mock()
    ret2 = HybridLightRAGRetriever(
        config=config,
        hybrid_retriever=bad_hybrid,
        lightrag_retriever=good_lightrag,
    )
    with pytest.raises(RuntimeError, match="hybrid close failed"):
        ret2.close()
    good_lightrag.close.assert_called_once()


def test_hybrid_lightrag_rejects_invalid_fusion():
    with pytest.raises(ValueError, match="Unsupported lightrag_fusion"):
        HybridLightRAGRetriever(
            config=RetrievalConfig(mode="hybrid_lightrag", lightrag_fusion="bad_mode")  # type: ignore[arg-type]
        )
