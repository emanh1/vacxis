"""Unit tests for the retriever factory and pipeline wiring."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vacxis_capstone_fa26.generation.generator import MedicalGenerator
from vacxis_capstone_fa26.generation.pipeline import VacxisRAGPipeline
from vacxis_capstone_fa26.generation.schemas import GenerationConfig
from vacxis_capstone_fa26.retrieval import build_retriever
from vacxis_capstone_fa26.retrieval.bm25 import BM25Retriever
from vacxis_capstone_fa26.retrieval.dense import DenseRetriever
from vacxis_capstone_fa26.retrieval.hybrid_lightrag import HybridLightRAGRetriever
from vacxis_capstone_fa26.retrieval.lightrag_retriever import LightRAGRetriever
from vacxis_capstone_fa26.retrieval.retriever import VacxisRetriever
from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig


def test_build_retriever_dispatches_supported_modes():
    bm25 = build_retriever(
        RetrievalConfig(mode="naive"),
        chunk_texts={"c1": "Vắc xin phòng bệnh viêm gan B."},
    )
    assert isinstance(bm25, BM25Retriever)

    dense = build_retriever(RetrievalConfig(mode="advanced", collection_name="col_a"))
    assert isinstance(dense, DenseRetriever)
    assert dense.config.collection_name == "col_a"

    hybrid = build_retriever(
        RetrievalConfig(mode="hybrid", merge_table_chunks=True, collection_name="col_b")
    )
    assert isinstance(hybrid, VacxisRetriever)
    assert hybrid.config.merge_table_chunks is True
    assert hybrid.config.collection_name == "col_b"

    lightrag = build_retriever(
        RetrievalConfig(mode="lightrag", lightrag_mode="local", lightrag_workspace="ws_a")
    )
    assert isinstance(lightrag, LightRAGRetriever)
    assert lightrag.config.lightrag_mode == "local"
    assert lightrag.config.lightrag_workspace == "ws_a"

    hybrid_lr = build_retriever(
        RetrievalConfig(
            mode="hybrid_lightrag",
            lightrag_mode="hybrid",
            lightrag_fusion="rrf_rerank",
            collection_name="col_dual",
        )
    )
    assert isinstance(hybrid_lr, HybridLightRAGRetriever)
    assert hybrid_lr.config.lightrag_fusion == "rrf_rerank"
    assert hybrid_lr.config.collection_name == "col_dual"


def test_build_retriever_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported retriever mode"):
        build_retriever(RetrievalConfig(mode="unknown"))  # type: ignore[arg-type]


def test_pipeline_uses_build_retriever_for_configured_mode():
    pipeline = VacxisRAGPipeline(
        retrieval_config=RetrievalConfig(mode="advanced", top_k=3),
        generator=MedicalGenerator(GenerationConfig(dry_run=True)),
    )
    assert isinstance(pipeline.retriever, DenseRetriever)
    assert pipeline.retriever.config.top_k == 3


def test_bm25_and_dense_protocol_compatibility():
    bm25 = BM25Retriever({"c1": "Liều tiêm Verorab 0.5 ml", "c2": "Bảo quản ở 2 đến 8 độ C"})
    with bm25 as active:
        single = active.retrieve("Verorab", top_k=1, skip_rerank=True)
        batch = active.batch_retrieve(["Verorab", "bảo quản"], top_k=1, skip_rerank=True)
    assert [c["chunk_id"] for c in single] == ["c1"]
    assert [[c["chunk_id"] for c in group] for group in batch] == [["c1"], ["c2"]]

    model, client = Mock(), Mock()
    model.encode.return_value = {"dense_vecs": [SimpleNamespace(tolist=lambda: [0.1, 0.9])]}
    client.query_points.return_value = SimpleNamespace(
        points=[SimpleNamespace(id=1, score=0.9, payload={"chunk_id": "c1", "text": "Text 1"})]
    )
    dense = DenseRetriever(RetrievalConfig(top_k=1), embed_model=model, client=client)
    batch_dense = dense.batch_retrieve(["q1", "q2"], top_k=1)
    assert len(batch_dense) == 2
    assert batch_dense[0][0]["chunk_id"] == "c1"
