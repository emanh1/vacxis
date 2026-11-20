"""Unit tests for scripts/reembed_lightrag_qdrant.py and local LightRAG BGE-M3 wiring."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from qdrant_client import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_script_module(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name,
        PROJECT_ROOT / "scripts" / filename,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_ensure_lightrag_target_collection_creates_schema_and_tenant_index():
    """Target collection should use COSINE 1024-d vectors, HNSW payload_m=16, and workspace_id index."""
    mod = _load_script_module("reembed_lightrag_qdrant", "reembed_lightrag_qdrant.py")
    client = MagicMock()
    client.collection_exists.return_value = False

    created = mod.ensure_lightrag_target_collection(
        client,
        "lightrag_vdb_chunks_baai_bge_m3_1024d",
    )

    assert created is True
    client.delete_collection.assert_not_called()
    client.create_collection.assert_called_once()
    create_kwargs = client.create_collection.call_args.kwargs
    assert create_kwargs["collection_name"] == "lightrag_vdb_chunks_baai_bge_m3_1024d"
    assert create_kwargs["vectors_config"] == models.VectorParams(
        size=1024,
        distance=models.Distance.COSINE,
    )
    assert create_kwargs["hnsw_config"] == models.HnswConfigDiff(payload_m=16, m=0)

    client.create_payload_index.assert_called_once_with(
        collection_name="lightrag_vdb_chunks_baai_bge_m3_1024d",
        field_name="workspace_id",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            is_tenant=True,
        ),
    )


def test_ensure_lightrag_target_collection_skips_or_recreates():
    """Existing target collections skip unless recreate=True."""
    mod = _load_script_module("reembed_lightrag_qdrant", "reembed_lightrag_qdrant.py")
    client = MagicMock()
    client.collection_exists.return_value = True

    assert mod.ensure_lightrag_target_collection(client, "target_col", recreate=False) is False
    client.delete_collection.assert_not_called()
    client.create_collection.assert_not_called()

    assert mod.ensure_lightrag_target_collection(client, "target_col", recreate=True) is True
    client.delete_collection.assert_called_once_with(collection_name="target_col")
    client.create_collection.assert_called_once()


def test_reembed_collection_preserves_ids_and_payloads_across_batches():
    """Scroll pagination should preserve IDs/payloads and pass max_length=8192 dense-only."""
    mod = _load_script_module("reembed_lightrag_qdrant", "reembed_lightrag_qdrant.py")

    p1 = SimpleNamespace(id="id-1", payload={"workspace_id": "vacxis", "content": "Vắc xin Adacel"})
    p2 = SimpleNamespace(id="id-2", payload={"workspace_id": "vacxis", "content": "Vắc xin Verorab"})
    p3 = SimpleNamespace(id="id-3", payload={"workspace_id": "vacxis", "content": "Liều tiêm nhắc lại"})

    client = MagicMock()
    client.collection_exists.side_effect = lambda name: name == "src_col"
    client.count.return_value = SimpleNamespace(count=3)
    client.scroll.side_effect = [
        ([p1, p2], "cursor-1"),
        ([p3], None),
    ]

    fake_model = MagicMock()
    fake_model.encode.side_effect = [
        {"dense_vecs": np.ones((2, 1024), dtype=np.float32) * 0.25},
        {"dense_vecs": np.ones((1, 1024), dtype=np.float32) * 0.5},
    ]
    embedder = mod.LocalBGEM3Embedder(
        model_path="models/bge-m3",
        batch_size=2,
        max_length=8192,
        model=fake_model,
    )

    summary = mod.reembed_collection(
        client=client,
        embedder=embedder,
        source_collection="src_col",
        target_collection="dst_col",
        scroll_batch_size=2,
        upsert_batch_size=2,
    )

    assert summary == {
        "source": "src_col",
        "target": "dst_col",
        "status": "migrated",
        "source_count": 3,
        "target_count": 3,
        "migrated": 3,
    }
    assert fake_model.encode.call_count == 2
    first_call_kwargs = fake_model.encode.call_args_list[0].kwargs
    assert first_call_kwargs["max_length"] == 8192
    assert first_call_kwargs["return_dense"] is True
    assert first_call_kwargs["return_sparse"] is False
    assert first_call_kwargs["return_colbert_vecs"] is False

    upserted_points = []
    for call in client.upsert.call_args_list:
        assert call.kwargs["collection_name"] == "dst_col"
        assert call.kwargs["wait"] is True
        upserted_points.extend(call.kwargs["points"])

    assert [pt.id for pt in upserted_points] == ["id-1", "id-2", "id-3"]
    assert [pt.payload for pt in upserted_points] == [p1.payload, p2.payload, p3.payload]
    assert pytest.approx(upserted_points[0].vector[0]) == 0.25
    assert pytest.approx(upserted_points[2].vector[0]) == 0.5


def test_reembed_collection_rejects_missing_content():
    """Points without nonblank payload['content'] must raise ValueError."""
    mod = _load_script_module("reembed_lightrag_qdrant", "reembed_lightrag_qdrant.py")
    bad_point = SimpleNamespace(id="bad-1", payload={"workspace_id": "vacxis", "content": "  "})

    client = MagicMock()
    client.collection_exists.side_effect = lambda name: name == "src_col"
    client.count.return_value = SimpleNamespace(count=1)
    client.scroll.return_value = ([bad_point], None)

    embedder = mod.LocalBGEM3Embedder(model=MagicMock())

    with pytest.raises(ValueError, match="bad-1"):
        mod.reembed_collection(
            client=client,
            embedder=embedder,
            source_collection="src_col",
            target_collection="dst_col",
        )


@pytest.mark.parametrize(
    "bad_shape",
    [(1, 1024), (2, 768)],
)
def test_local_bge_m3_embedder_rejects_shape_mismatch(bad_shape: tuple[int, int]):
    """encode_sync must raise ValueError on vector count or dimension mismatch."""
    mod = _load_script_module("reembed_lightrag_qdrant", "reembed_lightrag_qdrant.py")
    fake_model = MagicMock()
    fake_model.encode.return_value = {"dense_vecs": np.zeros(bad_shape, dtype=np.float32)}
    embedder = mod.LocalBGEM3Embedder(model=fake_model)

    with pytest.raises(ValueError, match="Expected embedding shape"):
        embedder.encode_sync(["text-1", "text-2"])


def test_reembed_collection_raises_on_count_mismatch():
    """reembed_collection must raise RuntimeError if target point count differs from source."""
    mod = _load_script_module("reembed_lightrag_qdrant", "reembed_lightrag_qdrant.py")
    p1 = SimpleNamespace(id="id-1", payload={"workspace_id": "vacxis", "content": "Nội dung 1"})

    client = MagicMock()
    client.collection_exists.side_effect = lambda name: name == "src_col"
    client.count.side_effect = [
        SimpleNamespace(count=1),
        SimpleNamespace(count=0),
    ]
    client.scroll.return_value = ([p1], None)

    fake_model = MagicMock()
    fake_model.encode.return_value = {"dense_vecs": np.ones((1, 1024), dtype=np.float32)}
    embedder = mod.LocalBGEM3Embedder(model=fake_model)

    with pytest.raises(RuntimeError, match="Point count mismatch"):
        mod.reembed_collection(
            client=client,
            embedder=embedder,
            source_collection="src_col",
            target_collection="dst_col",
        )


def test_local_bge_m3_embedder_async_and_ingestion_config():
    """LightRAG ingestion helper selects local BGE-M3 and canonical baai/bge-m3 suffix."""
    ingestion = _load_script_module("lightrag_ingestion", "lightrag_ingestion.py")
    reembed = _load_script_module("reembed_lightrag_qdrant", "reembed_lightrag_qdrant.py")

    assert ingestion.is_local_bge_m3_model("baai/bge-m3") is True
    assert ingestion.is_local_bge_m3_model("BAAI/bge-m3") is True
    assert ingestion.is_local_bge_m3_model("models/bge-m3") is True
    assert ingestion.is_local_bge_m3_model("Vietnamese_Embedding") is False

    canonical_name, embed_fn = ingestion.build_embedding_config(
        "BAAI/bge-m3",
        embed_model_path="models/bge-m3",
        base_url="https://example.com/v1",
        api_key="unused",
        embed_dim=1024,
    )
    assert canonical_name == "baai/bge-m3"
    assert isinstance(embed_fn, reembed.LocalBGEM3Embedder)

    fake_model = MagicMock()
    fake_model.encode.return_value = {"dense_vecs": np.zeros((2, 1024), dtype=np.float64)}
    embed_fn._model = fake_model
    result = asyncio.run(embed_fn(["a", "b"]))
    assert result.shape == (2, 1024)
    assert result.dtype == np.float32
