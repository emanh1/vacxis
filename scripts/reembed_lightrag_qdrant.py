"""Re-embed LightRAG Qdrant collections locally using BAAI/bge-m3.

Reads existing LightRAG Qdrant collections (which already store text in
``payload["content"]``), computes normalized 1024-d dense embeddings with
local ``models/bge-m3`` via ``BGEM3FlagModel``, and upserts points with
identical IDs and payloads into ``lightrag_vdb_*_baai_bge_m3_1024d``.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
import logging
import os
from pathlib import Path
import threading
from typing import Any

from dotenv import load_dotenv
import numpy as np
from qdrant_client import QdrantClient, models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

logger = logging.getLogger("vacxis.reembed_lightrag")

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "bge-m3"
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").strip() or "http://localhost:6333"
DENSE_DIM = 1024
DEFAULT_ENCODE_BATCH_SIZE = 32
DEFAULT_SCROLL_BATCH_SIZE = 256
DEFAULT_UPSERT_BATCH_SIZE = 128
DEFAULT_MAX_LENGTH = 8192
WORKSPACE_ID_FIELD = "workspace_id"

COLLECTION_PAIRS: tuple[tuple[str, str], ...] = (
    (
        "lightrag_vdb_chunks_vietnamese_embedding_1024d",
        "lightrag_vdb_chunks_baai_bge_m3_1024d",
    ),
    (
        "lightrag_vdb_entities_vietnamese_embedding_1024d",
        "lightrag_vdb_entities_baai_bge_m3_1024d",
    ),
    (
        "lightrag_vdb_relationships_vietnamese_embedding_1024d",
        "lightrag_vdb_relationships_baai_bge_m3_1024d",
    ),
)


class LocalBGEM3Embedder:
    """Lazy-loading local BGEM3FlagModel wrapper for 1024-d dense embeddings."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        batch_size: int = DEFAULT_ENCODE_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        *,
        model: Any | None = None,
    ) -> None:
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be positive integers.")
        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = model
        self._lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is None:
            import torch
            from FlagEmbedding import BGEM3FlagModel

            use_fp16 = torch.cuda.is_available()
            logger.info(
                "Loading local BGE-M3 model from '%s' (use_fp16=%s)...",
                self.model_path,
                use_fp16,
            )
            self._model = BGEM3FlagModel(self.model_path, use_fp16=use_fp16)
            logger.info("Local BGE-M3 model loaded.")
        return self._model

    def encode_sync(self, texts: list[str]) -> np.ndarray:
        """Encode texts synchronously into L2-normalized float32 dense vectors."""
        if not texts:
            return np.empty((0, DENSE_DIM), dtype=np.float32)
        # Serialize lazy initialization, fast tokenizer access, and GPU forward passes across worker threads.
        with self._lock:
            output = self._get_model().encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        vectors = np.asarray(output["dense_vecs"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(texts), DENSE_DIM):
            raise ValueError(
                f"Expected embedding shape ({len(texts)}, {DENSE_DIM}), got {vectors.shape}."
            )
        return vectors

    async def __call__(self, texts: list[str], *args: Any, **kwargs: Any) -> np.ndarray:
        """Async wrapper matching LightRAG EmbeddingFunc contract."""
        return await asyncio.to_thread(self.encode_sync, texts)


def ensure_lightrag_target_collection(
    client: QdrantClient,
    target_collection: str,
    *,
    dense_dim: int = DENSE_DIM,
    recreate: bool = False,
) -> bool:
    """Create a LightRAG-compatible Qdrant collection with workspace_id tenant index.

    Returns:
        True if the collection is ready for migration (newly created or recreated),
        False if the target already exists and ``recreate`` is False.
    """
    if client.collection_exists(target_collection):
        if not recreate:
            return False
        logger.warning("Deleting existing target collection '%s'...", target_collection)
        client.delete_collection(collection_name=target_collection)

    client.create_collection(
        collection_name=target_collection,
        vectors_config=models.VectorParams(
            size=dense_dim,
            distance=models.Distance.COSINE,
        ),
        hnsw_config=models.HnswConfigDiff(payload_m=16, m=0),
    )
    client.create_payload_index(
        collection_name=target_collection,
        field_name=WORKSPACE_ID_FIELD,
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            is_tenant=True,
        ),
    )
    logger.info("Created target collection '%s' with tenant payload index.", target_collection)
    return True


def reembed_collection(
    client: QdrantClient,
    embedder: LocalBGEM3Embedder,
    source_collection: str,
    target_collection: str,
    *,
    recreate: bool = False,
    scroll_batch_size: int = DEFAULT_SCROLL_BATCH_SIZE,
    upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
) -> dict[str, Any]:
    """Copy points from source_collection to target_collection with re-embedded vectors."""
    if scroll_batch_size < 1 or upsert_batch_size < 1:
        raise ValueError("scroll_batch_size and upsert_batch_size must be positive integers.")
    if not client.collection_exists(source_collection):
        raise RuntimeError(f"Source collection not found in Qdrant: '{source_collection}'.")

    source_count = client.count(collection_name=source_collection, exact=True).count
    created = ensure_lightrag_target_collection(
        client,
        target_collection,
        dense_dim=DENSE_DIM,
        recreate=recreate,
    )
    if not created:
        target_count = client.count(collection_name=target_collection, exact=True).count
        logger.info(
            "Target collection '%s' already exists (%d points) and recreate=False; skipping.",
            target_collection,
            target_count,
        )
        return {
            "source": source_collection,
            "target": target_collection,
            "status": "skipped",
            "source_count": source_count,
            "target_count": target_count,
            "migrated": 0,
        }

    migrated = 0
    scroll_offset: Any = None
    while True:
        points, scroll_offset = client.scroll(
            collection_name=source_collection,
            limit=scroll_batch_size,
            offset=scroll_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        texts: list[str] = []
        payloads: list[dict[str, Any]] = []
        for point in points:
            payload = dict(point.payload or {})
            content = payload.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Point {point.id!r} in '{source_collection}' is missing nonblank 'content'."
                )
            texts.append(content)
            payloads.append(payload)

        vectors = embedder.encode_sync(texts)
        upserts = [
            models.PointStruct(
                id=point.id,
                vector=vectors[idx].tolist(),
                payload=payloads[idx],
            )
            for idx, point in enumerate(points)
        ]

        for start in range(0, len(upserts), upsert_batch_size):
            client.upsert(
                collection_name=target_collection,
                points=upserts[start : start + upsert_batch_size],
                wait=True,
            )
        migrated += len(upserts)
        logger.info(
            "Migrated %d/%d points: '%s' -> '%s'",
            migrated,
            source_count,
            source_collection,
            target_collection,
        )

        if scroll_offset is None:
            break

    target_count = client.count(collection_name=target_collection, exact=True).count
    if target_count != source_count:
        raise RuntimeError(
            f"Point count mismatch for '{target_collection}': "
            f"expected {source_count}, got {target_count}."
        )

    return {
        "source": source_collection,
        "target": target_collection,
        "status": "migrated",
        "source_count": source_count,
        "target_count": target_count,
        "migrated": migrated,
    }


def migrate_all_collections(
    qdrant_url: str = DEFAULT_QDRANT_URL,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    recreate: bool = False,
    encode_batch_size: int = DEFAULT_ENCODE_BATCH_SIZE,
    scroll_batch_size: int = DEFAULT_SCROLL_BATCH_SIZE,
    upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    collection_pairs: tuple[tuple[str, str], ...] = COLLECTION_PAIRS,
    client: QdrantClient | None = None,
    embedder: LocalBGEM3Embedder | None = None,
) -> list[dict[str, Any]]:
    """Re-embed all configured LightRAG Qdrant collections using local BGE-M3."""
    active_embedder = embedder or LocalBGEM3Embedder(
        model_path=model_path,
        batch_size=encode_batch_size,
        max_length=max_length,
    )

    with ExitStack() as stack:
        active_client = client
        if active_client is None:
            active_client = QdrantClient(url=qdrant_url, timeout=30)
            stack.callback(active_client.close)

        results: list[dict[str, Any]] = []
        for source_col, target_col in collection_pairs:
            results.append(
                reembed_collection(
                    client=active_client,
                    embedder=active_embedder,
                    source_collection=source_col,
                    target_collection=target_col,
                    recreate=recreate,
                    scroll_batch_size=scroll_batch_size,
                    upsert_batch_size=upsert_batch_size,
                )
            )
        return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for LightRAG Qdrant re-embedding."""
    parser = argparse.ArgumentParser(
        description="Re-embed LightRAG Qdrant collections locally using BAAI/bge-m3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--qdrant-url", type=str, default=DEFAULT_QDRANT_URL)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_ENCODE_BATCH_SIZE)
    parser.add_argument("--scroll-batch-size", type=int, default=DEFAULT_SCROLL_BATCH_SIZE)
    parser.add_argument("--upsert-batch-size", type=int, default=DEFAULT_UPSERT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--recreate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete and recreate target collections if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    results = migrate_all_collections(
        qdrant_url=args.qdrant_url,
        model_path=args.model_path,
        recreate=args.recreate,
        encode_batch_size=args.batch_size,
        scroll_batch_size=args.scroll_batch_size,
        upsert_batch_size=args.upsert_batch_size,
        max_length=args.max_length,
    )
    for row in results:
        logger.info(
            "Collection summary: %s -> %s [%s] (source=%d, target=%d, migrated=%d)",
            row["source"],
            row["target"],
            row["status"],
            row["source_count"],
            row["target_count"],
            row["migrated"],
        )


if __name__ == "__main__":
    main()
