"""Qdrant Vector Store Indexer for VacxiS Capstone.

Generates Dense + Sparse embeddings using BGE-M3 model for chunks stored in Parquet
and upserts them into a Qdrant collection with hybrid search capability.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from uuid import NAMESPACE_URL, uuid5
from vacxis_capstone_fa26.indexing.table_pipeline import (
    DEFAULT_TABLE_CACHE_PATH,
    VALID_TABLE_STRATEGIES,
    TableStrategy,
    prepare_records_for_indexing,
)
from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Sequence

import pyarrow.compute as pc
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models
import torch
from tqdm import tqdm

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vacxis.indexer")

# Constants & Defaults
DEFAULT_COLLECTION_NAME = RetrievalConfig.collection_name
DEFAULT_PARQUET_PATH = "chunks_storage/chunks.parquet"
DEFAULT_MODEL_PATH = RetrievalConfig.embed_model_path
DEFAULT_QDRANT_HOST = RetrievalConfig.qdrant_host
DEFAULT_QDRANT_PORT = RetrievalConfig.qdrant_port

DENSE_DIM = 1024
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_MAX_LENGTH = 1024
MEGA_BATCH_SIZE = 500
QDRANT_UPSERT_BATCH = 64


def convert_sparse_to_qdrant(
    lexical_weight: dict[str, float] | dict[int, float] | None,
) -> models.SparseVector:
    """Convert BGE-M3 lexical_weight dict to Qdrant SparseVector.

    BGE-M3 output: {token_id_str: weight, ...}
    Qdrant input:  SparseVector(indices=[int, ...], values=[float, ...])

    Args:
        lexical_weight: Token ID to weight mapping from BGE-M3.

    Returns:
        Qdrant SparseVector instance.
    """
    if not lexical_weight:
        return models.SparseVector(indices=[], values=[])

    indices = [int(token_id) for token_id in lexical_weight.keys()]
    values = [float(weight) for weight in lexical_weight.values()]
    return models.SparseVector(indices=indices, values=values)


def init_qdrant_collection(
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    recreate: bool = False,
    dense_dim: int = DENSE_DIM,
) -> None:
    """Check and create Qdrant collection configured for hybrid search.

    Args:
        client: Active QdrantClient instance.
        collection_name: Name of target collection.
        recreate: If True, delete and recreate if exists.
        dense_dim: Dimensionality of dense embedding vector.
    """
    existing_collections = [c.name for c in client.get_collections().collections]

    if collection_name in existing_collections:
        if recreate:
            logger.warning(
                "Collection '%s' already exists. Deleting for fresh index...",
                collection_name,
            )
            client.delete_collection(collection_name)
            logger.info("Collection '%s' deleted.", collection_name)
        else:
            logger.info(
                "Collection '%s' already exists and recreate=False. Retaining collection.",
                collection_name,
            )
            return

    logger.info("Creating Qdrant collection '%s'...", collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                ),
            ),
        },
    )
    logger.info("Collection '%s' created successfully.", collection_name)


def load_chunks_from_parquet(parquet_path: str | Path) -> list[dict[str, Any]]:
    """Load chunk records from Parquet file using native PyArrow.

    Args:
        parquet_path: Path to parquet file.

    Returns:
        List of dictionaries containing chunk fields.
    """
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    logger.info("Reading parquet file via PyArrow: %s", path)
    table = pq.read_table(str(path))
    records = table.to_pylist()
    logger.info("Loaded %d chunk records from %s", len(records), path)
    return records


def ingest_to_qdrant(
    parquet_path: str | Path = DEFAULT_PARQUET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    qdrant_host: str = DEFAULT_QDRANT_HOST,
    qdrant_port: int = DEFAULT_QDRANT_PORT,
    recreate: bool = False,
    embedding_batch_size: int = EMBEDDING_BATCH_SIZE,
    mega_batch_size: int = MEGA_BATCH_SIZE,
    embedding_max_length: int = EMBEDDING_MAX_LENGTH,
    upsert_batch_size: int = QDRANT_UPSERT_BATCH,
    *,
    table_strategy: TableStrategy = "raw",
    table_cache_path: str | Path = DEFAULT_TABLE_CACHE_PATH,
    table_summary_model: str | None = None,
    client: QdrantClient | None = None,
    embed_model: Any | None = None,
) -> int:
    """Generate BGE-M3 dense + sparse embeddings and upsert into Qdrant.

    Args:
        parquet_path: Path to chunks.parquet.
        model_path: Directory path of local BGE-M3 model.
        collection_name: Name of Qdrant collection.
        qdrant_host: Qdrant service host.
        qdrant_port: Qdrant service port.
        recreate: Whether to recreate collection if it already exists.
        embedding_batch_size: Batch size for model inference.
        mega_batch_size: Chunk size for encode and upsert cycles.
        embedding_max_length: Maximum sequence length for BGE-M3.
        upsert_batch_size: Sub-batch size for Qdrant upsert points.
        table_strategy: Table preprocessing strategy (raw, merge, summarize, summarize_dual).
        table_cache_path: Path to JSON cache of table prose summaries.
        table_summary_model: Optional model key for summary cache lookup.

    Returns:
        Total number of points upserted.
    """
    logger.info("QDRANT VECTOR STORE INGESTION (HYBRID DENSE + SPARSE)")

    # 1. Load records from Parquet using PyArrow
    if any(type(value) is not int or value < 1 for value in
           (embedding_batch_size, mega_batch_size, embedding_max_length, upsert_batch_size)):
        raise ValueError("Indexing batch sizes and max length must be positive integers.")
    records = load_chunks_from_parquet(parquet_path)
    ids = [row.get("chunk_id") for row in records]
    if any(not isinstance(cid, str) or not cid.strip() for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Chunks must have unique, nonblank chunk IDs.")
    if len(records) == 0:
        logger.warning("No records found in %s. Ingestion aborted.", parquet_path)
        return 0

    records, table_stats = prepare_records_for_indexing(
        records,
        strategy=table_strategy,
        cache_path=table_cache_path,
        summary_model=table_summary_model,
    )
    total_rows = len(records)
    if table_strategy != "raw":
        logger.info(
            "Table-aware preprocessing (%s): %d -> %d records | tables=%d | "
            "cache_hits=%d | cache_misses=%d | llm_calls=%d",
            table_strategy,
            table_stats["total_in"],
            table_stats["total_out"],
            table_stats["tables_total"],
            table_stats["cache_hits"],
            table_stats["cache_misses"],
            table_stats["llm_calls"],
        )

    # 2. Connect to Qdrant
    logger.info("Connecting to Qdrant at %s:%d ...", qdrant_host, qdrant_port)
    with ExitStack() as stack:
        if client is None:
            client = QdrantClient(host=qdrant_host, port=qdrant_port)
            stack.callback(client.close)
        client.get_collections()
        logger.info("Qdrant connection: OK")

        # 3. Setup Collection
        init_qdrant_collection(
            client=client,
            collection_name=collection_name,
            recreate=recreate,
            dense_dim=DENSE_DIM,
        )

        # 4. Load BGE-M3 Model
        use_fp16 = torch.cuda.is_available()
        logger.info(
            "Loading BGE-M3 model from '%s' (use_fp16=%s, cuda=%s)...",
            model_path,
            use_fp16,
            torch.cuda.is_available(),
        )
        model = embed_model
        if model is None:
            from FlagEmbedding import BGEM3FlagModel
            model = BGEM3FlagModel(str(model_path), use_fp16=use_fp16)
        logger.info("BGE-M3 model loaded successfully.")

        # 5. Process in mega-batches: encode -> convert -> upsert
        total_upserted = 0
        num_mega_batches = (total_rows + mega_batch_size - 1) // mega_batch_size

        pbar = stack.enter_context(tqdm(total=total_rows, desc="Qdrant Hybrid Upsert", unit="pts"))

        for mega_idx in range(num_mega_batches):
            start = mega_idx * mega_batch_size
            end = min(start + mega_batch_size, total_rows)
            batch_records = records[start:end]
            current_batch_size = len(batch_records)

            # 5a. Prepare texts to encode (embed_text takes precedence for dual table strategy)
            texts: list[str] = []
            for rec in batch_records:
                text = str(rec.get("embed_text") or rec.get("text", "")).strip()
                texts.append(text)

            # 5b. Generate Dense + Sparse embeddings
            output = model.encode(
                texts,
                batch_size=embedding_batch_size,
                max_length=embedding_max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )

            dense_vecs = output["dense_vecs"]
            sparse_weights = output["lexical_weights"]
            if len(dense_vecs) != current_batch_size or len(sparse_weights) != current_batch_size:
                raise ValueError("Embedding output count does not match chunk count.")

            # 5c. Build PointStruct list
            points: list[models.PointStruct] = []
            for i in range(current_batch_size):
                rec = batch_records[i]

                sparse_vec = convert_sparse_to_qdrant(sparse_weights[i])

                payload: dict[str, Any] = {
                    "chunk_id": str(rec.get("chunk_id", "")),
                    "doc_id": str(rec.get("doc_id", "")),
                    "doc_title": str(rec.get("doc_title", "")),
                    "category": str(rec.get("category", "")),
                    "parent_header": str(rec.get("parent_header", "")),
                    "header": str(rec.get("header", "")),
                    "content_type": str(rec.get("content_type", "")),
                    "text": str(rec.get("text", "")),
                    "metadata": rec.get("metadata") or {},
                }

                point = models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, str(rec["chunk_id"]))),
                    vector={
                        "dense": dense_vecs[i].tolist(),
                        "sparse": sparse_vec,
                    },
                    payload=payload,
                )
                points.append(point)

            # 5d. Upsert in sub-batches
            for upsert_start in range(0, len(points), upsert_batch_size):
                upsert_end = min(upsert_start + upsert_batch_size, len(points))
                sub_points = points[upsert_start:upsert_end]
                client.upsert(
                    collection_name=collection_name,
                    points=sub_points,
                    wait=True,
                )
                total_upserted += len(sub_points)

            pbar.update(current_batch_size)

        pbar.close()

        # 6. Verify collection info
        collection_info = client.get_collection(collection_name)
        logger.info(
            "Qdrant ingestion complete: %d points upserted, collection has %d points.",
            total_upserted,
            collection_info.points_count,
        )

        return total_upserted


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Index chunked corpus into Qdrant with hybrid BGE-M3 embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default=DEFAULT_PARQUET_PATH,
        help="Path to chunked parquet file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to directory containing BGE-M3 model.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default=DEFAULT_QDRANT_HOST,
        help="Qdrant host.",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=DEFAULT_QDRANT_PORT,
        help="Qdrant port.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help="Embedding batch size for model inference.",
    )
    parser.add_argument(
        "--mega-batch-size",
        type=int,
        default=MEGA_BATCH_SIZE,
        help="Mega-batch size for encode and upsert cycles.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=EMBEDDING_MAX_LENGTH,
        help="Max sequence length for BGE-M3 embedding.",
    )
    parser.add_argument(
        "--recreate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recreate collection if it already exists.",
    )
    parser.add_argument(
        "--table-strategy",
        choices=VALID_TABLE_STRATEGIES,
        default="raw",
        help="Table preprocessing strategy before embedding and upsert.",
    )
    parser.add_argument(
        "--table-cache",
        type=Path,
        default=DEFAULT_TABLE_CACHE_PATH,
        help="Path to JSON cache of table prose summaries.",
    )
    parser.add_argument(
        "--table-summary-model",
        type=str,
        default=None,
        help="Model key for table summary cache lookup (defaults to LIGHTRAG_LLM).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for Qdrant Indexer CLI."""
    args = parse_args()
    start_time = time.time()

    logger.info("Parquet        : %s", args.parquet_path)
    logger.info("Model          : %s", args.model_path)
    logger.info("Collection     : %s", args.collection_name)
    logger.info("Host:Port      : %s:%d", args.qdrant_host, args.qdrant_port)
    logger.info("Recreate       : %s", args.recreate)
    logger.info("Table strategy : %s", args.table_strategy)

    points_count = ingest_to_qdrant(
        parquet_path=args.parquet_path,
        model_path=args.model_path,
        collection_name=args.collection_name,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        recreate=args.recreate,
        embedding_batch_size=args.batch_size,
        mega_batch_size=args.mega_batch_size,
        embedding_max_length=args.max_length,
        table_strategy=args.table_strategy,
        table_cache_path=args.table_cache,
        table_summary_model=args.table_summary_model,
    )

    elapsed = time.time() - start_time
    logger.info(
        "INDEXING COMPLETE — Total points: %d | Time: %.1f seconds (%.2f minutes)",
        points_count,
        elapsed,
        elapsed / 60,
    )


if __name__ == "__main__":
    main()
