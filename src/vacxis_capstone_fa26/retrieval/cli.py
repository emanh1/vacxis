"""CLI utility for running the VacxiS retrieval pipeline."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

from vacxis_capstone_fa26.retrieval.factory import build_retriever
from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vacxis.retrieval.cli")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VacxiS Retrieval CLI: Hybrid Search (Dense + Sparse) + RRF + Reranker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query text",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final top-k chunks to return",
    )
    parser.add_argument(
        "--top-k-dense",
        type=int,
        default=50,
        help="Number of dense search candidates",
    )
    parser.add_argument(
        "--top-k-sparse",
        type=int,
        default=50,
        help="Number of sparse search candidates",
    )
    parser.add_argument(
        "--rrf-top-k",
        type=int,
        default=30,
        help="Number of candidates after RRF fusion",
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        default=False,
        help="Skip cross-encoder reranking and output RRF top-k directly",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=RetrievalConfig.collection_name,
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default=RetrievalConfig.qdrant_host,
        help="Qdrant server host",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=RetrievalConfig.qdrant_port,
        help="Qdrant server port",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=RetrievalConfig.embed_model_path,
        help="Path to BGE-M3 model",
    )
    parser.add_argument(
        "--reranker-path",
        type=str,
        default=RetrievalConfig.reranker_model_path,
        help="Path to BGE-reranker model",
    )
    parser.add_argument(
        "--retriever-mode",
        choices=("naive", "advanced", "hybrid", "lightrag", "hybrid_lightrag"),
        default=RetrievalConfig.mode,
        help="Retriever backend mode",
    )
    parser.add_argument(
        "--merge-table-chunks",
        action="store_true",
        default=False,
        help="Merge split table chunks sharing (doc_id, header) before reranking",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path(RetrievalConfig.chunks_parquet_path),
        help="Path to chunks.parquet (used by naive/BM25 and graph reverse mapping)",
    )
    from vacxis_capstone_fa26.cli import add_lightrag_arguments

    add_lightrag_arguments(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output raw JSON array of retrieved chunks",
    )
    return parser.parse_args()


def _chunk_field(chunk: Any, name: str, default: Any = "") -> Any:
    """Read a field from either a RetrievedChunk dataclass or a plain dictionary."""
    if isinstance(chunk, dict):
        return chunk.get(name, default)
    return getattr(chunk, name, default)


def main() -> None:
    """Run retrieval CLI entrypoint."""
    args = parse_args()

    query = args.query.strip()
    if not query:
        logger.error("Query cannot be empty or whitespace-only.")
        sys.exit(1)

    config = RetrievalConfig(
        collection_name=args.collection_name,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        embed_model_path=args.model_path,
        reranker_model_path=args.reranker_path,
        top_k_dense=args.top_k_dense,
        top_k_sparse=args.top_k_sparse,
        rrf_top_k=args.rrf_top_k,
        top_k=args.top_k,
        skip_rerank=args.skip_rerank,
        mode=args.retriever_mode,
        merge_table_chunks=args.merge_table_chunks,
        lightrag_working_dir=args.lightrag_working_dir,
        lightrag_workspace=args.lightrag_workspace,
        lightrag_mode=args.lightrag_mode,
        lightrag_top_k=args.lightrag_top_k,
        lightrag_chunk_top_k=args.lightrag_chunk_top_k,
        lightrag_fusion=args.lightrag_fusion,
        chunks_parquet_path=args.chunks,
    )

    with build_retriever(config) as retriever:
        if config.mode in ("hybrid", "hybrid_lightrag"):
            results = retriever.retrieve(
                query=query,
                top_k=args.top_k,
                top_k_dense=args.top_k_dense,
                top_k_sparse=args.top_k_sparse,
                rrf_top_k=args.rrf_top_k,
                skip_rerank=args.skip_rerank,
            )
        else:
            results = retriever.retrieve(query=query, top_k=args.top_k)

    if args.json:
        data = [
            chunk.to_dict() if hasattr(chunk, "to_dict") else dict(chunk)
            for chunk in results
        ]
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    print(f"Query: {query}")
    print(f"Total retrieved: {len(results)}")
    for idx, chunk in enumerate(results, start=1):
        score = _chunk_field(chunk, "score", None)
        rerank_score = _chunk_field(chunk, "rerank_score", None)
        rrf_score = _chunk_field(chunk, "rrf_score", None)
        if score is not None:
            score_info = f"Score: {float(score):.4f}"
            if rerank_score is not None and rrf_score is not None:
                score_info += f" (Rerank: {float(rerank_score):.4f}, RRF: {float(rrf_score):.5f})"
            elif rrf_score is not None:
                score_info += f" (RRF: {float(rrf_score):.5f})"
        else:
            score_info = "Score: N/A"

        chunk_id = _chunk_field(chunk, "chunk_id", "")
        doc_title = _chunk_field(chunk, "doc_title", "")
        category = _chunk_field(chunk, "category", "")
        header = _chunk_field(chunk, "header", "")
        text = str(_chunk_field(chunk, "text", ""))

        print(f"[{idx}] {chunk_id} | {doc_title} | {category} | {score_info}")
        print(f"Header: {header}")
        snippet = text.replace("\n", " ").strip()
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        print(f"Snippet: {snippet}\n")


if __name__ == "__main__":
    main()
