"""CLI utility for running the VacxiS generation and end-to-end RAG pipeline."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from vacxis_capstone_fa26.generation.generator import MedicalGenerator
from vacxis_capstone_fa26.generation.pipeline import VacxisRAGPipeline
from vacxis_capstone_fa26.generation.prompts import get_chunk_field
from vacxis_capstone_fa26.generation.schemas import GenerationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vacxis.generation.cli")


@dataclass
class _GroundTruthChunk:
    chunk_id: str
    text: str
    doc_title: str = ""
    category: str = ""
    header: str = ""
    score: float = 0.0


def parse_args() -> argparse.Namespace:
    """Parse generation CLI command line arguments."""
    parser = argparse.ArgumentParser(
        description="VacxiS Generation CLI: Grounded Medical Answer Synthesis with Citations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Medical question to answer",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name for generation (e.g. gpt-4o-mini, gemini/gemini-1.5-flash, claude-3-5-haiku-20241022)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for LLM inference",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of chunks to retrieve for generation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run offline extractive answer synthesis without calling LLM APIs",
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        default=False,
        help="Skip cross-encoder reranking during retrieval",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output raw JSON response",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Input JSON file containing queries (e.g. test_dataset.json)",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Output JSON path to save generation results",
    )
    parser.add_argument(
        "--use-ground-truth-chunks",
        action="store_true",
        default=False,
        help="Use RELEVANT_CHUNKS from input file instead of running retrieval",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries processed from input file",
    )
    return parser.parse_args()


def main() -> None:
    """Run generation CLI entrypoint."""
    args = parse_args()

    model_name = args.model if args.model is not None else os.environ.get("VACXIS_LLM_MODEL", "gpt-4o-mini")
    gen_config = GenerationConfig(
        model=model_name,
        temperature=args.temperature,
        dry_run=args.dry_run,
    )
    generator = MedicalGenerator(config=gen_config)

    # Mode 1: Batch process from input file (e.g. test_dataset.json)
    if args.input_file:
        if not args.input_file.exists():
            logger.error("Input file not found: %s", args.input_file)
            sys.exit(1)

        with open(args.input_file, "r", encoding="utf-8") as f:
            data: list[dict[str, Any]] = json.load(f)

        if not isinstance(data, list):
            logger.error("Expected a list of query items in %s", args.input_file)
            sys.exit(1)

        items = data[:args.limit] if args.limit else data
        logger.info("Processing %d items from %s...", len(items), args.input_file)

        results: list[dict[str, Any]] = []

        if args.use_ground_truth_chunks:
            logger.info("Using ground truth RELEVANT_CHUNKS directly (bypassing retrieval).")
            for item in items:
                q_text = item.get("QUERY") or ""
                raw_chunks = item.get("RELEVANT_CHUNKS") or []
                chunks = [
                    _GroundTruthChunk(
                        chunk_id=c.get("CHUNK_ID", "") if isinstance(c, dict) else getattr(c, "chunk_id", ""),
                        doc_title=(item.get("ENTITIES") or {}).get("vaccine", "") if isinstance(item, dict) else "",
                        text=c.get("TEXT", "") if isinstance(c, dict) else getattr(c, "text", ""),
                        score=float((c.get("RELEVANCE_SCORE") if isinstance(c, dict) else getattr(c, "score", 0.0)) or 0.0),
                    )
                    for c in raw_chunks
                ]
                ans = generator.generate(query=q_text, chunks=chunks)
                res_dict = item.copy()
                res_dict["GENERATED_ANSWER"] = ans.to_dict()
                results.append(res_dict)
        else:
            try:
                from vacxis_capstone_fa26.retrieval.retriever import VacxisRetriever
                from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig
            except ImportError as exc:
                logger.error("Retrieval module requires PyTorch/CUDA dependencies (%s).", exc)
                sys.exit(1)

            ret_config = RetrievalConfig(top_k=args.top_k, skip_rerank=args.skip_rerank)
            with VacxisRetriever(config=ret_config) as retriever:
                pipeline = VacxisRAGPipeline(retriever=retriever, generator=generator)
                for item in items:
                    q_text = item.get("QUERY") or ""
                    ans = pipeline.answer(query=q_text, top_k=args.top_k, skip_rerank=args.skip_rerank)
                    res_dict = item.copy()
                    res_dict["GENERATED_ANSWER"] = ans.to_dict()
                    results.append(res_dict)

        out_path = args.out_file or Path("generation_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("Saved %d generation results to %s", len(results), out_path)
        return

    # Mode 2: Single interactive query
    if not args.query or not args.query.strip():
        logger.error("Please provide either --query or --input-file.")
        sys.exit(1)

    query = args.query.strip()
    try:
        from vacxis_capstone_fa26.retrieval.retriever import VacxisRetriever
        from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig
    except ImportError as exc:
        logger.error("Retrieval module requires PyTorch/CUDA dependencies (%s).", exc)
        sys.exit(1)

    ret_config = RetrievalConfig(top_k=args.top_k, skip_rerank=args.skip_rerank)

    with VacxisRetriever(config=ret_config) as retriever:
        pipeline = VacxisRAGPipeline(retriever=retriever, generator=generator)
        answer_obj = pipeline.answer(query=query, top_k=args.top_k, skip_rerank=args.skip_rerank)

    if args.json:
        print(json.dumps(answer_obj.to_dict(), ensure_ascii=False, indent=2))
        return

    print(f"Query: {answer_obj.query}")
    print(f"Model: {answer_obj.model}")
    print(f"Latency: {answer_obj.latency_seconds:.3f}s")
    print(f"Refusal: {answer_obj.is_refusal}")
    print(f"Citations: {answer_obj.citations}")
    if answer_obj.invalid_citations:
        print(f"Invalid Citations: {answer_obj.invalid_citations}")
    print("\nAnswer:")
    print(answer_obj.answer)
    print("\nRetrieved Context:")
    for idx, c in enumerate(answer_obj.retrieved_chunks, start=1):
        print(f"[{idx}] {get_chunk_field(c, 'chunk_id', '')} | {get_chunk_field(c, 'doc_title', '')} | {get_chunk_field(c, 'category', '')}")


if __name__ == "__main__":
    main()
