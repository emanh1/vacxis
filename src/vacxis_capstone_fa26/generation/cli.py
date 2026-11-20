"""CLI utility for running the VacxiS generation and end-to-end RAG pipeline."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from vacxis_capstone_fa26.resilience import add_retry_arguments, retry_from_args, atomic_json, fingerprint

import argparse
from dataclasses import dataclass
import json
import logging
import sys
from pathlib import Path
from typing import Any

from vacxis_capstone_fa26.generation.generator import MedicalGenerator
from vacxis_capstone_fa26.generation.pipeline import VacxisRAGPipeline
from vacxis_capstone_fa26.generation.prompts import get_chunk_field
from vacxis_capstone_fa26.generation.schemas import GenerationConfig
from vacxis_capstone_fa26.generation.datasets import load_dataset

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
        help="OpenRouter model override for VACXIS_GENERATION_MODEL",
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
    "--dataset-format",
    choices=["deepeval", "legacy"],
    default="deepeval",
    help="Input dataset format",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generation by skipping queries already saved in the output file",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--baseline", action="store_true", help="Naive BM25 retrieval")
    modes.add_argument("--advanced-rag", action="store_true", help="Dense-only retrieval")
    parser.add_argument("--chunks", type=Path, default=Path("chunks_storage/chunks.parquet"))
    parser.add_argument("--timeout", type=float, default=480)
    add_retry_arguments(parser)
    from vacxis_capstone_fa26.cli import add_retrieval_arguments
    add_retrieval_arguments(parser)
    return parser.parse_args()

def _get_query_id(item: dict[str, Any]) -> str | None:
    """Extract a stable query ID from either supported dataset format."""
    metadata = item.get("additional_metadata") or {}

    query_id = (
        metadata.get("query_id")
        or item.get("query_id")
        or item.get("QUERY_ID")
        or item.get("id")
    )

    if query_id is None or not str(query_id).strip():
        return None

    return str(query_id)


def save_results(results: list[dict[str, Any]], out_path: Path) -> None:
    """Atomically save generation results to avoid partially written JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    temp_path.replace(out_path)


def load_resume_results(
    out_path: Path,
    items: list,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Load existing results and validate them against the input dataset."""
    if not out_path.exists():
        logger.info("No previous output found. Starting fresh.")
        return [], set()

    try:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read existing results file: {exc}") from exc

    if not isinstance(existing, list):
        raise ValueError("Existing results file must contain a JSON list.")

    input_ids = [_get_query_id(item.original) for item in items]

    if any(query_id is None for query_id in input_ids):
        raise ValueError(
            "Resume requires every input record to have a query ID. "
            "Use query_id, QUERY_ID, or additional_metadata.query_id."
        )

    if len(input_ids) != len(set(input_ids)):
        raise ValueError("Input dataset contains duplicate query IDs.")

    valid_ids = set(input_ids)
    completed_ids: set[str] = set()

    for result in existing:
        if not isinstance(result, dict):
            raise ValueError("Existing results contain an invalid record.")

        query_id = _get_query_id(result)

        if query_id is None:
            raise ValueError(
                "An existing result has no query ID. Cannot safely resume."
            )

        if query_id not in valid_ids:
            raise ValueError(
                f"Existing result ID {query_id!r} is not in the input dataset. "
                "Check that you are using the correct output file."
            )

        if query_id in completed_ids:
            raise ValueError(
                f"Duplicate query ID {query_id!r} in existing results."
            )

        completed_ids.add(query_id)

    logger.info(
        "Resuming from %d existing results. %d questions remaining.",
        len(completed_ids),
        len(items) - len(completed_ids),
    )

    return existing, completed_ids

@contextmanager
def build_retriever(args):
    """Build and yield the configured retriever for generation CLI."""
    from vacxis_capstone_fa26.cli import retrieval_config

    try:
        if getattr(args, "baseline", False):
            from vacxis_capstone_fa26.evaluation.datasets import load_chunk_texts
            from vacxis_capstone_fa26.retrieval.bm25 import BM25Retriever

            class Adapter(BM25Retriever):
                def retrieve(self, query, top_k=5, **kwargs):
                    return super().retrieve(query, top_k)

            yield Adapter(load_chunk_texts(args.chunks))
            return

        if getattr(args, "advanced_rag", False):
            from vacxis_capstone_fa26.retrieval.dense import DenseRetriever

            with DenseRetriever(retrieval_config(args)) as retriever:
                yield retriever
            return

        from vacxis_capstone_fa26.retrieval.factory import build_retriever as build_factory_retriever

        with build_factory_retriever(retrieval_config(args)) as retriever:
            yield retriever
    except ImportError as exc:
        logger.error("Retrieval module requires PyTorch/CUDA dependencies (%s).", exc)
        sys.exit(1)


def main() -> None:
    """Run generation CLI entrypoint."""
    from vacxis_capstone_fa26.cli import retrieval_config
    args = parse_args()

    gen_config = GenerationConfig(
        temperature=args.temperature,
        dry_run=args.dry_run, timeout=args.timeout, retry_policy=retry_from_args(args),
    )
    if args.model is not None:
        gen_config.model = args.model
    if not args.dry_run:
        from vacxis_capstone_fa26.llm import validate_config
        validate_config(gen_config.model)
    generator = MedicalGenerator(config=gen_config)

    # Mode 1: Batch process from input file (e.g. test_dataset.json)

    if args.input_file:
        if not args.input_file.exists():
            logger.error("Input file not found: %s", args.input_file)
            sys.exit(1)

        try:
            items = load_dataset(
                path=args.input_file,
                dataset_format=args.dataset_format,
                limit=args.limit,
            )
        except (ValueError, KeyError, TypeError) as exc:
            logger.error("Failed to load dataset: %s", exc)
            sys.exit(1)

        logger.info(
            "Processing %d items using %s format...",
            len(items),
            args.dataset_format,
        )

        out_path = args.out_file or Path("generation_results.json")

        manifest = out_path.with_suffix(out_path.suffix + ".manifest.json")
        settings = vars(args).copy()
        for key in ("resume", "out_file", "retries", "retry_initial", "retry_max", "timeout", "llm_call_delay"):
            settings.pop(key, None)
        signature = fingerprint({"items": [item.original for item in items], "settings": settings,
                                 "generation": {k: v for k, v in asdict(gen_config).items() if k not in ("retry_policy", "timeout")}})
        if args.resume and out_path.exists():
            if not manifest.exists() or json.loads(manifest.read_text()).get("fingerprint") != signature:
                raise ValueError("Generation checkpoint settings changed or manifest missing; use a new output file.")
        if args.resume:
            try:
                results, completed_ids = load_resume_results(out_path, items)
            except ValueError as exc:
                logger.error("Cannot resume generation: %s", exc)
                sys.exit(1)
        else:
            results = []
            completed_ids = set()

            if out_path.exists():
                logger.warning(
                    "Starting fresh. Existing output file will be overwritten: %s",
                    out_path,
                )

        if not args.resume:
            save_results([], out_path)
        atomic_json(manifest, {"fingerprint": signature})

        if args.use_ground_truth_chunks:
            logger.info("Using provided ground-truth contexts.")

            for item in items:
                query_id = _get_query_id(item.original)

                if args.resume and query_id in completed_ids:
                    logger.info("Skipping completed query: %s", query_id)
                    continue
                
                logger.info(
                    "Generating query %s (%s)",
                    item.query,
                    query_id or "no ID",
                )

                ans = generator.generate(
                    query=item.query,
                    chunks=item.chunks,
                )

                result = item.original.copy()
                result["GENERATED_ANSWER"] = ans.to_dict()

                results.append(result)

                if query_id:
                    completed_ids.add(query_id)

                save_results(results, out_path)

        else:
            with build_retriever(args) as retriever:
                pipeline = VacxisRAGPipeline(
                    retriever=retriever,
                    generator=generator,
                )

                for item in items:
                    query_id = _get_query_id(item.original)

                    if args.resume and query_id in completed_ids:
                        logger.info("Skipping completed query: %s", query_id)
                        continue
                    
                    logger.info(
                        "Generating query %s (%s)",
                        item.query,
                        query_id or "no ID",
                    )

                    ans = pipeline.answer(
                        query=item.query,
                        top_k=args.top_k,
                        skip_rerank=args.skip_rerank,
                    )

                    result = item.original.copy()
                    result["GENERATED_ANSWER"] = ans.to_dict()

                    results.append(result)

                    if query_id:
                        completed_ids.add(query_id)

                    save_results(results, out_path)
        save_results(results, out_path)
        
        logger.info(
            "Saved %d generation results to %s",
            len(results),
            out_path,
        )

        return

    # Mode 2: Single interactive query
    if not args.query or not args.query.strip():
        logger.error("Please provide either --query or --input-file.")
        sys.exit(1)

    query = args.query
    with build_retriever(args) as retriever:
        pipeline = VacxisRAGPipeline(retriever=retriever, generator=generator)
        answer_obj = pipeline.answer(query=query.strip(), top_k=args.top_k, skip_rerank=args.skip_rerank)

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
