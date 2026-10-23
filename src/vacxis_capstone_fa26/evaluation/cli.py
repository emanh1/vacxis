"""CLI entrypoint for running VacxiS evaluation metrics."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from vacxis_capstone_fa26.evaluation.evaluator import (
    GenerationEvaluator,
    RetrievalEvaluator,
    VacxisRAGEvaluator,
)
from vacxis_capstone_fa26.evaluation.schemas import EvaluationSummary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vacxis.evaluation.cli")


def parse_args() -> argparse.Namespace:
    """Parse evaluation command line arguments."""
    parser = argparse.ArgumentParser(
        description="VacxiS Evaluation CLI: Comprehensive Evaluation for Retrieval and Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("test_dataset.json"),
        help="Path to golden test dataset (e.g. test_dataset.json)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to generation/retrieval results JSON file",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5,10",
        help="Comma-separated k cutoffs for retrieval evaluation",
    )
    parser.add_argument(
        "--eval-retrieval-only",
        action="store_true",
        default=False,
        help="Only evaluate retrieval metrics",
    )
    parser.add_argument(
        "--eval-generation-only",
        action="store_true",
        default=False,
        help="Only evaluate generation metrics",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Path to save evaluation summary JSON",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Path to save evaluation report Markdown",
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=None,
        help="Path to export per-query metrics Parquet table (PyArrow)",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        default=False,
        help="Print breakdowns by query type and difficulty to console",
    )
    return parser.parse_args()


def _print_clean_summary(summary: EvaluationSummary, show_breakdown: bool = False) -> None:
    """Print clean terminal summary without separator lines."""
    print(f"Task: {summary.task.upper()}")
    print(f"Total Evaluated Queries: {summary.num_queries}")
    print("\nOverall Metrics:")
    for k, v in sorted(summary.metric_aggregates.items()):
        print(f"  {k:28s}: mean={v.mean:.4f}  std={v.std:.4f}  min={v.min:.4f}  max={v.max:.4f}")

    if show_breakdown and summary.by_query_type:
        print("\nBreakdown by Query Type:")
        for qtype, scores in sorted(summary.by_query_type.items()):
            cnt = int(scores.get("count", 0))
            score_strs = [
                f"{k}={v:.3f}"
                for k, v in sorted(scores.items())
                if k not in ("count", "latency_seconds", "total_tokens")
            ][:5]
            print(f"  {qtype} (n={cnt}): " + ", ".join(score_strs))

    if show_breakdown and summary.by_difficulty:
        print("\nBreakdown by Difficulty:")
        for diff, scores in sorted(summary.by_difficulty.items()):
            cnt = int(scores.get("count", 0))
            score_strs = [
                f"{k}={v:.3f}"
                for k, v in sorted(scores.items())
                if k not in ("count", "latency_seconds", "total_tokens")
            ][:5]
            print(f"  {diff} (n={cnt}): " + ", ".join(score_strs))


def main() -> None:
    """Run evaluation CLI entrypoint."""
    args = parse_args()

    if not args.dataset.exists():
        logger.error("Dataset file not found: %s", args.dataset)
        sys.exit(1)

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset: list[dict[str, Any]] = json.load(f)

    logger.info("Loaded %d golden test items from %s", len(dataset), args.dataset)

    k_vals = [int(x.strip()) for x in args.k_values.split(",") if x.strip().isdigit()]
    if not k_vals:
        k_vals = [1, 3, 5, 10]

    # If results file is provided, evaluate it
    if args.results:
        if not args.results.exists():
            logger.error("Results file not found: %s", args.results)
            sys.exit(1)

        with open(args.results, "r", encoding="utf-8") as f:
            results: list[dict[str, Any]] = json.load(f)

        logger.info("Loaded %d result items from %s", len(results), args.results)

        if args.eval_retrieval_only:
            evaluator = RetrievalEvaluator(k_values=k_vals)
            retrieved_lists = [
                r.get("GENERATED_ANSWER", r).get("retrieved_chunks")
                or r.get("GENERATED_ANSWER", r).get("retrieved_chunk_ids")
                or []
                for r in results
            ]
            summary = evaluator.evaluate(queries=results, retrieved_per_query=retrieved_lists)
        elif args.eval_generation_only:
            evaluator = GenerationEvaluator()
            summary = evaluator.evaluate(queries=results, answers=results)
        else:
            rag_evaluator = VacxisRAGEvaluator(k_values=k_vals)
            summary = rag_evaluator.evaluate_results(dataset_items=dataset, generation_results=results)
    else:
        # Default mode: evaluate golden dataset reference chunks & reference answers
        logger.info("No --results provided. Evaluating dataset self-consistency and ground-truth references.")
        # Self-retrieval evaluation using ground truth chunks
        ret_lists = [q.get("RELEVANT_CHUNKS", []) for q in dataset]
        ret_eval = RetrievalEvaluator(k_values=k_vals)
        ret_summary = ret_eval.evaluate(queries=dataset, retrieved_per_query=ret_lists)

        # Self-generation evaluation using ground truth reference answers
        gen_eval = GenerationEvaluator()
        dummy_answers = [
            {
                "answer": q.get("REFERENCE_ANSWER", ""),
                "citations": [
                    c.get("CHUNK_ID")
                    for c in q.get("RELEVANT_CHUNKS", [])
                    if isinstance(c, dict) and c.get("RELEVANCE_SCORE", 0) > 0
                ],
                "retrieved_chunk_ids": [
                    c.get("CHUNK_ID")
                    for c in q.get("RELEVANT_CHUNKS", [])
                    if isinstance(c, dict)
                ],
                "is_refusal": (q.get("ANSWERABILITY") == "UNANSWERABLE"),
            }
            for q in dataset
        ]
        gen_summary = gen_eval.evaluate(queries=dataset, answers=dummy_answers)

        rag_eval = VacxisRAGEvaluator(k_values=k_vals)
        summary = rag_eval.evaluate_results(
            dataset_items=dataset,
            generation_results=[
                {
                    "QUERY_ID": q.get("QUERY_ID"),
                    "QUERY": q.get("QUERY"),
                    "GENERATED_ANSWER": dummy_answers[i],
                }
                for i, q in enumerate(dataset)
            ],
        )

    _print_clean_summary(summary, show_breakdown=args.breakdown)

    if args.out_json:
        VacxisRAGEvaluator.save_json(summary, args.out_json)

    if args.out_md:
        VacxisRAGEvaluator.save_markdown(summary, args.out_md)

    if args.out_parquet:
        VacxisRAGEvaluator.save_parquet(summary, args.out_parquet)


if __name__ == "__main__":
    main()
