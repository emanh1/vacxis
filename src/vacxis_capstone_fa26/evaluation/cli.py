"""CLI entrypoint for running VacxiS DeepEval evaluation metrics."""
from __future__ import annotations

from vacxis_capstone_fa26.resilience import add_retry_arguments, retry_from_args
from vacxis_capstone_fa26.evaluation.judge import OpenRouterJudgeLLM

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from vacxis_capstone_fa26.evaluation import (
    VacxisRAGEvaluator,
    load_chunk_texts,
    require_judge_credentials,
    resolve_judge_model,
)
from vacxis_capstone_fa26.evaluation.datasets import to_record
from vacxis_capstone_fa26.evaluation.ir_metrics import DEFAULT_K_VALUES
from vacxis_capstone_fa26.evaluation.schemas import EvaluationSummary
from vacxis_capstone_fa26.generation.prompts import get_chunk_field
from vacxis_capstone_fa26.generation.schemas import GenerationConfig, extract_chunk_aliases

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vacxis.evaluation.cli")

_DEFAULT_UNANSWERABLE = Path("test_dataset.json")
_DEFAULT_CHUNKS = Path("chunks_storage/chunks.parquet")
_MAIN_DATASET = Path("evaluation/eval_dataset_300_final_fixed.json")


def parse_args() -> argparse.Namespace:
    """Parse evaluation command line arguments."""
    parser = argparse.ArgumentParser(
        description="VacxiS DeepEval CLI: RAG evaluation via DeepEval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Golden dataset JSON. In live mode, defaults to the main dataset; "
        "in precomputed mode, defaults to the results file itself",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Precomputed generation results JSON; when set, the RAG pipeline "
        "is skipped and these results are scored instead",
    )
    parser.add_argument(
        "--unanswerable",
        type=Path,
        default=_DEFAULT_UNANSWERABLE,
        help="Flat dataset JSON with unanswerable queries (evaluated separately)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="DeepEval judge model. Defaults to VACXIS_EVALUATION_MODEL, then "
        "VACXIS_GENERATION_MODEL, then gpt-oss-120b",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Passing threshold for each metric",
    )
    parser.add_argument(
        "--skip-geval",
        action="store_true",
        help="Skip the GEval correctness metric",
    )
    parser.add_argument(
        "--skip-unanswerable",
        action="store_true",
        help="Skip the separate unanswerable-set evaluation pass",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run only deterministic IR metrics without invoking LLM judge models",
    )
    parser.add_argument(
        "--skip-ir",
        action="store_true",
        help="Skip deterministic IR metrics during full evaluation",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=_DEFAULT_CHUNKS,
        help="Path to chunks.parquet for hydrating retrieval context",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Save evaluation summary JSON to this path",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Save evaluation Markdown report to this path",
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=None,
        help="Export per-query metrics as Parquet",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Print breakdowns by stratification axis",
    )
    # Live-mode pipeline arguments (used when --results is absent).
    baselines = parser.add_mutually_exclusive_group()
    baselines.add_argument(
        "--baseline", action="store_true",
        help="Use BM25 over --chunks for live retrieval and keep the standard generator",
    )
    baselines.add_argument(
        "--advanced-rag", action="store_true",
        help="Original query -> BGE-M3 dense embedding -> Qdrant top-k -> standard generator",
    )
    parser.add_argument(
        "--retriever-mode",
        choices=("naive", "advanced", "hybrid", "lightrag", "hybrid_lightrag"),
        default=None,
        help="Explicit retriever backend mode for live evaluation (default: hybrid)",
    )
    parser.add_argument(
        "--merge-table-chunks",
        action="store_true",
        default=False,
        help="Merge split table chunks sharing (doc_id, header) before reranking",
    )
    from vacxis_capstone_fa26.cli import add_lightrag_arguments

    add_lightrag_arguments(parser)
    parser.add_argument("--collection-name", type=str, default="VacxiS_Capstone_FA26")
    parser.add_argument("--qdrant-host", type=str, default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--model-path", type=str, default="models/bge-m3")
    parser.add_argument("--reranker-path", type=str, default="models/bge-reranker")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--ir-k-values", type=int, nargs="+", default=None, metavar="K",
        help="IR cutoffs; defaults to 1, 3, 5, 10, capped at --top-k in live mode",
    )
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument(
        "--timeout",
        type=float,
        default=480,
        help="Per-request LLM timeout in seconds for the live pipeline "
        "(default: 480)",
    )
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=None,
        help="Per-task timeout in seconds for DeepEval's judge-metric phase. "
        "When set, DeepEval's per-task budget (default 180s) is raised so slow "
        "judge models do not cancel in-flight cases mid-run",
    )
    add_retry_arguments(parser)
    parser.add_argument("--judge-request-timeout", type=float, default=360)
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, help="Explicit per-query report checkpoint (precomputed evaluation)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    retry_from_args(args)
    if args.resume and args.checkpoint is None:
        parser.error("--resume requires --checkpoint.")
    if args.checkpoint is not None and (args.results is None or args.retrieval_only):
        parser.error("--checkpoint requires --results and full DeepEval evaluation.")
    if args.retriever_mode is not None and (args.baseline or args.advanced_rag):
        parser.error("--retriever-mode cannot be combined with --baseline or --advanced-rag.")
    if args.retriever_mode is not None and args.results is not None:
        parser.error("--retriever-mode requires live evaluation without --results.")
    if args.advanced_rag and args.results is not None:
        parser.error("--advanced-rag requires live evaluation without --results.")
    if args.advanced_rag and not args.retrieval_only and not args.chunks.is_file():
        parser.error(f"Full --advanced-rag evaluation requires a chunk corpus at {args.chunks}.")
    if args.baseline and args.results is not None:
        parser.error("--baseline requires live evaluation without --results.")
    if args.baseline and not args.chunks.is_file():
        parser.error(f"--baseline requires a chunk corpus at {args.chunks}.")
    if args.retriever_mode == "naive" and not args.chunks.is_file():
        parser.error(f"--retriever-mode naive requires a chunk corpus at {args.chunks}.")
    if args.results is None and args.top_k < 1:
        parser.error("--top-k must be positive.")
    if args.lightrag_top_k < 1 or args.lightrag_chunk_top_k < 1:
        parser.error("--lightrag-top-k and --lightrag-chunk-top-k must be positive.")
    if args.ir_k_values is not None and any(k < 1 for k in args.ir_k_values):
        parser.error("--ir-k-values must contain positive integers.")
    if args.results is None:
        if args.ir_k_values is not None and max(args.ir_k_values) > args.top_k:
            parser.error("--top-k must be at least the largest --ir-k-values cutoff.")
        args.ir_k_values = args.ir_k_values or [k for k in DEFAULT_K_VALUES if k <= args.top_k]
    else:
        args.ir_k_values = args.ir_k_values or list(DEFAULT_K_VALUES)
    args.ir_k_values = sorted(set(args.ir_k_values))
    return args


def _print_summary(summary: EvaluationSummary, show_breakdown: bool) -> None:
    """Print a clean terminal summary without separator lines."""
    print(f"Task: {summary.task.upper()}")
    print(f"Total evaluated queries: {summary.num_queries}")
    print("\nOverall Metrics:")
    for name in summary.metric_names:
        agg = summary.metric_aggregates.get(name)
        if agg is None:
            continue
        print(
            f"  {name:36s}: mean={agg.mean:.4f}  min={agg.min:.4f}  "
            f"max={agg.max:.4f}  pass_rate={agg.pass_rate:.2%}  n={agg.count}"
        )

    if show_breakdown:
        for label, table in (
            ("Answerability", summary.by_answerability),
            ("Complexity", summary.by_complexity),
            ("Query Style", summary.by_query_style),
            ("Difficulty", summary.by_difficulty),
        ):
            if not table:
                continue
            print(f"\nBreakdown by {label}:")
            for group, scores in sorted(table.items()):
                n = int(scores.get("count", 0))
                first = [
                    f"{k}={v:.3f}"
                    for k, v in list(scores.items())
                    if k != "count"
                ][:4]
                print(f"  {group} (n={n}): " + ", ".join(first))


def _save_parquet(summary: EvaluationSummary, path: Path) -> None:
    """Export per-query metric scores to a Parquet file via PyArrow."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for item in summary.per_query_results:
        row: dict[str, Any] = {
            "query_id": str(item.get("query_id", "")),
            "task": summary.task,
        }
        for dim, val in (item.get("metadata") or {}).items():
            row[f"meta_{dim}"] = str(val)
        for metric, score in (item.get("metric_scores") or {}).items():
            row[f"metric_{metric}"] = float(score)
        rows.append(row)

    if not rows:
        logger.warning("No per-query rows to write to Parquet.")
        return
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    logger.info("Saved %d rows to Parquet %s", len(rows), path)


def _load_dataset_items(path: Path) -> list[dict[str, Any]]:
    """Load a golden dataset file and normalize both shapes to a raw list."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "goldens" in payload:
        items = payload["goldens"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"{path} must be a JSON list or a goldens object.")
    if not isinstance(items, list):
        raise ValueError(f"goldens in {path} must be a list.")
    return items


def _load_flat_dataset(path: Path) -> list[dict[str, Any]]:
    """Load the flat unanswerable test_dataset.json and normalize QUERY_ID."""
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} must be a JSON list.")
    return records


def _effective_retriever_mode(args: argparse.Namespace) -> str:
    """Resolve the active retriever mode from mutually exclusive CLI flags."""
    if getattr(args, "baseline", False):
        return "naive"
    if getattr(args, "advanced_rag", False):
        return "advanced"
    return getattr(args, "retriever_mode", None) or "hybrid"


def _build_advanced_retriever(args: argparse.Namespace):
    from vacxis_capstone_fa26.cli import retrieval_config
    from vacxis_capstone_fa26.retrieval.dense import DenseRetriever

    return DenseRetriever(config=retrieval_config(args))


def _attach_retrieval_traces(summary: EvaluationSummary, traces: dict[str, Any]) -> None:
    if not traces:
        return
    for row in summary.per_query_results:
        if row["query_id"] in traces:
            row.setdefault("metadata", {})["retrieval_trace"] = traces[row["query_id"]]


def _build_answer_fn(args: argparse.Namespace, chunk_texts: dict[str, str]):
    """Build a live-pipeline answer callback (lazy-imports heavy deps).

    Returns ``(pipeline, answer_fn)``; the caller must keep ``pipeline`` open
    (context manager) for the entire evaluation and close it after.
    """
    from vacxis_capstone_fa26.cli import retrieval_config
    from vacxis_capstone_fa26.generation.generator import MedicalGenerator
    from vacxis_capstone_fa26.generation.pipeline import VacxisRAGPipeline
    from vacxis_capstone_fa26.generation.schemas import GenerationConfig
    from vacxis_capstone_fa26.retrieval.factory import build_retriever

    gen_config = GenerationConfig(timeout=args.timeout, retry_policy=retry_from_args(args))
    generator = MedicalGenerator(config=gen_config)
    eff_mode = _effective_retriever_mode(args)
    if args.baseline:
        from vacxis_capstone_fa26.retrieval.bm25 import BM25Retriever

        retriever = BM25Retriever(chunk_texts)
        pipeline = VacxisRAGPipeline(retriever=retriever, generator=generator)
    elif args.advanced_rag:
        retriever = _build_advanced_retriever(args)
        pipeline = VacxisRAGPipeline(retriever=retriever, generator=generator)
    else:
        ret_cfg = retrieval_config(args)
        ret_cfg.mode = eff_mode
        if eff_mode == "naive":
            retriever = build_retriever(ret_cfg, chunk_texts=chunk_texts)
        else:
            retriever = build_retriever(ret_cfg)
        pipeline = VacxisRAGPipeline(retriever=retriever, generator=generator)

    traces: dict[str, Any] = {}

    def answer_fn(query: str, golden: dict) -> dict:
        if args.baseline or eff_mode == "advanced":
            chunks = retriever.retrieve(query, top_k=args.top_k)
            if eff_mode == "advanced" and hasattr(retriever, "last_trace"):
                traces[golden["query_id"]] = dict(retriever.last_trace)
            return pipeline.answer(query=query, top_k=args.top_k, override_chunks=chunks).to_dict()
        return pipeline.answer(query=query, top_k=args.top_k).to_dict()

    answer_fn.retrieval_traces = traces
    return pipeline, answer_fn


def _evaluate_live(
    evaluator: VacxisRAGEvaluator,
    dataset_items: list[dict[str, Any]],
    args: argparse.Namespace,
    chunk_texts: dict[str, str],
) -> EvaluationSummary:
    """Run the RAG pipeline per golden, then score the live answers with DeepEval."""
    pipeline, answer_fn = _build_answer_fn(args, chunk_texts)
    with pipeline:
        summary = evaluator.evaluate_live(dataset_items, answer_fn, chunk_texts=chunk_texts)
    _record_retrieval_run(summary, args)
    _attach_retrieval_traces(summary, answer_fn.retrieval_traces)
    return summary


def _evaluate_live_retrieval(
    evaluator: VacxisRAGEvaluator,
    dataset_items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> EvaluationSummary:
    """Run live retrieval over goldens and evaluate deterministic IR metrics."""
    from contextlib import ExitStack

    results: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    eff_mode = _effective_retriever_mode(args)
    with ExitStack() as resources:
        if args.baseline:
            from vacxis_capstone_fa26.retrieval.bm25 import BM25Retriever

            retriever = BM25Retriever(load_chunk_texts(args.chunks))
        elif args.advanced_rag:
            retriever = resources.enter_context(_build_advanced_retriever(args))
        else:
            from vacxis_capstone_fa26.cli import retrieval_config
            from vacxis_capstone_fa26.retrieval.factory import build_retriever

            ret_cfg = retrieval_config(args)
            ret_cfg.mode = eff_mode
            retriever = resources.enter_context(build_retriever(ret_cfg))

        for item in dataset_items:
            record = to_record(item)
            chunks = retriever.retrieve(query=record.query, top_k=args.top_k)
            if eff_mode == "advanced" and hasattr(retriever, "last_trace"):
                traces[record.query_id] = dict(retriever.last_trace)
            generated_answer: dict[str, Any] = {
                "retrieved_chunk_ids": [get_chunk_field(c, "chunk_id", "") for c in chunks],
                "answer": "",
                "is_refusal": False,
            }
            aliases = extract_chunk_aliases(chunks)
            if aliases:
                generated_answer["retrieved_chunk_aliases"] = aliases
            results.append(
                {
                    "QUERY_ID": record.query_id,
                    "GENERATED_ANSWER": generated_answer,
                }
            )
    summary = evaluator.evaluate_retrieval(dataset_items, results)
    _attach_retrieval_traces(summary, traces)
    return summary


def _apply_judge_timeout(args: argparse.Namespace) -> None:
    """Raise DeepEval's per-task judge timeout when --judge-timeout is set.

    DeepEval enforces a 180s per-task budget by default; a slow judge model
    (or heavy GEval) that exceeds it gets its in-flight cases cancelled at
    the gather step, which is exactly what truncates a run to fewer results
    than goldens. Setting the env var here (before ``evaluate`` runs) lets
    ``get_settings`` pick up the override.
    """
    if args.judge_timeout is None:
        return
    os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = str(args.judge_timeout)
    logger.info(
        "DeepEval per-task judge timeout set to %.0fs", args.judge_timeout
    )


def main() -> None:
    """Run evaluation CLI entrypoint.

    By default the RAG pipeline runs live over the main golden dataset and the
    unanswerable set, then DeepEval scores each answer. Passing ``--results``
    switches to a fast path that scores precomputed outputs without touching the
    pipeline. Passing ``--retrieval-only`` scores only deterministic IR metrics
    without requiring judge credentials.
    """
    args = parse_args()
    _apply_judge_timeout(args)

    if args.retrieval_only:
        evaluator = VacxisRAGEvaluator(
            threshold=args.threshold,
            enable_geval=False,
            enable_ir_metrics=True,
            ir_k_values=args.ir_k_values,
        )
        if args.results is not None:
            results = json.loads(args.results.read_text(encoding="utf-8"))
            if not isinstance(results, list):
                logger.error("Results file %s must be a JSON list.", args.results)
                sys.exit(1)
            dataset = _load_dataset_items(args.dataset) if args.dataset is not None else results
            summary = evaluator.evaluate_retrieval(dataset, results)
            _record_retrieval_run(summary, args)
            _print_summary(summary, args.breakdown)
            _maybe_export(summary, args)

            unanswerable = [] if args.skip_unanswerable else _load_flat_dataset(args.unanswerable)
            if unanswerable and args.unanswerable != args.results and not args.skip_unanswerable:
                stubs = _unanswerable_stubs(unanswerable)
                unsummary = evaluator.evaluate_retrieval(unanswerable, stubs)
                print("\n--- Unanswerable evaluation ---")
                _print_summary(unsummary, args.breakdown)
                _maybe_export(unsummary, args, suffix="_unanswerable")
            return

        main_dataset = (
            _load_dataset_items(args.dataset)
            if args.dataset is not None
            else _load_dataset_items(_MAIN_DATASET)
        )
        logger.info("Running live retrieval evaluation over %d main goldens", len(main_dataset))
        summary = _evaluate_live_retrieval(evaluator, main_dataset, args)
        _record_retrieval_run(summary, args)
        _print_summary(summary, args.breakdown)
        _maybe_export(summary, args)
        return

    # Fail fast when the OpenRouter key is missing (the judge needs it to run).
    try:
        require_judge_credentials(args.judge_model)
        judge_model = resolve_judge_model(args.judge_model)
    except ValueError as exc:
        logger.error("Judge configuration failed: %s", exc)
        sys.exit(1)

    chunk_texts: dict[str, str] = {}
    if args.chunks.exists():
        chunk_texts = load_chunk_texts(args.chunks)
        logger.info("Loaded %d chunk texts from %s", len(chunk_texts), args.chunks)
    else:
        logger.warning("Chunks file %s not found; retrieval context will be empty.", args.chunks)

    evaluator = VacxisRAGEvaluator(
        judge_model=OpenRouterJudgeLLM(model=judge_model, retry_policy=retry_from_args(args),
                                       timeout=args.judge_request_timeout, schema_retries=args.schema_retries),
        threshold=args.threshold,
        enable_geval=not args.skip_geval,
        enable_ir_metrics=not args.skip_ir,
        ir_k_values=args.ir_k_values,
    )

    if args.results is not None:
        _run_precomputed(evaluator, args, chunk_texts)
        return

    _run_live(evaluator, args, chunk_texts)


def _record_retrieval_run(summary: EvaluationSummary, args: argparse.Namespace) -> None:
    """Keep the scoring inputs and live retriever settings with exported results."""
    run: dict[str, Any] = {
        "mode": "precomputed" if args.results is not None else "live",
        "dataset": str(args.dataset or (args.results if args.results is not None else _MAIN_DATASET)),
        "results": str(args.results) if args.results is not None else None,
    }
    if args.results is None:
        eff_mode = _effective_retriever_mode(args)
        name_map = {"advanced": "advanced_dense", "naive": "bm25"}
        name = name_map.get(eff_mode, eff_mode)
        run.update({
            "retriever": name,
            "retriever_mode": eff_mode,
            "top_k": args.top_k,
            "chunks": str(args.chunks),
        })
        if eff_mode == "advanced":
            run.update({
                "dense_only": True,
                "skip_rerank": True,
                "collection_name": args.collection_name,
                "model_path": args.model_path,
            })
        elif eff_mode == "lightrag":
            run.update({
                "model_path": args.model_path,
                "lightrag_mode": getattr(args, "lightrag_mode", "hybrid"),
                "lightrag_working_dir": str(
                    getattr(args, "lightrag_working_dir", "lightrag_storage")
                ),
                "lightrag_workspace": getattr(args, "lightrag_workspace", "vacxis"),
                "lightrag_top_k": getattr(args, "lightrag_top_k", 20),
                "lightrag_chunk_top_k": getattr(args, "lightrag_chunk_top_k", 5),
            })
        elif eff_mode == "hybrid_lightrag":
            run.update({
                "collection_name": args.collection_name,
                "model_path": args.model_path,
                "reranker_path": args.reranker_path,
                "skip_rerank": args.skip_rerank,
                "merge_table_chunks": bool(getattr(args, "merge_table_chunks", False)),
                "lightrag_mode": getattr(args, "lightrag_mode", "hybrid"),
                "lightrag_fusion": getattr(args, "lightrag_fusion", "context_append"),
                "lightrag_working_dir": str(
                    getattr(args, "lightrag_working_dir", "lightrag_storage")
                ),
                "lightrag_workspace": getattr(args, "lightrag_workspace", "vacxis"),
                "lightrag_top_k": getattr(args, "lightrag_top_k", 20),
                "lightrag_chunk_top_k": getattr(args, "lightrag_chunk_top_k", 5),
            })
        elif eff_mode != "naive":
            run.update({
                "collection_name": args.collection_name,
                "model_path": args.model_path,
                "reranker_path": args.reranker_path,
                "skip_rerank": args.skip_rerank,
                "merge_table_chunks": bool(getattr(args, "merge_table_chunks", False)),
            })
    summary.metadata["run"] = run
    for warning in summary.metadata.get("warnings", []):
        logger.warning("%s", warning)


def _run_live(
    evaluator: VacxisRAGEvaluator,
    args: argparse.Namespace,
    chunk_texts: dict[str, str],
) -> None:
    """Default mode: run the RAG pipeline over the datasets, then score."""
    main_dataset = (
        _load_dataset_items(args.dataset)
        if args.dataset is not None
        else _load_dataset_items(_MAIN_DATASET)
    )
    logger.info("Running live RAG pipeline over %d main goldens", len(main_dataset))
    summary = _evaluate_live(evaluator, main_dataset, args, chunk_texts)
    _print_summary(summary, args.breakdown)
    _maybe_export(summary, args)

    unanswerable = [] if args.skip_unanswerable else _load_flat_dataset(args.unanswerable)
    if unanswerable and not args.skip_unanswerable:
        logger.info("Running live RAG pipeline over %d unanswerable goldens", len(unanswerable))
        unsummary = _evaluate_live(evaluator, unanswerable, args, chunk_texts)
        unsummary.metadata["run"]["dataset"] = str(args.unanswerable)
        print("\n--- Unanswerable evaluation ---")
        _print_summary(unsummary, args.breakdown)
        _maybe_export(unsummary, args, suffix="_unanswerable")


def _run_precomputed(
    evaluator: VacxisRAGEvaluator,
    args: argparse.Namespace,
    chunk_texts: dict[str, str],
) -> None:
    """Fast path: score precomputed generation results without the pipeline."""
    results = json.loads(args.results.read_text(encoding="utf-8"))
    if not isinstance(results, list):
        logger.error("Results file %s must be a JSON list.", args.results)
        sys.exit(1)

    if args.dataset is not None:
        dataset = _load_dataset_items(args.dataset)
    else:
        # No explicit dataset: the results carry the golden fields, so self-align.
        dataset = results

    summary = evaluator.evaluate_results(dataset, results, chunk_texts=chunk_texts,
                                         checkpoint_path=args.checkpoint, resume=args.resume)
    _print_summary(summary, args.breakdown)
    _maybe_export(summary, args)

    unanswerable = [] if args.skip_unanswerable else _load_flat_dataset(args.unanswerable)
    if unanswerable and args.unanswerable != args.results and not args.skip_unanswerable:
        stubs = _unanswerable_stubs(unanswerable)
        unsummary = evaluator.evaluate_results(unanswerable, stubs, chunk_texts=chunk_texts,
            checkpoint_path=(args.checkpoint.with_stem(args.checkpoint.stem + "_unanswerable") if args.checkpoint else None),
            resume=args.resume)
        print("\n--- Unanswerable evaluation ---")
        _print_summary(unsummary, args.breakdown)
        _maybe_export(unsummary, args, suffix="_unanswerable")


def _unanswerable_stubs(flat_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build self-consistency result stubs from the flat unanswerable set.

    Precomputed mode has no pipeline to run, so each item's reference answer and
    relevant chunks stand in for the generated answer; this measures whether the
    dataset's own refusal labels are internally consistent.
    """
    return [
        {
            "QUERY_ID": item.get("QUERY_ID", item.get("query_id", "")),
            "GENERATED_ANSWER": {
                "answer": item.get("REFERENCE_ANSWER", ""),
                "citations": [
                    c.get("CHUNK_ID")
                    for c in (item.get("RELEVANT_CHUNKS") or [])
                    if isinstance(c, dict) and c.get("RELEVANCE_SCORE", 0) > 0
                ],
                "invalid_citations": [],
                "retrieved_chunk_ids": [
                    c.get("CHUNK_ID") for c in (item.get("RELEVANT_CHUNKS") or [])
                    if isinstance(c, dict)
                ],
                "is_refusal": item.get("ANSWERABILITY", "") == "UNANSWERABLE",
                "latency_seconds": 0.0,
                "usage": {},
            },
        }
        for item in flat_items
    ]


def _maybe_export(summary: EvaluationSummary, args: argparse.Namespace, suffix: str = "") -> None:
    """Write JSON / Markdown / Parquet outputs when requested."""
    if args.out_json:
        path = args.out_json.with_stem(args.out_json.stem + suffix) if suffix else args.out_json
        VacxisRAGEvaluator.save_json(summary, path)
    if args.out_md:
        path = args.out_md.with_stem(args.out_md.stem + suffix) if suffix else args.out_md
        VacxisRAGEvaluator.save_markdown(summary, path)
    if args.out_parquet:
        path = args.out_parquet.with_stem(args.out_parquet.stem + suffix) if suffix else args.out_parquet
        _save_parquet(summary, path)


if __name__ == "__main__":
    main()
