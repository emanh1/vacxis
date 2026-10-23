"""Orchestrator classes for evaluating retrieval, generation, and end-to-end VacxiS RAG."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from vacxis_capstone_fa26.evaluation.generation import evaluate_generation_query
from vacxis_capstone_fa26.evaluation.retrieval import evaluate_retrieval_query
from vacxis_capstone_fa26.evaluation.schemas import (
    EvaluationSummary,
    MetricAggregate,
    QueryGenerationResult,
    QueryRetrievalResult,
)

logger = logging.getLogger("vacxis.evaluation.evaluator")


def _get_key(d: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Helper to extract value trying multiple key case variations."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def compute_metric_aggregates(
    metric_records: Sequence[dict[str, float]],
) -> tuple[dict[str, float], dict[str, MetricAggregate]]:
    """Compute overall means and detailed statistical aggregates for each metric."""
    if not metric_records:
        return {}, {}

    metric_names = set().union(*(r.keys() for r in metric_records))
    overall: dict[str, float] = {}
    aggregates: dict[str, MetricAggregate] = {}

    for name in metric_names:
        values = [r[name] for r in metric_records if name in r and not math.isnan(r[name])]
        n = len(values)
        if n == 0:
            continue
        mean_val = sum(values) / float(n)
        variance = sum((x - mean_val) ** 2 for x in values) / float(n) if n > 1 else 0.0
        std_val = math.sqrt(variance)
        min_val = min(values)
        max_val = max(values)

        overall[name] = mean_val
        aggregates[name] = MetricAggregate(
            mean=mean_val,
            std=std_val,
            min=min_val,
            max=max_val,
            count=n,
        )

    return overall, aggregates


def group_metrics_by_dimension(
    queries: Sequence[dict[str, Any]],
    per_query_metrics: Sequence[dict[str, float]],
    dimension_key: str,
) -> dict[str, dict[str, float]]:
    """Aggregate metric scores grouped by metadata dimension (e.g. QUERY_TYPE)."""
    groups: dict[str, list[dict[str, float]]] = {}
    for q, m in zip(queries, per_query_metrics):
        group_val = str(_get_key(q, dimension_key, dimension_key.lower(), default="UNKNOWN"))
        groups.setdefault(group_val, []).append(m)

    breakdown: dict[str, dict[str, float]] = {}
    for group_name, records in groups.items():
        mean_scores, _ = compute_metric_aggregates(records)
        mean_scores["count"] = float(len(records))
        breakdown[group_name] = mean_scores

    return breakdown


class RetrievalEvaluator:
    """Evaluates retrieval quality against ground-truth relevant chunks and documents."""

    def __init__(self, k_values: Sequence[int] = (1, 3, 5, 10)) -> None:
        self.k_values = list(k_values)

    def evaluate(
        self,
        queries: Sequence[dict[str, Any]],
        retrieved_per_query: Sequence[Sequence[Any]],
    ) -> EvaluationSummary:
        """Run retrieval evaluation across all queries."""
        per_query_results: list[QueryRetrievalResult] = []
        metrics_list: list[dict[str, float]] = []

        for q, retrieved in zip(queries, retrieved_per_query):
            qid = str(_get_key(q, "QUERY_ID", "query_id", default=""))
            qtext = str(_get_key(q, "QUERY", "query", default=""))
            rel_chunks = _get_key(q, "RELEVANT_CHUNKS", "relevant_chunks", default=[])
            rel_docs = _get_key(q, "RELEVANT_DOCS", "relevant_docs", default=[])

            metadata = {
                "query_type": _get_key(q, "QUERY_TYPE", "query_type", default=""),
                "difficulty": _get_key(q, "QUERY_DIFFICULTY", "query_difficulty", default=""),
                "complexity": _get_key(q, "QUERY_COMPLEXITY", "query_complexity", default=""),
                "answerability": _get_key(q, "ANSWERABILITY", "answerability", default=""),
            }

            res = evaluate_retrieval_query(
                retrieved=retrieved,
                relevant_chunks=rel_chunks,
                relevant_docs=rel_docs,
                k_values=self.k_values,
                query_id=qid,
                query_text=qtext,
                metadata=metadata,
            )
            per_query_results.append(res)
            metrics_list.append(res.metrics)

        overall, aggregates = compute_metric_aggregates(metrics_list)
        by_type = group_metrics_by_dimension(queries, metrics_list, "QUERY_TYPE")
        by_diff = group_metrics_by_dimension(queries, metrics_list, "QUERY_DIFFICULTY")
        by_comp = group_metrics_by_dimension(queries, metrics_list, "QUERY_COMPLEXITY")
        by_ans = group_metrics_by_dimension(queries, metrics_list, "ANSWERABILITY")

        return EvaluationSummary(
            task="retrieval",
            num_queries=len(queries),
            overall_metrics=overall,
            metric_aggregates=aggregates,
            by_query_type=by_type,
            by_difficulty=by_diff,
            by_complexity=by_comp,
            by_answerability=by_ans,
            per_query_results=[r.to_dict() for r in per_query_results],
        )


class GenerationEvaluator:
    """Evaluates answer generation quality (citations, text similarity, refusal)."""

    def evaluate(
        self,
        queries: Sequence[dict[str, Any]],
        answers: Sequence[Any],
    ) -> EvaluationSummary:
        """Run generation evaluation across all queries."""
        per_query_results: list[QueryGenerationResult] = []
        metrics_list: list[dict[str, float]] = []

        for q, ans in zip(queries, answers):
            qid = str(_get_key(q, "QUERY_ID", "query_id", default=""))
            qtext = str(_get_key(q, "QUERY", "query", default=""))
            gold_ref = str(_get_key(q, "REFERENCE_ANSWER", "reference_answer", default=""))
            rel_chunks = _get_key(q, "RELEVANT_CHUNKS", "relevant_chunks", default=[])
            expected_ans = str(_get_key(q, "ANSWERABILITY", "answerability", default="ANSWERABLE"))

            # Extract fields from GeneratedAnswer or dict
            if isinstance(ans, dict):
                # May be wrapped in GENERATED_ANSWER
                gen_dict = ans.get("GENERATED_ANSWER", ans)
                pred_text = gen_dict.get("answer", "")
                citations = gen_dict.get("citations", [])
                invalid_cits = gen_dict.get("invalid_citations", [])
                ret_chunk_ids = (
                    gen_dict.get("retrieved_chunk_ids")
                    or [_get_key(c, "chunk_id", "CHUNK_ID") for c in gen_dict.get("retrieved_chunks", [])]
                )
                is_refusal = bool(gen_dict.get("is_refusal", False))
                latency = float(gen_dict.get("latency_seconds", 0.0))
                usage = gen_dict.get("usage", {})
            else:
                pred_text = getattr(ans, "answer", "")
                citations = getattr(ans, "citations", [])
                invalid_cits = getattr(ans, "invalid_citations", [])
                ret_chunks = getattr(ans, "retrieved_chunks", [])
                ret_chunk_ids = [
                    getattr(c, "chunk_id", c.get("chunk_id", "") if isinstance(c, dict) else str(c))
                    for c in ret_chunks
                ]
                is_refusal = bool(getattr(ans, "is_refusal", False))
                latency = float(getattr(ans, "latency_seconds", 0.0))
                usage = getattr(ans, "usage", {})

            metadata = {
                "query_type": _get_key(q, "QUERY_TYPE", "query_type", default=""),
                "difficulty": _get_key(q, "QUERY_DIFFICULTY", "query_difficulty", default=""),
                "complexity": _get_key(q, "QUERY_COMPLEXITY", "query_complexity", default=""),
                "answerability": expected_ans,
            }

            res = evaluate_generation_query(
                prediction=pred_text,
                reference=gold_ref,
                citations=citations,
                invalid_citations=invalid_cits,
                relevant_chunks=rel_chunks,
                retrieved_chunk_ids=ret_chunk_ids,
                is_refusal=is_refusal,
                expected_answerability=expected_ans,
                query_id=qid,
                query_text=qtext,
                latency_seconds=latency,
                token_usage=usage,
                metadata=metadata,
            )
            per_query_results.append(res)
            metrics_list.append(res.metrics)

        overall, aggregates = compute_metric_aggregates(metrics_list)
        by_type = group_metrics_by_dimension(queries, metrics_list, "QUERY_TYPE")
        by_diff = group_metrics_by_dimension(queries, metrics_list, "QUERY_DIFFICULTY")
        by_comp = group_metrics_by_dimension(queries, metrics_list, "QUERY_COMPLEXITY")
        by_ans = group_metrics_by_dimension(queries, metrics_list, "ANSWERABILITY")

        return EvaluationSummary(
            task="generation",
            num_queries=len(queries),
            overall_metrics=overall,
            metric_aggregates=aggregates,
            by_query_type=by_type,
            by_difficulty=by_diff,
            by_complexity=by_comp,
            by_answerability=by_ans,
            per_query_results=[r.to_dict() for r in per_query_results],
        )


class VacxisRAGEvaluator:
    """Unified Evaluator for Retrieval, Generation, and End-to-End RAG."""

    def __init__(self, k_values: Sequence[int] = (1, 3, 5, 10)) -> None:
        self.k_values = list(k_values)
        self.retrieval_evaluator = RetrievalEvaluator(k_values=self.k_values)
        self.generation_evaluator = GenerationEvaluator()

    def evaluate_results(
        self,
        dataset_items: Sequence[dict[str, Any]],
        generation_results: Sequence[dict[str, Any]],
    ) -> EvaluationSummary:
        """Evaluate precomputed generation results against golden dataset."""
        # Align dataset items with results by QUERY_ID
        items_by_id = {
            str(_get_key(item, "QUERY_ID", "query_id")): item for item in dataset_items
        }

        matched_queries: list[dict[str, Any]] = []
        matched_answers: list[dict[str, Any]] = []
        retrieved_lists: list[list[Any]] = []

        for res in generation_results:
            qid = str(_get_key(res, "QUERY_ID", "query_id"))
            query_item = items_by_id.get(qid, res)
            matched_queries.append(query_item)
            matched_answers.append(res)

            gen_info = res.get("GENERATED_ANSWER", res)
            ret_chunks = (
                gen_info.get("retrieved_chunks")
                or gen_info.get("retrieved_chunk_ids")
                or []
            )
            retrieved_lists.append(ret_chunks)

        gen_summary = self.generation_evaluator.evaluate(matched_queries, matched_answers)

        # If retrieved chunks are present, also evaluate retrieval
        has_retrieval = any(bool(r) for r in retrieved_lists)
        if has_retrieval:
            ret_summary = self.retrieval_evaluator.evaluate(matched_queries, retrieved_lists)
            # Combine metrics
            combined_overall = {**ret_summary.overall_metrics, **gen_summary.overall_metrics}
            combined_aggregates = {
                **ret_summary.metric_aggregates,
                **gen_summary.metric_aggregates,
            }

            combined_by_type: dict[str, dict[str, float]] = {}
            for qtype in set(gen_summary.by_query_type) | set(ret_summary.by_query_type):
                combined_by_type[qtype] = {
                    **ret_summary.by_query_type.get(qtype, {}),
                    **gen_summary.by_query_type.get(qtype, {}),
                }

            combined_by_diff: dict[str, dict[str, float]] = {}
            for diff in set(gen_summary.by_difficulty) | set(ret_summary.by_difficulty):
                combined_by_diff[diff] = {
                    **ret_summary.by_difficulty.get(diff, {}),
                    **gen_summary.by_difficulty.get(diff, {}),
                }

            combined_by_comp: dict[str, dict[str, float]] = {}
            for comp in set(gen_summary.by_complexity) | set(ret_summary.by_complexity):
                combined_by_comp[comp] = {
                    **ret_summary.by_complexity.get(comp, {}),
                    **gen_summary.by_complexity.get(comp, {}),
                }

            combined_by_ans: dict[str, dict[str, float]] = {}
            for ans in set(gen_summary.by_answerability) | set(ret_summary.by_answerability):
                combined_by_ans[ans] = {
                    **ret_summary.by_answerability.get(ans, {}),
                    **gen_summary.by_answerability.get(ans, {}),
                }

            # Merge per-query results
            combined_per_query: list[dict[str, Any]] = []
            for g_res, r_res in zip(gen_summary.per_query_results, ret_summary.per_query_results):
                merged = {**g_res}
                merged["retrieval_metrics"] = r_res.get("metrics", {})
                merged["retrieved_chunk_ids"] = r_res.get("retrieved_chunk_ids", [])
                combined_per_query.append(merged)

            return EvaluationSummary(
                task="rag",
                num_queries=len(matched_queries),
                overall_metrics=combined_overall,
                metric_aggregates=combined_aggregates,
                by_query_type=combined_by_type,
                by_difficulty=combined_by_diff,
                by_complexity=combined_by_comp,
                by_answerability=combined_by_ans,
                per_query_results=combined_per_query,
            )

        return gen_summary

    @staticmethod
    def save_json(summary: EvaluationSummary, path: str | Path) -> None:
        """Save evaluation summary as JSON."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Saved evaluation JSON to %s", target_path)

    @staticmethod
    def save_markdown(summary: EvaluationSummary, path: str | Path) -> None:
        """Save evaluation summary as Markdown."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(summary.render_markdown(), encoding="utf-8")
        logger.info("Saved evaluation Markdown to %s", target_path)

    @staticmethod
    def save_parquet(summary: EvaluationSummary, path: str | Path) -> None:
        """Export per-query evaluation results as a Parquet file using PyArrow."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        for item in summary.per_query_results:
            row: dict[str, Any] = {
                "query_id": str(item.get("query_id", "")),
                "query": str(item.get("query", "")),
                "task": summary.task,
            }
            # Add metadata columns
            meta = item.get("metadata", {})
            for mk, mv in meta.items():
                row[f"meta_{mk}"] = str(mv)

            # Flatten metrics
            metrics = item.get("metrics", {})
            for mk, mv in metrics.items():
                row[f"metric_{mk}"] = float(mv) if mv is not None else 0.0

            if "retrieval_metrics" in item:
                for rk, rv in item["retrieval_metrics"].items():
                    row[f"retrieval_{rk}"] = float(rv) if rv is not None else 0.0

            rows.append(row)

        if not rows:
            logger.warning("No per-query rows to write to parquet.")
            return

        # Build PyArrow Table natively
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, target_path)
        logger.info("Saved %d evaluation rows to Parquet %s", len(rows), target_path)
