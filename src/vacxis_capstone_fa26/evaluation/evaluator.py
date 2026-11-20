"""DeepEval-powered evaluation orchestrator for the VacxiS RAG pipeline.

The module scores golden questions with DeepEval's RAG metric family:

* generator-side: ``AnswerRelevancyMetric``, ``FaithfulnessMetric``
* retriever-side: ``ContextualRelevancyMetric``, ``ContextualPrecisionMetric``,
  ``ContextualRecallMetric``
* a GEval ``Correctness`` judge comparing the answer with the reference
* a deterministic refusal-correctness check against ANSWERABILITY

All judges run through a configurable judge model (``judge.OpenRouterJudgeLLM``
by default), so no OpenAI key is required. Evaluation can consume precomputed
generation results (from the generation CLI) or run a live RAG pipeline via an
injected answer callback.
"""
from __future__ import annotations

import json
import math
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from vacxis_capstone_fa26.evaluation import datasets
from vacxis_capstone_fa26.evaluation.datasets import GoldenRecord
from vacxis_capstone_fa26.evaluation.ir_metrics import (
    DEFAULT_K_VALUES,
    compute_query_ir_metrics,
)
from vacxis_capstone_fa26.evaluation.judge import OpenRouterJudgeLLM
from vacxis_capstone_fa26.evaluation.schemas import (
    EvaluationSummary,
    MetricAggregate,
    QueryEvalResult,
)

logger = logging.getLogger("vacxis.evaluation.evaluator")

CORRECTNESS_CRITERIA = (
    "Judge the medical correctness of 'actual_output' relative to "
    "'expected_output'. Score 1.0 when the answer is medically accurate, fully "
    "consistent with the reference, and covers the required points. Score lower "
    "for missing facts, contradictions, or unsafe additions; a correct refusal "
    "to answer an unanswerable question scores 1.0."
)

REFUSAL_CORRECTNESS = "refusal_correctness"
_NON_DOCUMENT_CHUNK_IDS = frozenset({"lightrag_kg_context"})


def _filter_document_chunk_ids(chunk_ids: Sequence[str]) -> list[str]:
    """Exclude synthetic knowledge-graph context IDs from chunk-level IR scoring."""
    return [
        str(cid).strip()
        for cid in chunk_ids
        if str(cid).strip() and str(cid).strip() not in _NON_DOCUMENT_CHUNK_IDS
    ]


def _resolve_query_id(item: dict[str, Any]) -> str | None:
    """Find the query ID across the two dataset shapes (flat and DeepEval)."""
    raw = item.get("QUERY_ID", item.get("query_id"))
    if raw is not None and str(raw).strip():
        return str(raw)
    metadata = item.get("additional_metadata")
    if isinstance(metadata, dict):
        raw = metadata.get("query_id")
        if raw is not None and str(raw).strip():
            return str(raw)
    return None


def validate_query_ids(items: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Reject missing and duplicate IDs before joining benchmark records."""
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Query records must be objects.")
        qid = _resolve_query_id(item)
        if qid is None:
            raise ValueError("Every record must have a nonblank query ID.")
        if qid in by_id:
            raise ValueError(f"Duplicate QUERY_ID '{qid}' in evaluation records.")
        by_id[qid] = item
    return by_id


def align_results(
    dataset_items: Sequence[dict[str, Any]],
    generation_results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join complete result sets to the dataset while preserving result order."""
    items = validate_query_ids(dataset_items)
    results = validate_query_ids(generation_results)
    if items.keys() != results.keys():
        raise ValueError(
            f"Dataset and results disagree on QUERY_IDs: "
            f"{len(items)} dataset items vs {len(results)} results."
        )
    return [items[qid] for qid in results]


class _CaseInfo:
    """Per-query answer fields kept alongside a golden for summary building."""

    def __init__(
        self,
        actual_output: str,
        is_refusal: bool,
        retrieval_context: list[str],
        retrieved_chunk_ids: list[str] | None = None,
        retrieved_chunk_aliases: dict[str, list[str]] | None = None,
    ) -> None:
        self.actual_output = actual_output
        self.is_refusal = is_refusal
        self.retrieval_context = retrieval_context
        self.retrieved_chunk_ids = retrieved_chunk_ids or []
        self.retrieved_chunk_aliases = retrieved_chunk_aliases or {}


class VacxisRAGEvaluator:
    """Unified DeepEval evaluator for retrieval and generation quality."""

    def __init__(
        self,
        judge_model: str | DeepEvalBaseLLM | None = None,
        threshold: float = 0.5,
        *,
        enable_geval: bool = True,
        enable_ir_metrics: bool = True,
        ir_k_values: Sequence[int] = DEFAULT_K_VALUES,
    ) -> None:
        # Judge resolution is deferred so constructing an evaluator never raises
        # on missing credentials; it only matters once LLM-judged metrics run.
        self._judge_arg: str | DeepEvalBaseLLM | None = judge_model
        self._judge: DeepEvalBaseLLM | None = None
        self.threshold = threshold
        self.enable_geval = enable_geval
        self.enable_ir_metrics = enable_ir_metrics
        self.ir_k_values = tuple(ir_k_values)

    @property
    def judge(self) -> DeepEvalBaseLLM:
        """Lazily build the judge LLM on first access."""
        if self._judge is None:
            self._judge = (
                self._judge_arg
                if isinstance(self._judge_arg, DeepEvalBaseLLM)
                else OpenRouterJudgeLLM(model=self._judge_arg)
            )
        return self._judge

    # ------------------------------------------------------------------
    # Metric and test-case construction
    # ------------------------------------------------------------------

    def build_metrics(self) -> list:
        """Instantiate the configured DeepEval metric set on the judge model."""
        model = self.judge
        metrics = [
            AnswerRelevancyMetric(model=model, threshold=self.threshold),
            FaithfulnessMetric(model=model, threshold=self.threshold),
            ContextualRelevancyMetric(model=model, threshold=self.threshold),
            ContextualPrecisionMetric(model=model, threshold=self.threshold),
            ContextualRecallMetric(model=model, threshold=self.threshold),
        ]
        if self.enable_geval:
            metrics.append(
                GEval(
                    name="Correctness",
                    criteria=CORRECTNESS_CRITERIA,
                    evaluation_params=[
                        SingleTurnParams.ACTUAL_OUTPUT,
                        SingleTurnParams.EXPECTED_OUTPUT,
                    ],
                    threshold=self.threshold,
                    model=model,
                )
            )
        return metrics

    def _metric_names(self) -> list[str]:
        """List of metric display names (DeepEval ``__name__``) plus refusal flag."""
        names = [metric.__name__ for metric in self.build_metrics()]
        return [*names, REFUSAL_CORRECTNESS]

    def _case_info(self, fields: dict[str, Any], chunk_texts: dict[str, str]) -> _CaseInfo:
        """Derive answer, refusal flag, and hydrated context for one result."""
        answer_payload: dict[str, Any] = {
            "retrieved_chunk_ids": fields.get("retrieved_chunk_ids", []),
        }
        if "retrieved_contexts" in fields:
            answer_payload["retrieved_contexts"] = fields["retrieved_contexts"]
        return _CaseInfo(
            actual_output=fields["actual_output"],
            is_refusal=fields["is_refusal"],
            retrieval_context=datasets.retrieval_context_from_answer(
                answer_payload, chunk_texts
            ),
            retrieved_chunk_ids=fields.get("retrieved_chunk_ids", []),
            retrieved_chunk_aliases=fields.get("retrieved_chunk_aliases"),
        )

    def _build_test_case(
        self,
        record: GoldenRecord,
        info: _CaseInfo,
    ) -> LLMTestCase:
        """Create one DeepEval test case from a golden and its answer fields."""
        return LLMTestCase(
            input=record.query,
            actual_output=info.actual_output,
            expected_output=record.expected_output,
            retrieval_context=info.retrieval_context,
            metadata={
                "query_id": record.query_id,
                "answerability": record.answerability,
                "complexity": record.complexity,
                "query_style": record.query_style,
                "difficulty": record.difficulty,
                "query_type": record.query_type,
            },
        )

    # ------------------------------------------------------------------
    # Evaluation entrypoints
    # ------------------------------------------------------------------

    def evaluate_results(
        self,
        dataset_items: Sequence[dict[str, Any]],
        generation_results: Sequence[dict[str, Any]],
        chunk_texts: dict[str, str] | None = None,
        *,
        max_concurrent: int = 1,
        write_cache: bool = True,
        use_cache: bool = True,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> EvaluationSummary:
        """Evaluate precomputed results against the golden dataset with DeepEval.

        ``generation_results`` is the shape the generation CLI writes: a list of
        dataset items each carrying a ``GENERATED_ANSWER`` dict (or the dict
        itself). ``chunk_texts`` hydrates retrieved chunk ids into context text
        so DeepEval's retrieval metrics have real material to judge.
        """
        aligned = align_results(dataset_items, generation_results)
        goldens = [datasets.to_record(item) for item in aligned]
        chunk_texts = chunk_texts or {}
        case_infos: list[_CaseInfo] = []
        for result in generation_results:
            fields = datasets.answer_fields(result)
            case_infos.append(self._case_info(fields, chunk_texts))

        test_cases = [
            self._build_test_case(record, info) for record, info in zip(goldens, case_infos)
        ]
        if checkpoint_path is None:
            eval_result = self._run_deepeval(
                test_cases, max_concurrent=max_concurrent,
                write_cache=write_cache, use_cache=use_cache,
            )
            return self.build_summary(goldens, eval_result, case_infos)

        from vacxis_capstone_fa26.resilience import atomic_json, fingerprint
        signature = fingerprint({
            "dataset": list(dataset_items), "results": list(generation_results),
            "contexts": [info.retrieval_context for info in case_infos],
            "judge": self.judge.get_model_name(), "threshold": self.threshold,
            "geval": self.enable_geval, "ir": self.enable_ir_metrics, "k": self.ir_k_values,
        })
        path = Path(checkpoint_path)
        rows = []
        if resume and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("fingerprint") != signature:
                raise ValueError("Evaluation checkpoint inputs or settings changed; use a new checkpoint.")
            rows = saved["report"]["per_query_results"]
            ids = [row["query_id"] for row in rows]
            expected = [record.query_id for record in goldens]
            if ids != expected[:len(ids)] or len(ids) > len(expected):
                raise ValueError("Evaluation checkpoint query order is invalid.")
        names = self._metric_names()
        def report():
            queries = [QueryEvalResult(**row) for row in rows]
            all_names = list(dict.fromkeys(names + [name for row in rows for name in row["metric_scores"]]))
            aggregates = _aggregate_stats(queries, all_names)
            return EvaluationSummary(
                num_queries=len(rows), metric_names=all_names,
                metric_aggregates=aggregates,
                overall_metrics={name: value.mean for name, value in aggregates.items()},
                per_query_results=rows.copy(),
                by_answerability=_group_breakdown(queries, "answerability", all_names),
                by_complexity=_group_breakdown(queries, "complexity", all_names),
                by_query_style=_group_breakdown(queries, "query_style", all_names),
                by_difficulty=_group_breakdown(queries, "difficulty", all_names),
                metadata={"complete": len(rows) == len(goldens), "total_queries": len(goldens)},
            )
        atomic_json(path, {"fingerprint": signature, "report": report().to_dict()})
        for index in range(len(rows), len(goldens)):
            result = self._run_deepeval([test_cases[index]], max_concurrent=1,
                                      write_cache=write_cache, use_cache=use_cache)
            summary = self.build_summary([goldens[index]], result, [case_infos[index]])
            row = summary.per_query_results[0]
            expected_metrics = {metric.__name__ for metric in self.build_metrics()}
            if not expected_metrics.issubset(row["metric_results"]) or any(
                value.get("error") or value.get("score") is None or not math.isfinite(value["score"])
                for value in row["metric_results"].values()
            ):
                raise ValueError("Incomplete metric results cannot be checkpointed as complete.")
            rows.append(row)
            atomic_json(path, {"fingerprint": signature, "report": report().to_dict()})
        return report()

    def evaluate_live(
        self,
        dataset_items: Sequence[dict[str, Any]],
        answer_fn: Callable[[str, dict[str, Any]], dict[str, Any]],
        chunk_texts: dict[str, str] | None = None,
        *,
        max_concurrent: int = 1,
        write_cache: bool = True,
        use_cache: bool = True,
    ) -> EvaluationSummary:
        """Run a live RAG pipeline per golden, then evaluate with DeepEval.

        ``answer_fn(query, golden_dict)`` returns a dict shaped like the
        generation CLI's ``GENERATED_ANSWER`` (``answer``,
        ``retrieved_chunk_ids``, ``is_refusal``). This keeps the evaluator free
        of retrieval dependencies; the caller injects its pipeline.
        """
        goldens = [datasets.to_record(item) for item in dataset_items]
        chunk_texts = chunk_texts or {}
        case_infos: list[_CaseInfo] = []
        test_cases: list[LLMTestCase] = []
        for record in goldens:
            logger.info("Answering live %s", record.query_id)
            answer = answer_fn(record.query, record.to_dict())
            fields = datasets.answer_fields(answer)
            info = self._case_info(fields, chunk_texts)
            case_infos.append(info)
            test_cases.append(self._build_test_case(record, info))

        eval_result = self._run_deepeval(
            test_cases,
            max_concurrent=max_concurrent,
            write_cache=write_cache,
            use_cache=use_cache,
        )
        return self.build_summary(goldens, eval_result, case_infos)

    def evaluate_retrieval(
        self,
        dataset_items: Sequence[dict[str, Any]],
        generation_results: Sequence[dict[str, Any]],
        *,
        k_values: Sequence[int] | None = None,
    ) -> EvaluationSummary:
        """Evaluate deterministic IR metrics without invoking any LLM judge."""
        k_vals = tuple(k_values) if k_values is not None else self.ir_k_values
        if not k_vals or any(type(k) is not int or k < 1 for k in k_vals):
            raise ValueError("IR cutoffs must be positive integers.")
        k_vals = tuple(sorted(set(k_vals)))
        aligned = align_results(dataset_items, generation_results)
        goldens = [datasets.to_record(item) for item in aligned]
        case_infos: list[_CaseInfo] = []
        for result in generation_results:
            fields = datasets.answer_fields(result)
            case_infos.append(self._case_info(fields, {}))

        ir_metric_names = [
            f"{m}@{k}"
            for k in k_vals
            for m in ("Hit", "Recall", "Precision", "MRR", "NDCG", "MAP")
        ]
        per_query: list[QueryEvalResult] = []
        for record, info in zip(goldens, case_infos):
            scores: dict[str, float] = {}
            details: dict[str, dict[str, Any]] = {}
            if record.relevant_chunk_ids:
                ir_scores = compute_query_ir_metrics(
                    _filter_document_chunk_ids(info.retrieved_chunk_ids),
                    record.relevant_chunk_ids,
                    k_values=k_vals,
                    chunk_aliases=info.retrieved_chunk_aliases or None,
                )
                scores.update(ir_scores)
                for ir_name, ir_score in ir_scores.items():
                    details[ir_name] = {
                        "score": ir_score,
                        "success": ir_score >= self.threshold,
                        "reason": f"Deterministic IR metric {ir_name}",
                        "error": None,
                    }
            query_meta: dict[str, Any] = {
                "answerability": record.answerability,
                "complexity": record.complexity,
                "query_style": record.query_style,
                "difficulty": record.difficulty,
                "query_type": record.query_type,
                "retrieved_chunk_ids": info.retrieved_chunk_ids,
                "relevant_chunk_ids": record.relevant_chunk_ids,
                "ir_scored": bool(record.relevant_chunk_ids),
            }
            if info.retrieved_chunk_aliases:
                query_meta["retrieved_chunk_aliases"] = info.retrieved_chunk_aliases
            per_query.append(
                QueryEvalResult(
                    query_id=record.query_id,
                    query=record.query,
                    actual_output=info.actual_output,
                    expected_output=record.expected_output,
                    retrieval_context=info.retrieval_context,
                    is_refusal=info.is_refusal,
                    answerability=record.answerability,
                    metric_scores=scores,
                    metric_results=details,
                    metadata=query_meta,
                )
            )

        aggregates = _aggregate_stats(per_query, ir_metric_names)
        summary = EvaluationSummary(
            task="retrieval",
            num_queries=len(per_query),
            metric_names=ir_metric_names,
            overall_metrics={name: agg.mean for name, agg in aggregates.items()},
            metric_aggregates=aggregates,
            per_query_results=[q.to_dict() for q in per_query],
            metadata={"threshold": self.threshold, "k_values": list(k_vals)},
        )
        scored_count = sum(bool(record.relevant_chunk_ids) for record in goldens)
        ranking_lengths = [
            len(set(_filter_document_chunk_ids(info.retrieved_chunk_ids)))
            for info in case_infos
        ]
        short_counts = {
            str(k): sum(length < k for length in ranking_lengths)
            for k in k_vals
        }
        warnings = []
        if scored_count < len(goldens):
            warnings.append(f"{len(goldens) - scored_count} queries have no relevance labels and are excluded from IR averages.")
        if any(short_counts.values()):
            warnings.append("Some ranked lists are shorter than the metric cutoff; missing ranks count as nonrelevant. Check retrieval depth before comparing runs.")
        used_aliases = any(bool(info.retrieved_chunk_aliases) for info in case_infos)
        relevance_note = (
            "Binary relevance against labeled chunk IDs with rank-preserving merged_chunk_ids alias matching; unlabeled alternatives count as nonrelevant."
            if used_aliases
            else "Binary relevance against labeled chunk IDs; unlabeled alternatives count as nonrelevant."
        )
        summary.metadata.update({
            "scored_queries": scored_count,
            "unscored_queries": len(goldens) - scored_count,
            "short_rankings_by_cutoff": short_counts,
            "relevance": relevance_note,
            "warnings": warnings,
        })
        summary.by_answerability = _group_breakdown(per_query, "answerability", ir_metric_names)
        summary.by_complexity = _group_breakdown(per_query, "complexity", ir_metric_names)
        summary.by_query_style = _group_breakdown(per_query, "query_style", ir_metric_names)
        summary.by_difficulty = _group_breakdown(per_query, "difficulty", ir_metric_names)
        return summary

    def _run_deepeval(
        self,
        test_cases: list[LLMTestCase],
        *,
        max_concurrent: int,
        write_cache: bool = True,
        use_cache: bool = True,
    ) -> Any:
        """Run DeepEval over the test cases with quiet, non-interactive settings.

        The cache round-trip lets a stopped or crashed run resume: already scored
        test cases are reused by content key, and only the rest are re-judged.
        """
        if not test_cases:
            raise ValueError("No test cases to evaluate.")
        return deepeval_evaluate(
            test_cases=test_cases,
            metrics=self.build_metrics(),
            _skip_reset=True,
            display_config=DisplayConfig(
                show_indicator=True,
                print_results=True,
                inspect_after_run=True,
            ),
            cache_config=CacheConfig(write_cache=write_cache, use_cache=use_cache),
            async_config=AsyncConfig(run_async=True, max_concurrent=max_concurrent),
            error_config=ErrorConfig(ignore_errors=False, skip_on_missing_params=False),
        )

    # ------------------------------------------------------------------
    # Summary construction
    # ------------------------------------------------------------------

    def build_summary(
        self,
        goldens: list[GoldenRecord],
        eval_result: Any,
        case_infos: Sequence[_CaseInfo],
    ) -> EvaluationSummary:
        """Fold DeepEval's per-query metric data into an EvaluationSummary.

        Appends a deterministic refusal-correctness indicator for goldens that
        carry an ANSWERABILITY label, then aggregates per-metric statistics and
        breakdowns by stratification axis.
        """
        if len(goldens) != len(case_infos):
            raise ValueError("Goldens and case infos must line up one-to-one.")
        # Async execution can complete cases out of order; re-sort by the
        # input index each TestResult carries so goldens stay aligned with
        # their metric data.
        test_results = sorted(
            eval_result.test_results,
            key=lambda result: (result.index is None, result.index),
        )
        by_case: list[list[Any]] = [list(m.metrics_data or []) for m in test_results]
        if len(by_case) != len(goldens):
            raise ValueError(
                f"DeepEval returned {len(by_case)} results for {len(goldens)} goldens."
            )

        metric_names = self._metric_names()
        ir_metric_names = [
            f"{m}@{k}"
            for k in self.ir_k_values
            for m in ("Hit", "Recall", "Precision", "MRR", "NDCG", "MAP")
        ]
        has_ir_metrics = False
        per_query: list[QueryEvalResult] = []
        for record, metrics_data, info in zip(goldens, by_case, case_infos):
            scores: dict[str, float] = {}
            details: dict[str, dict[str, Any]] = {}
            for metric in metrics_data:
                details[metric.name] = {
                    "score": metric.score,
                    "success": metric.success,
                    "reason": metric.reason,
                    "error": metric.error,
                }
                if metric.score is not None:
                    scores[metric.name] = float(metric.score)

            if record.answerability:
                expected_unanswerable = record.answerability.upper() == "UNANSWERABLE"
                refusal_correctness = 1.0 if (info.is_refusal == expected_unanswerable) else 0.0
                scores[REFUSAL_CORRECTNESS] = refusal_correctness
                details[REFUSAL_CORRECTNESS] = {
                    "score": refusal_correctness,
                    "success": refusal_correctness >= self.threshold,
                    "reason": (
                        "refusal matches UNANSWERABLE ground truth"
                        if refusal_correctness == 1.0
                        else "refusal flag disagrees with ANSWERABILITY"
                    ),
                    "error": None,
                }

            if self.enable_ir_metrics and record.relevant_chunk_ids:
                has_ir_metrics = True
                ir_scores = compute_query_ir_metrics(
                    _filter_document_chunk_ids(info.retrieved_chunk_ids),
                    record.relevant_chunk_ids,
                    k_values=self.ir_k_values,
                    chunk_aliases=info.retrieved_chunk_aliases or None,
                )
                scores.update(ir_scores)
                for ir_name, ir_score in ir_scores.items():
                    details[ir_name] = {
                        "score": ir_score,
                        "success": ir_score >= self.threshold,
                        "reason": f"Deterministic IR metric {ir_name}",
                        "error": None,
                    }

            per_query.append(
                QueryEvalResult(
                    query_id=record.query_id,
                    query=record.query,
                    actual_output=info.actual_output,
                    expected_output=record.expected_output,
                    retrieval_context=info.retrieval_context,
                    is_refusal=info.is_refusal,
                    answerability=record.answerability,
                    metric_scores=scores,
                    metric_results=details,
                    metadata={
                        "answerability": record.answerability,
                        "complexity": record.complexity,
                        "query_style": record.query_style,
                        "difficulty": record.difficulty,
                        "query_type": record.query_type,
                    },
                )
            )

        all_metric_names = [*metric_names, *(ir_metric_names if has_ir_metrics else [])]
        aggregates = _aggregate_stats(per_query, all_metric_names)
        summary = EvaluationSummary(
            task="rag",
            num_queries=len(per_query),
            metric_names=all_metric_names,
            overall_metrics={name: agg.mean for name, agg in aggregates.items()},
            metric_aggregates=aggregates,
            per_query_results=[q.to_dict() for q in per_query],
            metadata={"threshold": self.threshold},
        )
        summary.by_answerability = _group_breakdown(per_query, "answerability", all_metric_names)
        summary.by_complexity = _group_breakdown(per_query, "complexity", all_metric_names)
        summary.by_query_style = _group_breakdown(per_query, "query_style", all_metric_names)
        summary.by_difficulty = _group_breakdown(per_query, "difficulty", all_metric_names)
        return summary

    # ------------------------------------------------------------------
    # Exporters
    # ------------------------------------------------------------------

    @staticmethod
    def save_json(summary: EvaluationSummary, path: str | Path) -> None:
        """Save the evaluation summary as JSON."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as handle:
            json.dump(summary.to_dict(), handle, ensure_ascii=False, indent=2)
        logger.info("Saved evaluation JSON to %s", target_path)

    @staticmethod
    def save_markdown(summary: EvaluationSummary, path: str | Path) -> None:
        """Save the evaluation summary as Markdown."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(summary.render_markdown(), encoding="utf-8")
        logger.info("Saved evaluation Markdown to %s", target_path)


# ----------------------------------------------------------------------
# Module-level aggregation helpers
# ----------------------------------------------------------------------


def _aggregate_stats(
    per_query: list[QueryEvalResult],
    metric_names: list[str],
) -> dict[str, MetricAggregate]:
    """Compute mean/min/max/count and pass rate for every named metric."""
    aggregates: dict[str, MetricAggregate] = {}
    for name in metric_names:
        values: list[float] = []
        passed = 0
        for item in per_query:
            if name in item.metric_scores:
                values.append(item.metric_scores[name])
            result = item.metric_results.get(name, {})
            if result.get("success"):
                passed += 1
        if not values:
            continue
        aggregates[name] = MetricAggregate(
            mean=sum(values) / len(values),
            min=min(values),
            max=max(values),
            count=len(values),
            pass_rate=passed / len(values),
        )
    return aggregates


def _group_breakdown(
    per_query: list[QueryEvalResult],
    dimension: str,
    metric_names: list[str],
) -> dict[str, dict[str, float]]:
    """Mean metric score per value of a stratification axis.

    Rows with a blank axis value are grouped under "UNSPECIFIED" so breakdowns
    stay readable for mixed datasets.
    """
    # metric name -> axis value -> accumulated scores
    groups: dict[str, dict[str, list[float]]] = {name: {} for name in metric_names}
    counts: dict[str, int] = {}
    for item in per_query:
        key = str(item.metadata.get(dimension) or "UNSPECIFIED")
        counts[key] = counts.get(key, 0) + 1
        for name in metric_names:
            if name in item.metric_scores:
                groups[name].setdefault(key, []).append(item.metric_scores[name])

    breakdown: dict[str, dict[str, float]] = {}
    for key, count in counts.items():
        row: dict[str, float] = {"count": float(count)}
        for name in metric_names:
            values = groups[name].get(key)
            if values:
                row[name] = sum(values) / len(values)
        breakdown[key] = row
    return breakdown
