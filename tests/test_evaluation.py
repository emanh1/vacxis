"""Tests for the DeepEval-based VacxiS evaluation module."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from openai import OpenAI

from deepeval.evaluate.types import EvaluationResult, TestResult
from deepeval.models import DeepEvalBaseLLM
from deepeval.tracing.api import MetricData

from vacxis_capstone_fa26.evaluation import (
    VacxisRAGEvaluator,
    align_results,
    answer_fields,
    load_chunk_texts,
    load_goldens,
    retrieval_context_from_answer,
    validate_query_ids,
)
from vacxis_capstone_fa26.evaluation.cli import _save_parquet
from vacxis_capstone_fa26.evaluation.schemas import EvaluationSummary

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_DATASET = PROJECT_ROOT / "evaluation" / "eval_dataset_300_final.json"
FLAT_DATASET = PROJECT_ROOT / "test_dataset.json"
CHUNKS_PATH = PROJECT_ROOT / "chunks_storage" / "chunks.parquet"


class _StubJudge(DeepEvalBaseLLM):
    """Offline judge that satisfies the abstract methods without a live client."""

    def __init__(self) -> None:
        super().__init__(model="stub-judge")

    def load_model(self) -> OpenAI:
        return OpenAI(api_key="x", base_url="http://localhost")

    def get_model_name(self) -> str:
        return "stub-judge"

    def generate(self, prompt: str, schema=None):
        return "ok"

    async def a_generate(self, prompt: str, schema=None):
        return "ok"


def _make_evaluator(enable_geval: bool = True) -> VacxisRAGEvaluator:
    """An evaluator with a stub judge so no network or key is needed."""
    return VacxisRAGEvaluator(
        judge_model=_StubJudge(), threshold=0.5, enable_geval=enable_geval
    )


def _fake_metric(name: str, score: float, success: bool = True) -> MetricData:
    """Build a MetricData row the way DeepEval's evaluate() does."""
    return MetricData(
        name=name,
        threshold=0.5,
        success=success,
        score=score,
        reason="stub reason",
        strict_mode=False,
        flaky=False,
        evaluation_model="stub-judge",
        error=None,
        evaluation_cost=0.0,
        input_tokens=0,
        output_tokens=0,
        verbose_logs=None,
        expectations_data=None,
    )


def _fake_result(num_cases: int, metric_names: list[str]) -> EvaluationResult:
    """Fabricate a DeepEval EvaluationResult with one TestResult per case."""
    test_results = []
    for i in range(num_cases):
        metrics_data = [_fake_metric(name, 0.5 + 0.01 * i) for name in metric_names]
        test_results.append(
            TestResult(
                name=f"case{i}",
                success=True,
                conversational=False,
                index=i,
                input=f"q{i}",
                actual_output=f"a{i}",
                expected_output=f"e{i}",
                retrieval_context=["ctx"],
                metadata={"query_id": f"qid{i}"},
                metrics_data=metrics_data,
            )
        )
    return EvaluationResult(
        test_results=test_results, confident_link=None, test_run_id="stub-run"
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def test_load_goldens_main_dataset():
    records = load_goldens(MAIN_DATASET)
    assert len(records) == 300
    first = records[0]
    assert first.query_id == "eval_0001"
    assert first.answerability == "ANSWERABLE"
    assert first.complexity == "SINGLE_HOP"
    assert first.query_style == "PATIENT_STYLE"
    assert first.query.strip()
    assert first.expected_output.strip()
    assert first.context and first.context[0].strip()


def test_load_goldens_flat_dataset():
    records = load_goldens(FLAT_DATASET)
    assert len(records) == 460
    first = records[0]
    assert first.query_id == "VAX_Q_0001"
    assert first.answerability == "UNANSWERABLE"
    assert first.difficulty == "MEDIUM"
    assert first.query_type == "ADVERSE_EFFECT"
    # The flat RELEVANT_CHUNKS carry a TEXT field, normalized to context strings.
    has_context = any(r.context for r in records)
    assert has_context


def test_load_chunk_texts():
    texts = load_chunk_texts(CHUNKS_PATH)
    assert texts, "chunk text store should not be empty"
    assert all(isinstance(v, str) and v.strip() for v in texts.values())


def test_load_chunk_texts_missing_returns_empty(tmp_path: Path):
    assert load_chunk_texts(tmp_path / "nope.parquet") == {}


def test_retrieval_context_from_answer(tmp_path: Path):
    chunk_texts = {"abc_c_001": "text A", "abc_c_002": "text B"}
    hydrated = retrieval_context_from_answer(
        {"retrieved_chunk_ids": ["abc_c_001", "abc_c_002", "missing"]},
        chunk_texts,
    )
    assert hydrated == ["text A", "text B"]
    # Unknown ids are dropped.
    assert retrieval_context_from_answer({"retrieved_chunk_ids": ["nope"]}, chunk_texts) == []


def test_answer_fields_unwrapped():
    fields = answer_fields(
        {
            "answer": "a",
            "is_refusal": True,
            "retrieved_chunk_ids": ["c1", "c2"],
        }
    )
    assert fields == {
        "actual_output": "a",
        "is_refusal": True,
        "retrieved_chunk_ids": ["c1", "c2"],
    }


def test_answer_fields_wrapped_in_generated_answer():
    fields = answer_fields(
        {"GENERATED_ANSWER": {"answer": "a", "is_refusal": False, "retrieved_chunk_ids": ["c"]}}
    )
    assert fields["actual_output"] == "a"
    assert fields["is_refusal"] is False
    assert fields["retrieved_chunk_ids"] == ["c"]


# ---------------------------------------------------------------------------
# Alignment & IDs
# ---------------------------------------------------------------------------


def test_validate_query_ids_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_query_ids([{"QUERY_ID": "a"}, {"QUERY_ID": "a"}])


def test_validate_query_ids_rejects_missing():
    with pytest.raises(ValueError):
        validate_query_ids([{"QUERY": "q"}])


def test_validate_query_ids_reads_deep_metadata():
    by_id = validate_query_ids([{"additional_metadata": {"query_id": "eval_1"}}])
    assert "eval_1" in by_id


def test_align_results_preserves_result_order():
    dataset = [
        {"QUERY_ID": "B", "QUERY": "qb"},
        {"QUERY_ID": "A", "QUERY": "qa"},
    ]
    results = [
        {"QUERY_ID": "B", "GENERATED_ANSWER": {"answer": "xb"}},
        {"QUERY_ID": "A", "GENERATED_ANSWER": {"answer": "xa"}},
    ]
    aligned = align_results(dataset, results)
    assert [item["QUERY_ID"] for item in aligned] == ["B", "A"]


def test_align_results_mismatched_ids_raises():
    dataset = [{"QUERY_ID": "A"}]
    results = [{"QUERY_ID": "B"}]
    with pytest.raises(ValueError, match="disagree"):
        align_results(dataset, results)


# ---------------------------------------------------------------------------
# Metrics & test-case construction
# ---------------------------------------------------------------------------


def test_build_metrics_default_includes_geval():
    evaluator = _make_evaluator(enable_geval=True)
    names = [m.__name__ for m in evaluator.build_metrics()]
    assert len(names) == 6
    assert "Answer Relevancy" in names
    assert "Faithfulness" in names
    assert any("Contextual" in n for n in names)


def test_build_metrics_without_geval():
    evaluator = _make_evaluator(enable_geval=False)
    assert len(evaluator.build_metrics()) == 5


# ---------------------------------------------------------------------------
# Summary construction, refusal correctness, and exports
# ---------------------------------------------------------------------------


def _small_dataset_and_results(chunk_texts: dict[str, str]):
    """Two flat goldens plus matching precomputed results."""
    dataset = [
        {
            "QUERY_ID": "Q1",
            "QUERY": "Liều dùng Verorab?",
            "QUERY_TYPE": "DOSAGE",
            "QUERY_DIFFICULTY": "EASY",
            "QUERY_COMPLEXITY": "SINGLE_HOP",
            "QUERY_STYLE": "PATIENT_STYLE",
            "ANSWERABILITY": "ANSWERABLE",
            "REFERENCE_ANSWER": "Tiêm 0.5ml.",
            "RELEVANT_CHUNKS": [],
        },
        {
            "QUERY_ID": "Q2",
            "QUERY": "Có chữa được sốt rét không?",
            "QUERY_TYPE": "INDICATION",
            "QUERY_DIFFICULTY": "HARD",
            "QUERY_COMPLEXITY": "SINGLE_HOP",
            "QUERY_STYLE": "CLINICAL_STYLE",
            "ANSWERABILITY": "UNANSWERABLE",
            "REFERENCE_ANSWER": "Không có thông tin.",
            "RELEVANT_CHUNKS": [],
        },
    ]
    results = [
        {
            "QUERY_ID": "Q1",
            "GENERATED_ANSWER": {
                "answer": "Tiêm 0.5ml [verorab_c_001].",
                "citations": ["verorab_c_001"],
                "invalid_citations": [],
                "retrieved_chunk_ids": list(chunk_texts.keys())[:1],
                "is_refusal": False,
            },
        },
        {
            "QUERY_ID": "Q2",
            "GENERATED_ANSWER": {
                "answer": "Tài liệu không đề cập đến sốt rét.",
                "citations": [],
                "invalid_citations": [],
                "retrieved_chunk_ids": [],
                "is_refusal": True,
            },
        },
    ]
    return dataset, results


def test_evaluate_results_builds_full_summary(tmp_path: Path, monkeypatch):
    evaluator = _make_evaluator(enable_geval=True)
    chunk_texts = {"verorab_c_001": "text about Verorab dosing"}
    dataset, results = _small_dataset_and_results(chunk_texts)

    metric_names = [m.__name__ for m in evaluator.build_metrics()]
    monkeypatch.setattr(
        evaluator,
        "_run_deepeval",
        lambda cases, **kw: _fake_result(len(cases), metric_names),
    )

    summary = evaluator.evaluate_results(dataset, results, chunk_texts=chunk_texts)

    assert summary.num_queries == 2
    # 5 RAG + GEval + refusal indicator.
    assert len(summary.metric_names) == 7
    assert "refusal_correctness" in summary.metric_names
    # Both answers align with their ANSWERABILITY ground truth.
    assert summary.overall_metrics["refusal_correctness"] == 1.0
    assert "ANSWERABLE" in summary.by_answerability
    assert "UNANSWERABLE" in summary.by_answerability
    assert "EASY" in summary.by_difficulty
    # Per-query records carry the judge reasons and hydrated context.
    assert summary.per_query_results[0]["query_id"] == "Q1"
    assert "Answer Relevancy" in summary.per_query_results[0]["metric_results"]

    # Exports.
    json_path = tmp_path / "summary.json"
    VacxisRAGEvaluator.save_json(summary, json_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["num_queries"] == 2
    assert loaded["overall_metrics"]["refusal_correctness"] == 1.0

    md_path = tmp_path / "report.md"
    VacxisRAGEvaluator.save_markdown(summary, md_path)
    markdown = md_path.read_text(encoding="utf-8")
    assert "# VacxiS DeepEval Report" in markdown
    assert "Answer Relevancy" in markdown


def test_refusal_mismatch_scores_zero():
    evaluator = _make_evaluator(enable_geval=False)
    # An ANSWERABLE question answered with a refusal should score 0.
    dataset = [
        {
            "QUERY_ID": "Q1",
            "QUERY": "q",
            "ANSWERABILITY": "ANSWERABLE",
            "REFERENCE_ANSWER": "a",
        }
    ]
    results = [
        {"QUERY_ID": "Q1", "GENERATED_ANSWER": {"answer": "x", "is_refusal": True}}
    ]
    metric_names = [m.__name__ for m in evaluator.build_metrics()]
    evaluator._run_deepeval = lambda cases, **kw: _fake_result(1, metric_names)
    summary = evaluator.evaluate_results(dataset, results, chunk_texts={})
    assert summary.overall_metrics["refusal_correctness"] == 0.0


def test_save_parquet_round_trip(tmp_path: Path):
    evaluator = _make_evaluator(enable_geval=True)
    dataset, results = _small_dataset_and_results({"verorab_c_001": "t"})
    metric_names = [m.__name__ for m in evaluator.build_metrics()]
    evaluator._run_deepeval = lambda cases, **kw: _fake_result(2, metric_names)
    summary = evaluator.evaluate_results(dataset, results, chunk_texts={})

    parquet_path = tmp_path / "per_query.parquet"
    _save_parquet(summary, parquet_path)
    assert parquet_path.exists()

    table = pq.read_table(parquet_path)
    assert len(table) == 2
    assert "query_id" in table.column_names
    assert "metric_refusal_correctness" in table.column_names
    assert "meta_answerability" in table.column_names


def test_summary_to_dict_is_json_safe(tmp_path: Path):
    evaluator = _make_evaluator(enable_geval=True)
    dataset, results = _small_dataset_and_results({})
    metric_names = [m.__name__ for m in evaluator.build_metrics()]
    evaluator._run_deepeval = lambda cases, **kw: _fake_result(2, metric_names)
    summary = evaluator.evaluate_results(dataset, results, chunk_texts={})
    # A JSON round-trip must not raise (no sets, no non-serializable objects).
    serialized = json.dumps(summary.to_dict(), ensure_ascii=False)
    assert "refusal_correctness" in serialized


# ---------------------------------------------------------------------------
# DeepEval cache (resume) wiring
# ---------------------------------------------------------------------------


def test_run_deepeval_defaults_read_the_cache():
    """_run_deepeval must read and write the cache so stopped runs can resume."""
    from vacxis_capstone_fa26.evaluation import evaluator as evaluator_module
    from vacxis_capstone_fa26.evaluation.datasets import to_record

    captured = {}

    def _fake_deepeval_evaluate(**kwargs):
        captured.update(kwargs)
        return _fake_result(1, [])

    evaluator = _make_evaluator()
    record = to_record(
        {"QUERY_ID": "Q1", "QUERY": "q", "REFERENCE_ANSWER": "a"}
    )
    info = evaluator._case_info(
        {"actual_output": "a", "is_refusal": False, "retrieved_chunk_ids": []}, {}
    )
    cases = [evaluator._build_test_case(record, info)]

    original = evaluator_module.deepeval_evaluate
    evaluator_module.deepeval_evaluate = _fake_deepeval_evaluate
    try:
        evaluator._run_deepeval(cases, max_concurrent=1)
    finally:
        evaluator_module.deepeval_evaluate = original

    cache_config = captured["cache_config"]
    assert cache_config.write_cache is True
    assert cache_config.use_cache is True


def test_evaluate_live_forwards_cache_flags():
    """evaluate_live must forward use_cache/write_cache down to _run_deepeval."""
    evaluator = _make_evaluator()
    dataset, results = _small_dataset_and_results({})
    metric_names = [m.__name__ for m in evaluator.build_metrics()]

    seen: dict = {}

    def _capture(cases, **kw):
        seen.update(kw)
        return _fake_result(len(cases), metric_names)

    evaluator._run_deepeval = _capture
    evaluator.evaluate_live(
        dataset,
        lambda query, golden: {"answer": "a", "is_refusal": False, "retrieved_chunk_ids": []},
        chunk_texts={},
        use_cache=False,
        write_cache=False,
    )
    assert seen["use_cache"] is False
    assert seen["write_cache"] is False


# ---------------------------------------------------------------------------
# Judge-timeout CLI flag
# ---------------------------------------------------------------------------


def test_apply_judge_timeout_sets_env_override(monkeypatch):
    from vacxis_capstone_fa26.evaluation import cli as eval_cli

    monkeypatch.delenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", raising=False)
    args = argparse.Namespace(judge_timeout=1200.0)
    eval_cli._apply_judge_timeout(args)
    assert os.environ.get("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE") == "1200.0"


def test_apply_judge_timeout_noop_when_unset(monkeypatch):
    from vacxis_capstone_fa26.evaluation import cli as eval_cli

    monkeypatch.delenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", raising=False)
    args = argparse.Namespace(judge_timeout=None)
    eval_cli._apply_judge_timeout(args)
    assert "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE" not in os.environ


# ---------------------------------------------------------------------------
# Deterministic IR Metrics
# ---------------------------------------------------------------------------


def test_ir_metrics_empty_inputs():
    from vacxis_capstone_fa26.evaluation.ir_metrics import (
        hit_at_k,
        map_at_k,
        mrr_at_k,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )

    assert hit_at_k([], ["c1"], 5) == 0.0
    assert hit_at_k(["c1"], [], 5) == 0.0
    assert hit_at_k(["c1"], ["c1"], 0) == 0.0
    assert recall_at_k([], ["c1"], 5) == 0.0
    assert recall_at_k(["c1"], [], 5) == 0.0
    assert precision_at_k([], ["c1"], 5) == 0.0
    assert mrr_at_k([], ["c1"], 5) == 0.0
    assert ndcg_at_k([], ["c1"], 5) == 0.0
    assert map_at_k([], ["c1"], 5) == 0.0


def test_ir_metrics_perfect_ranking():
    from vacxis_capstone_fa26.evaluation.ir_metrics import (
        hit_at_k,
        map_at_k,
        mrr_at_k,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )

    retrieved = ["c1", "c2", "c3"]
    relevant = ["c1", "c2"]

    assert hit_at_k(retrieved, relevant, 1) == 1.0
    assert recall_at_k(retrieved, relevant, 1) == 0.5
    assert precision_at_k(retrieved, relevant, 1) == 1.0
    assert mrr_at_k(retrieved, relevant, 1) == 1.0
    assert map_at_k(retrieved, relevant, 1) == 0.5
    assert ndcg_at_k(retrieved, relevant, 1) == 1.0

    assert hit_at_k(retrieved, relevant, 2) == 1.0
    assert recall_at_k(retrieved, relevant, 2) == 1.0
    assert precision_at_k(retrieved, relevant, 2) == 1.0
    assert mrr_at_k(retrieved, relevant, 2) == 1.0
    assert map_at_k(retrieved, relevant, 2) == 1.0
    assert ndcg_at_k(retrieved, relevant, 2) == 1.0


def test_ir_metrics_partial_ranking():
    from vacxis_capstone_fa26.evaluation.ir_metrics import (
        hit_at_k,
        map_at_k,
        mrr_at_k,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )

    # c1 is distractor, c2 is relevant
    retrieved = ["distractor", "c2", "other"]
    relevant = ["c2"]

    assert hit_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert mrr_at_k(retrieved, relevant, 1) == 0.0

    assert hit_at_k(retrieved, relevant, 3) == 1.0
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert pytest.approx(precision_at_k(retrieved, relevant, 3)) == 1.0 / 3.0
    assert pytest.approx(mrr_at_k(retrieved, relevant, 3)) == 0.5
    assert pytest.approx(map_at_k(retrieved, relevant, 3)) == 0.5
    assert 0.0 < ndcg_at_k(retrieved, relevant, 3) < 1.0


def test_ir_metrics_deduplication():
    from vacxis_capstone_fa26.evaluation.ir_metrics import hit_at_k, recall_at_k

    # Duplicates in retrieved must not artificially inflate recall or hit count
    retrieved = ["c1", "c1", "c2"]
    relevant = ["c1"]
    assert recall_at_k(retrieved, relevant, 2) == 1.0
    assert hit_at_k(retrieved, relevant, 1) == 1.0


def test_compute_query_ir_metrics_and_aggregate():
    from vacxis_capstone_fa26.evaluation.ir_metrics import (
        aggregate_ir_metrics,
        compute_query_ir_metrics,
    )

    scores1 = compute_query_ir_metrics(["c1", "c2"], ["c1"], k_values=(1, 3))
    assert scores1["Hit@1"] == 1.0
    assert scores1["Recall@1"] == 1.0
    assert scores1["MRR@1"] == 1.0
    assert "Hit@3" in scores1

    scores2 = compute_query_ir_metrics(["c_wrong"], ["c1"], k_values=(1, 3))
    assert scores2["Hit@1"] == 0.0
    assert scores2["Recall@1"] == 0.0

    agg = aggregate_ir_metrics([scores1, scores2], threshold=0.5)
    assert agg["Hit@1"].count == 2
    assert agg["Hit@1"].mean == 0.5
    assert agg["Hit@1"].pass_rate == 0.5


def test_load_goldens_extracts_relevant_chunk_ids():
    records = load_goldens(MAIN_DATASET)
    assert len(records) == 300
    first = records[0]
    assert first.relevant_chunk_ids == ["proquad_001310197825_c_0000"]

    flat_records = load_goldens(FLAT_DATASET)
    assert len(flat_records) == 460
    has_rel = any(len(r.relevant_chunk_ids) > 0 for r in flat_records)
    assert has_rel


def test_evaluate_retrieval_standalone(tmp_path: Path):
    evaluator = _make_evaluator()
    dataset = [
        {
            "QUERY_ID": "Q1",
            "QUERY": "q1",
            "ANSWERABILITY": "ANSWERABLE",
            "additional_metadata": {"source_chunk_ids": ["c1", "c2"]},
        },
        {
            "QUERY_ID": "Q2",
            "QUERY": "q2",
            "ANSWERABILITY": "ANSWERABLE",
            "additional_metadata": {"source_chunk_ids": ["c3"]},
        },
    ]
    results = [
        {
            "QUERY_ID": "Q1",
            "GENERATED_ANSWER": {
                "answer": "ans1",
                "retrieved_chunk_ids": ["c1", "distractor"],
            },
        },
        {
            "QUERY_ID": "Q2",
            "GENERATED_ANSWER": {
                "answer": "ans2",
                "retrieved_chunk_ids": ["other", "c3"],
            },
        },
    ]
    summary = evaluator.evaluate_retrieval(dataset, results, k_values=(1, 3))
    assert summary.task == "retrieval"
    assert summary.num_queries == 2
    assert "Hit@1" in summary.metric_names
    assert "Hit@3" in summary.metric_names
    assert summary.overall_metrics["Hit@1"] == 0.5
    assert summary.overall_metrics["Hit@3"] == 1.0
    assert summary.overall_metrics["MRR@1"] == 0.5
    assert summary.overall_metrics["MRR@3"] == 0.75

    # Check Markdown report formatting
    md_path = tmp_path / "retrieval_report.md"
    VacxisRAGEvaluator.save_markdown(summary, md_path)
    md_content = md_path.read_text(encoding="utf-8")
    assert "# VacxiS DeepEval Report: RETRIEVAL" in md_content
    assert "Hit@1" in md_content


def test_evaluate_results_includes_ir_metrics(monkeypatch):
    evaluator = _make_evaluator(enable_geval=False)
    dataset = [
        {
            "QUERY_ID": "Q1",
            "QUERY": "q1",
            "ANSWERABILITY": "ANSWERABLE",
            "additional_metadata": {"source_chunk_ids": ["c1"]},
        }
    ]
    results = [
        {
            "QUERY_ID": "Q1",
            "GENERATED_ANSWER": {
                "answer": "ans1",
                "is_refusal": False,
                "retrieved_chunk_ids": ["c1"],
            },
        }
    ]
    metric_names = [m.__name__ for m in evaluator.build_metrics()]
    evaluator._run_deepeval = lambda cases, **kw: _fake_result(1, metric_names)
    summary = evaluator.evaluate_results(dataset, results, chunk_texts={})
    assert "Hit@1" in summary.metric_names
    assert summary.overall_metrics["Hit@1"] == 1.0
    assert summary.overall_metrics["MRR@1"] == 1.0


def test_cli_retrieval_only_flag():
    from vacxis_capstone_fa26.evaluation.cli import parse_args
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["vacxis-eval", "--retrieval-only", "--results", "results.json"]
        args = parse_args()
        assert args.retrieval_only is True
        assert args.skip_ir is False
    finally:
        sys.argv = old_argv


def test_retrieval_report_preserves_rankings_and_excludes_unlabeled_queries():
    evaluator = VacxisRAGEvaluator(ir_k_values=(1, 3, 5))
    dataset = [
        {"QUERY_ID": "Q1", "QUERY": "q1", "relevant_chunk_ids": ["c1"]},
        {"QUERY_ID": "Q2", "QUERY": "q2"},
    ]
    results = [
        {"QUERY_ID": "Q2", "retrieved_chunk_ids": []},
        {"QUERY_ID": "Q1", "retrieved_chunk_ids": ["c1", " c1 ", " ", "other"]},
    ]
    summary = evaluator.evaluate_retrieval(dataset, results)
    assert summary.metric_aggregates["Hit@1"].count == 1
    assert summary.metric_aggregates["Hit@1"].pass_rate == 1.0
    assert summary.metadata["scored_queries"] == 1
    assert summary.metadata["unscored_queries"] == 1
    assert summary.metadata["short_rankings_by_cutoff"] == {"1": 1, "3": 2, "5": 2}
    assert summary.per_query_results[0]["metadata"]["ir_scored"] is False
    metadata = summary.per_query_results[1]["metadata"]
    assert metadata["retrieved_chunk_ids"] == ["c1", " c1 ", " ", "other"]
    assert metadata["relevant_chunk_ids"] == ["c1"]
    assert "excluded without relevance labels: 1" in summary.render_markdown()
    assert "shorter than the metric cutoff" in summary.render_markdown()


def test_cli_live_ir_cutoffs_respect_retrieval_depth(monkeypatch):
    from vacxis_capstone_fa26.evaluation.cli import parse_args

    monkeypatch.setattr("sys.argv", ["vacxis-eval", "--retrieval-only"])
    assert parse_args().ir_k_values == [1, 3, 5]
    monkeypatch.setattr("sys.argv", ["vacxis-eval", "--retrieval-only", "--top-k", "10"])
    assert parse_args().ir_k_values == [1, 3, 5, 10]
    monkeypatch.setattr("sys.argv", ["vacxis-eval", "--ir-k-values", "1", "10"])
    with pytest.raises(SystemExit):
        parse_args()
    monkeypatch.setattr("sys.argv", ["vacxis-eval", "--results", "results.json"])
    assert parse_args().ir_k_values == [1, 3, 5, 10]


@pytest.mark.parametrize("cutoffs", [(), (0,), (-1,), (1.5,), (True,)])
def test_retrieval_rejects_invalid_cutoffs(cutoffs):
    with pytest.raises(ValueError, match="positive integers"):
        VacxisRAGEvaluator().evaluate_retrieval([], [], k_values=cutoffs)


def test_cli_default_dataset_matches_saved_generation_goldens():
    from vacxis_capstone_fa26.evaluation.cli import _MAIN_DATASET

    corrected = json.loads((PROJECT_ROOT / _MAIN_DATASET).read_text())["goldens"]
    saved = json.loads((PROJECT_ROOT / "evaluation/eval_dataset_300_generation_results.json").read_text())
    by_id = {item["additional_metadata"]["query_id"]: item for item in saved}
    assert len(corrected) == len(by_id) == 300
    for item in corrected:
        result = by_id[item["additional_metadata"]["query_id"]]
        assert item["input"] == result["input"]
        assert item["additional_metadata"]["source_chunk_ids"] == result["additional_metadata"]["source_chunk_ids"]


def test_retrieval_context_from_answer_prefers_retrieved_contexts_with_fallback():
    chunk_texts = {"c1": "raw parquet chunk text"}
    dual_text = "Prose summary\n\n[BẢNG DỮ LIỆU GỐC]\n<table><tr><td>0.5 ml</td></tr></table>"

    # Prefers explicit retrieved_contexts over raw chunk_texts
    assert retrieval_context_from_answer(
        {"retrieved_chunk_ids": ["c1"], "retrieved_contexts": [dual_text]},
        chunk_texts,
    ) == [dual_text]

    # Falls back to chunk_texts when retrieved_contexts is absent or empty
    assert retrieval_context_from_answer(
        {"retrieved_chunk_ids": ["c1"], "retrieved_contexts": []},
        chunk_texts,
    ) == ["raw parquet chunk text"]


def test_answer_fields_and_evaluator_preserve_retrieved_contexts():
    fields = answer_fields(
        {
            "GENERATED_ANSWER": {
                "answer": "Tiêm 0.5 ml.",
                "is_refusal": False,
                "retrieved_chunk_ids": ["c1"],
                "retrieved_contexts": ["Dual context text", "   "],
            }
        }
    )
    assert fields["retrieved_contexts"] == ["Dual context text"]

    evaluator = _make_evaluator(enable_geval=False)
    info = evaluator._case_info(fields, {"c1": "raw text"})
    assert info.retrieval_context == ["Dual context text"]


def test_generated_answer_to_dict_includes_retrieved_contexts():
    from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer

    ans = GeneratedAnswer(
        query="q",
        answer="a",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "  Context 1  "},
            {"chunk_id": "c2", "TEXT": "Context 2"},
        ],
    )
    data = ans.to_dict()
    assert data["retrieved_chunk_ids"] == ["c1", "c2"]
    assert data["retrieved_contexts"] == ["Context 1", "Context 2"]


def test_cli_retriever_mode_and_merge_table_chunks_metadata(monkeypatch, tmp_path):
    from vacxis_capstone_fa26.evaluation import cli

    monkeypatch.setattr(
        "sys.argv",
        ["vacxis-eval", "--retrieval-only", "--merge-table-chunks", "--collection-name", "col_dual"],
    )
    args = cli.parse_args()
    assert args.merge_table_chunks is True
    assert args.retriever_mode is None

    summary = EvaluationSummary()
    cli._record_retrieval_run(summary, args)
    run = summary.metadata["run"]
    assert run["retriever"] == "hybrid"
    assert run["retriever_mode"] == "hybrid"
    assert run["merge_table_chunks"] is True
    assert run["collection_name"] == "col_dual"

    # Mutually exclusive checks
    for bad_args in (
        ["vacxis-eval", "--baseline", "--retriever-mode", "hybrid"],
        ["vacxis-eval", "--advanced-rag", "--retriever-mode", "advanced"],
        ["vacxis-eval", "--retriever-mode", "hybrid", "--results", "saved.json"],
        ["vacxis-eval", "--retriever-mode", "naive", "--chunks", str(tmp_path / "missing.parquet")],
    ):
        monkeypatch.setattr("sys.argv", bad_args)
        with pytest.raises(SystemExit):
            cli.parse_args()


def test_ir_metrics_exclude_lightrag_kg_context_while_deepeval_keeps_it():
    evaluator = _make_evaluator(enable_geval=False)
    dataset = [
        {
            "QUERY_ID": "Q1",
            "QUERY": "Vắc-xin Gardasil 9 phòng bệnh gì?",
            "ANSWERABILITY": "ANSWERABLE",
            "additional_metadata": {"source_chunk_ids": ["c1", "c2"]},
        }
    ]
    kg_text = "[ĐỒ THỊ TRI THỨC Y KHOA (LIGHTRAG)]\n\nThực thể liên quan:\n- Gardasil 9 (VACCINE)"
    results = [
        {
            "QUERY_ID": "Q1",
            "GENERATED_ANSWER": {
                "answer": "Gardasil 9 phòng HPV [c1].",
                "is_refusal": False,
                "retrieved_chunk_ids": ["lightrag_kg_context", "c1", "c2"],
                "retrieved_contexts": [kg_text, "Text c1", "Text c2"],
            },
        }
    ]

    # Retrieval-only IR scoring excludes synthetic lightrag_kg_context
    ir_summary = evaluator.evaluate_retrieval(dataset, results, k_values=(1, 2))
    assert ir_summary.overall_metrics["Hit@1"] == 1.0
    assert ir_summary.overall_metrics["Precision@2"] == 1.0
    assert ir_summary.overall_metrics["Recall@2"] == 1.0
    assert ir_summary.metadata["short_rankings_by_cutoff"] == {"1": 0, "2": 0}
    assert ir_summary.per_query_results[0]["metadata"]["retrieved_chunk_ids"] == [
        "lightrag_kg_context",
        "c1",
        "c2",
    ]

    # Full DeepEval scoring preserves KG context in LLMTestCase.retrieval_context
    captured_cases = []
    metric_names = [m.__name__ for m in evaluator.build_metrics()]

    def _capture(cases, **kw):
        captured_cases.extend(cases)
        return _fake_result(len(cases), metric_names)

    evaluator._run_deepeval = _capture
    full_summary = evaluator.evaluate_results(dataset, results, chunk_texts={})
    assert captured_cases[0].retrieval_context == [kg_text, "Text c1", "Text c2"]
    assert full_summary.overall_metrics["Hit@1"] == 1.0
    assert full_summary.overall_metrics["MRR@1"] == 1.0


def test_cli_lightrag_flags_and_metadata(monkeypatch):
    from vacxis_capstone_fa26.evaluation import cli

    monkeypatch.setattr(
        "sys.argv",
        [
            "vacxis-eval",
            "--retrieval-only",
            "--retriever-mode",
            "hybrid_lightrag",
            "--collection-name",
            "VacxiS_Capstone_FA26_table_dual",
            "--lightrag-mode",
            "mix",
            "--lightrag-fusion",
            "rrf_rerank",
            "--lightrag-working-dir",
            "custom_lr_dir",
            "--lightrag-workspace",
            "vacxis_v2",
            "--lightrag-top-k",
            "15",
            "--lightrag-chunk-top-k",
            "6",
        ],
    )
    args = cli.parse_args()
    summary = EvaluationSummary()
    cli._record_retrieval_run(summary, args)
    run = summary.metadata["run"]
    assert run["retriever"] == "hybrid_lightrag"
    assert run["retriever_mode"] == "hybrid_lightrag"
    assert run["collection_name"] == "VacxiS_Capstone_FA26_table_dual"
    assert run["lightrag_mode"] == "mix"
    assert run["lightrag_fusion"] == "rrf_rerank"
    assert run["lightrag_working_dir"] == "custom_lr_dir"
    assert run["lightrag_workspace"] == "vacxis_v2"
    assert run["lightrag_top_k"] == 15
    assert run["lightrag_chunk_top_k"] == 6

    monkeypatch.setattr(
        "sys.argv",
        ["vacxis-eval", "--retrieval-only", "--retriever-mode", "lightrag", "--lightrag-top-k", "0"],
    )
    with pytest.raises(SystemExit):
        cli.parse_args()


def test_ir_metrics_alias_matching_preserves_ranks_and_bounds():
    from vacxis_capstone_fa26.evaluation.ir_metrics import compute_query_ir_metrics

    # Non-primary sub-chunk c_0028 was merged into primary c_0026 at rank 2
    retrieved = ["c_0003", "c_0026", "c_0040"]
    relevant = ["c_0003", "c_0028"]
    aliases = {"c_0026": ["c_0026", "c_0027", "c_0028"]}

    strict_scores = compute_query_ir_metrics(retrieved, relevant, k_values=(1, 2, 3))
    assert strict_scores["Recall@2"] == 0.5
    assert strict_scores["Precision@2"] == 0.5

    alias_scores = compute_query_ir_metrics(
        retrieved, relevant, k_values=(1, 2, 3), chunk_aliases=aliases
    )
    assert alias_scores["Hit@1"] == 1.0
    assert alias_scores["Hit@2"] == 1.0
    assert alias_scores["Recall@2"] == 1.0
    assert alias_scores["Precision@2"] == 1.0
    assert alias_scores["MRR@2"] == 1.0
    assert alias_scores["NDCG@2"] == pytest.approx(1.0)
    assert alias_scores["MAP@2"] == pytest.approx(1.0)

    # Single merged chunk covering two relevant labels at rank 1 stays bounded <= 1.0
    multi_scores = compute_query_ir_metrics(
        ["c_0026", "other"],
        ["c_0027", "c_0028"],
        k_values=(1, 2),
        chunk_aliases=aliases,
    )
    assert multi_scores["Hit@1"] == 1.0
    assert multi_scores["Recall@1"] == 1.0
    assert multi_scores["Precision@1"] == 1.0
    assert multi_scores["Precision@2"] == 0.5
    assert multi_scores["NDCG@1"] == pytest.approx(1.0)
    assert multi_scores["NDCG@2"] == pytest.approx(1.0)
    assert multi_scores["MAP@2"] == pytest.approx(1.0)

    # Two slots sharing the same alias cannot double-count Precision/NDCG/MAP
    dup_alias_scores = compute_query_ir_metrics(
        ["slot_a", "slot_b"],
        ["target_1"],
        k_values=(2,),
        chunk_aliases={"slot_a": ["slot_a", "target_1"], "slot_b": ["slot_b", "target_1"]},
    )
    assert dup_alias_scores["Precision@2"] == 0.5
    assert dup_alias_scores["Recall@2"] == 1.0
    assert dup_alias_scores["NDCG@2"] == pytest.approx(1.0)
    assert dup_alias_scores["MAP@2"] == pytest.approx(1.0)


def test_merged_chunk_aliases_flow_through_generation_and_evaluator():
    from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer, extract_chunk_aliases
    from vacxis_capstone_fa26.retrieval.schemas import RetrievedChunk

    merged_chunk = RetrievedChunk(
        chunk_id="doc_c_0026",
        doc_id="doc",
        doc_title="Menactra",
        category="DOSAGE",
        parent_header="LIỀU",
        header="Bảng 1",
        content_type="table",
        text="Merged table text",
        score=0.9,
        rrf_score=0.03,
        metadata={"has_table": True, "merged_chunk_ids": ["doc_c_0026", "doc_c_0027", "doc_c_0028"]},
    )
    normal_chunk = RetrievedChunk(
        chunk_id="doc_c_0003",
        doc_id="doc",
        doc_title="Menactra",
        category="INDICATION",
        parent_header="CHỈ ĐỊNH",
        header="Mở đầu",
        content_type="text",
        text="Normal text",
        score=0.8,
        rrf_score=0.02,
        metadata={"has_table": False},
    )

    assert extract_chunk_aliases([normal_chunk]) == {}
    assert extract_chunk_aliases([merged_chunk, normal_chunk]) == {
        "doc_c_0026": ["doc_c_0026", "doc_c_0027", "doc_c_0028"]
    }

    ans_plain = GeneratedAnswer(query="q", answer="a", retrieved_chunks=[normal_chunk])
    assert "retrieved_chunk_aliases" not in ans_plain.to_dict()

    ans_merged = GeneratedAnswer(
        query="q", answer="a", retrieved_chunks=[normal_chunk, merged_chunk]
    )
    ans_dict = ans_merged.to_dict()
    assert ans_dict["retrieved_chunk_aliases"] == {
        "doc_c_0026": ["doc_c_0026", "doc_c_0027", "doc_c_0028"]
    }

    fields = answer_fields({"GENERATED_ANSWER": ans_dict})
    assert fields["retrieved_chunk_aliases"] == {
        "doc_c_0026": ["doc_c_0026", "doc_c_0027", "doc_c_0028"]
    }

    dataset = [
        {
            "QUERY_ID": "eval_0110",
            "QUERY": "Liều tiêm Menactra?",
            "ANSWERABILITY": "ANSWERABLE",
            "additional_metadata": {"source_chunk_ids": ["doc_c_0003", "doc_c_0028"]},
        }
    ]
    results = [{"QUERY_ID": "eval_0110", "GENERATED_ANSWER": ans_dict}]

    evaluator = _make_evaluator(enable_geval=False)
    ir_summary = evaluator.evaluate_retrieval(dataset, results, k_values=(1, 2))
    assert ir_summary.overall_metrics["Hit@2"] == 1.0
    assert ir_summary.overall_metrics["Recall@2"] == 1.0
    assert ir_summary.overall_metrics["Precision@2"] == 1.0
    assert ir_summary.overall_metrics["NDCG@2"] == pytest.approx(1.0)
    assert "merged_chunk_ids" in ir_summary.metadata["relevance"]
    assert ir_summary.per_query_results[0]["metadata"]["retrieved_chunk_aliases"] == {
        "doc_c_0026": ["doc_c_0026", "doc_c_0027", "doc_c_0028"]
    }

    # Full evaluate_results also passes chunk_aliases into IR scoring
    metric_names = [m.__name__ for m in evaluator.build_metrics()]
    evaluator._run_deepeval = lambda cases, **kw: _fake_result(len(cases), metric_names)
    full_summary = evaluator.evaluate_results(dataset, results, chunk_texts={})
    assert full_summary.overall_metrics["Recall@3"] == 1.0
    assert full_summary.overall_metrics["NDCG@3"] == pytest.approx(1.0)


def test_explicit_report_checkpoint_survives_failure(tmp_path, monkeypatch):
    evaluator = _make_evaluator()
    dataset = [
        {'QUERY_ID': str(i), 'QUERY': f'q{i}', 'REFERENCE_ANSWER': 'expected'}
        for i in range(2)
    ]
    results = [
        {'QUERY_ID': str(i), 'GENERATED_ANSWER': {'answer': 'answer', 'retrieved_chunk_ids': []}}
        for i in range(2)
    ]
    names = [metric.__name__ for metric in evaluator.build_metrics()]
    calls = []
    def run(cases, **kwargs):
        calls.append(cases[0].input)
        if len(calls) == 2:
            raise RuntimeError('connection lost')
        return _fake_result(1, names)
    monkeypatch.setattr(evaluator, '_run_deepeval', run)
    path = tmp_path / 'checkpoint.json'
    with pytest.raises(RuntimeError, match='connection lost'):
        evaluator.evaluate_results(dataset, results, checkpoint_path=path)
    assert len(json.loads(path.read_text())['report']['per_query_results']) == 1
    summary = evaluator.evaluate_results(dataset, results, checkpoint_path=path, resume=True)
    assert calls == ['q0', 'q1', 'q1']
    assert summary.num_queries == 2
    assert summary.metadata['complete'] is True
    results[0]['GENERATED_ANSWER']['answer'] = 'changed'
    with pytest.raises(ValueError, match='settings changed'):
        evaluator.evaluate_results(dataset, results, checkpoint_path=path, resume=True)


def test_explicit_checkpoint_rejects_missing_metrics(tmp_path, monkeypatch):
    evaluator = _make_evaluator()
    dataset = [{'QUERY_ID': '1', 'QUERY': 'q', 'REFERENCE_ANSWER': 'a'}]
    results = [{'QUERY_ID': '1', 'GENERATED_ANSWER': {'answer': 'a'}}]
    monkeypatch.setattr(evaluator, '_run_deepeval', lambda *a, **kw: _fake_result(1, []))
    path = tmp_path / 'report.json'
    with pytest.raises(ValueError, match='Incomplete metric'):
        evaluator.evaluate_results(dataset, results, checkpoint_path=path)
    assert json.loads(path.read_text())['report']['per_query_results'] == []
