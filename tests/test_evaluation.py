"""Tests for VacxiS evaluation metrics and evaluators."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from vacxis_capstone_fa26.evaluation.evaluator import (
    GenerationEvaluator,
    RetrievalEvaluator,
    VacxisRAGEvaluator,
    compute_metric_aggregates,
)
from vacxis_capstone_fa26.evaluation.generation import (
    bleu_score,
    citation_f1,
    citation_precision,
    citation_recall,
    citation_validity,
    compute_rouge,
    evidence_span_coverage,
    exact_match,
    normalize_text,
    refusal_correctness,
    rouge_l,
    rouge_n,
    token_precision_recall_f1,
)
from vacxis_capstone_fa26.evaluation.retrieval import (
    average_precision,
    dcg_at_k,
    evaluate_retrieval_query,
    f1_at_k,
    hit_rate_at_k,
    idcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from vacxis_capstone_fa26.evaluation.schemas import EvaluationSummary, MetricAggregate


# ---------------------------------------------------------------------------
# Retrieval Metrics Tests
# ---------------------------------------------------------------------------


def test_hit_rate_at_k():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    assert hit_rate_at_k(retrieved, ["c1"], k=1) == 1.0
    assert hit_rate_at_k(retrieved, ["c3"], k=2) == 0.0
    assert hit_rate_at_k(retrieved, ["c3"], k=3) == 1.0
    assert hit_rate_at_k(retrieved, ["c99"], k=5) == 0.0
    assert hit_rate_at_k(retrieved, ["c1"], k=0) == 0.0
    assert hit_rate_at_k([], [], k=5) == 1.0


def test_precision_at_k():
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = {"c1", "c3"}
    assert precision_at_k(retrieved, relevant, k=1) == 1.0  # 1/1
    assert precision_at_k(retrieved, relevant, k=2) == 0.5  # 1/2
    assert precision_at_k(retrieved, relevant, k=4) == 0.5  # 2/4
    assert precision_at_k(retrieved, set(), k=2) == 0.0
    assert precision_at_k(retrieved, relevant, k=0) == 0.0


def test_recall_at_k():
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = {"c1", "c3", "c5", "c6"}
    assert recall_at_k(retrieved, relevant, k=1) == 0.25  # 1/4
    assert recall_at_k(retrieved, relevant, k=3) == 0.5   # 2/4
    assert recall_at_k(retrieved, relevant, k=4) == 0.5   # 2/4
    assert recall_at_k(retrieved, set(), k=2) == 0.0


def test_f1_at_k():
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = {"c1", "c2"}
    # at k=2: prec = 2/2 = 1.0, rec = 2/2 = 1.0 => f1 = 1.0
    assert pytest.approx(f1_at_k(retrieved, relevant, k=2), 0.001) == 1.0
    # at k=4: prec = 2/4 = 0.5, rec = 2/2 = 1.0 => f1 = 2*(0.5*1)/(1.5) = 0.6667
    assert pytest.approx(f1_at_k(retrieved, relevant, k=4), 0.001) == 0.6667
    assert f1_at_k(retrieved, set(), k=2) == 0.0


def test_reciprocal_rank():
    retrieved = ["c1", "c2", "c3", "c4"]
    assert reciprocal_rank(retrieved, ["c1"]) == 1.0
    assert reciprocal_rank(retrieved, ["c2"]) == 0.5
    assert reciprocal_rank(retrieved, ["c4"]) == 0.25
    assert reciprocal_rank(retrieved, ["c99"]) == 0.0
    assert reciprocal_rank(retrieved, ["c4"], k=2) == 0.0


def test_average_precision():
    retrieved = ["c1", "c2", "c3", "c4"]
    relevant = ["c1", "c3"]
    # rank 1: hit (P=1/1)
    # rank 2: miss
    # rank 3: hit (P=2/3)
    # AP = (1.0 + 2/3) / 2 = 1.6667 / 2 = 0.8333
    ap = average_precision(retrieved, relevant, k=4)
    assert pytest.approx(ap, 0.001) == 0.8333
    assert average_precision(retrieved, [], k=4) == 0.0


def test_ndcg_at_k():
    retrieved = ["c1", "c2", "c3"]
    scores = {"c1": 3.0, "c2": 2.0, "c3": 1.0}
    # Ideal order is c1, c2, c3
    assert pytest.approx(ndcg_at_k(retrieved, scores, k=3), 0.001) == 1.0

    # Reversed order
    rev_retrieved = ["c3", "c2", "c1"]
    ndcg_rev = ndcg_at_k(rev_retrieved, scores, k=3)
    assert ndcg_rev < 1.0
    assert ndcg_rev > 0.0

    # Zero scores
    assert ndcg_at_k(retrieved, {}, k=3) == 1.0


def test_evaluate_retrieval_query_chunk_and_doc():
    retrieved = [
        {"chunk_id": "vax1_c_0001", "score": 0.9},
        {"chunk_id": "vax2_c_0002", "score": 0.8},
        {"chunk_id": "vax1_c_0003", "score": 0.7},
    ]
    relevant_chunks = [
        {"CHUNK_ID": "vax1_c_0001", "RELEVANCE_SCORE": 3},
        {"CHUNK_ID": "vax1_c_0003", "RELEVANCE_SCORE": 1},
    ]
    relevant_docs = ["vax1"]

    res = evaluate_retrieval_query(
        retrieved=retrieved,
        relevant_chunks=relevant_chunks,
        relevant_docs=relevant_docs,
        k_values=[1, 3],
        query_id="Q01",
        query_text="Liều dùng vắc xin?",
    )

    assert res.query_id == "Q01"
    assert res.metrics["hit_rate@1"] == 1.0
    assert res.metrics["recall@1"] == 0.5
    assert res.metrics["recall@3"] == 1.0
    assert res.metrics["mrr"] == 1.0
    assert res.metrics["doc_hit_rate@1"] == 1.0
    assert res.metrics["doc_recall@1"] == 1.0


# ---------------------------------------------------------------------------
# Generation & Grounding Metrics Tests
# ---------------------------------------------------------------------------


def test_normalize_text():
    raw = "Theo hướng dẫn [chunk_001], liều dùng là 0.5ml!"
    norm = normalize_text(raw)
    assert "[chunk_001]" not in norm
    assert norm == "theo hướng dẫn liều dùng là 05ml"


def test_exact_match():
    pred = "Vaccine Gardasil 9 tiêm 0.5 ml [c1]."
    ref = "vaccine gardasil 9 tiêm 0.5 ml"
    assert exact_match(pred, ref) == 1.0
    assert exact_match(pred, "Vaccine Verorab") == 0.0


def test_token_precision_recall_f1():
    pred = "tiêm bắp 0.5 ml vào cơ delta"
    ref = "tiêm bắp liều 0.5 ml"
    prec, rec, f1 = token_precision_recall_f1(pred, ref)
    assert prec > 0.0
    assert rec > 0.0
    assert f1 > 0.0
    assert pytest.approx(f1, 0.01) == (2 * prec * rec) / (prec + rec)

    # Empty inputs
    assert token_precision_recall_f1("", "") == (1.0, 1.0, 1.0)
    assert token_precision_recall_f1("abc", "") == (0.0, 0.0, 0.0)


def test_rouge_n_and_rouge_l():
    pred = "vắc xin phòng bệnh dại cho người"
    ref = "vắc xin phòng bệnh dại ở người lớn"

    p1, r1, f1_1 = rouge_n(pred, ref, n=1)
    assert f1_1 > 0.6

    p2, r2, f1_2 = rouge_n(pred, ref, n=2)
    assert f1_2 > 0.4

    pl, rl, f1_l = rouge_l(pred, ref)
    assert f1_l > 0.5

    all_rouge = compute_rouge(pred, ref)
    assert "rouge_1_f1" in all_rouge
    assert "rouge_2_f1" in all_rouge
    assert "rouge_l_f1" in all_rouge


def test_bleu_score():
    pred = "Gardasil 9 được chỉ định cho người từ 9 đến 45 tuổi"
    ref = "Gardasil 9 được chỉ định cho người từ 9 đến 45 tuổi"
    scores = bleu_score(pred, ref)
    assert pytest.approx(scores["bleu"], 0.01) == 1.0
    assert pytest.approx(scores["bleu_1"], 0.01) == 1.0

    # Completely disjoint
    disjoint_scores = bleu_score("khác hoàn toàn", "nội dung mẫu", smooth=False)
    assert disjoint_scores["bleu"] == 0.0


def test_citation_metrics():
    citations = ["chunk_1", "chunk_2"]
    relevant = ["chunk_1", "chunk_3"]
    retrieved = ["chunk_1", "chunk_2", "chunk_4"]

    prec = citation_precision(citations, relevant)
    assert prec == 0.5  # 1/2
    rec = citation_recall(citations, relevant)
    assert rec == 0.5   # 1/2
    f1 = citation_f1(citations, relevant)
    assert f1 == 0.5

    val = citation_validity(citations, retrieved, invalid_citations=["invalid_bracket"])
    assert pytest.approx(val, 0.01) == 2.0 / 3.0  # 2 valid / 3 total


def test_evidence_span_coverage():
    relevant_chunks = [
        {"CHUNK_ID": "c1", "EVIDENCE_SPANS": [{"START": 10, "END": 50}]},
        {"CHUNK_ID": "c2", "EVIDENCE_SPANS": []},
        {"CHUNK_ID": "c3", "EVIDENCE_SPANS": [{"START": 0, "END": 20}]},
    ]
    # Cited c1 only
    cov = evidence_span_coverage(["c1"], relevant_chunks)
    assert cov == 0.5  # c1 cited out of [c1, c3]

    # Cited both
    cov2 = evidence_span_coverage(["c1", "c3"], relevant_chunks)
    assert cov2 == 1.0


def test_refusal_correctness():
    assert refusal_correctness(is_refusal=True, expected_answerability="UNANSWERABLE") == 1.0
    assert refusal_correctness(is_refusal=False, expected_answerability="ANSWERABLE") == 1.0
    assert refusal_correctness(is_refusal=False, expected_answerability="PARTIALLY_ANSWERABLE") == 1.0
    assert refusal_correctness(is_refusal=True, expected_answerability="ANSWERABLE") == 0.0
    assert refusal_correctness(is_refusal=False, expected_answerability="UNANSWERABLE") == 0.0


# ---------------------------------------------------------------------------
# Evaluator & Summary Tests
# ---------------------------------------------------------------------------


def test_compute_metric_aggregates():
    records = [
        {"f1": 0.8, "recall": 0.9},
        {"f1": 0.6, "recall": 0.7},
    ]
    overall, agg = compute_metric_aggregates(records)
    assert pytest.approx(overall["f1"], 0.001) == 0.7
    assert pytest.approx(overall["recall"], 0.001) == 0.8
    assert agg["f1"].count == 2
    assert agg["f1"].min == 0.6
    assert agg["f1"].max == 0.8


def test_evaluator_and_export(tmp_path: Path):
    queries = [
        {
            "QUERY_ID": "Q1",
            "QUERY": "Liều dùng Verorab?",
            "QUERY_TYPE": "DOSAGE",
            "QUERY_DIFFICULTY": "EASY",
            "QUERY_COMPLEXITY": "SINGLE_HOP",
            "ANSWERABILITY": "ANSWERABLE",
            "RELEVANT_CHUNKS": [{"CHUNK_ID": "verorab_c_001", "RELEVANCE_SCORE": 3}],
            "REFERENCE_ANSWER": "Tiêm 0.5ml.",
        },
        {
            "QUERY_ID": "Q2",
            "QUERY": "Có chữa được sốt rét không?",
            "QUERY_TYPE": "INDICATION",
            "QUERY_DIFFICULTY": "HARD",
            "QUERY_COMPLEXITY": "SINGLE_HOP",
            "ANSWERABILITY": "UNANSWERABLE",
            "RELEVANT_CHUNKS": [],
            "REFERENCE_ANSWER": "Không có thông tin.",
        },
    ]

    answers = [
        {
            "QUERY_ID": "Q1",
            "GENERATED_ANSWER": {
                "answer": "Tiêm 0.5ml [verorab_c_001].",
                "citations": ["verorab_c_001"],
                "invalid_citations": [],
                "retrieved_chunk_ids": ["verorab_c_001"],
                "is_refusal": False,
                "latency_seconds": 0.12,
                "usage": {"total_tokens": 150},
            },
        },
        {
            "QUERY_ID": "Q2",
            "GENERATED_ANSWER": {
                "answer": "Tài liệu không đề cập đến việc điều trị sốt rét.",
                "citations": [],
                "invalid_citations": [],
                "retrieved_chunk_ids": [],
                "is_refusal": True,
                "latency_seconds": 0.08,
                "usage": {"total_tokens": 80},
            },
        },
    ]

    rag_evaluator = VacxisRAGEvaluator(k_values=[1, 3])
    summary = rag_evaluator.evaluate_results(dataset_items=queries, generation_results=answers)

    assert summary.num_queries == 2
    assert summary.overall_metrics["refusal_correctness"] == 1.0
    assert summary.overall_metrics["hit_rate@1"] == 1.0
    assert "DOSAGE" in summary.by_query_type
    assert "INDICATION" in summary.by_query_type
    assert "EASY" in summary.by_difficulty

    # Test Markdown rendering
    md = summary.render_markdown()
    assert "# VacxiS Evaluation Report" in md
    assert "DOSAGE" in md

    # Test JSON export
    json_path = tmp_path / "summary.json"
    VacxisRAGEvaluator.save_json(summary, json_path)
    assert json_path.exists()
    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["num_queries"] == 2

    # Test Parquet export using PyArrow
    parquet_path = tmp_path / "per_query.parquet"
    VacxisRAGEvaluator.save_parquet(summary, parquet_path)
    assert parquet_path.exists()

    # Read back and verify with pyarrow.parquet
    table = pq.read_table(parquet_path)
    assert len(table) == 2
    assert "metric_exact_match" in table.column_names
    assert "metric_token_f1" in table.column_names
    assert "retrieval_hit_rate@1" in table.column_names
