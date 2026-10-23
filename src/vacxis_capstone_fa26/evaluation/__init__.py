"""Evaluation metrics and assessment suites for VacxiS RAG."""
from __future__ import annotations

from vacxis_capstone_fa26.evaluation.evaluator import (
    GenerationEvaluator,
    RetrievalEvaluator,
    VacxisRAGEvaluator,
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
    f1_at_k,
    hit_rate_at_k,
    idcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from vacxis_capstone_fa26.evaluation.schemas import (
    EvaluationSummary,
    MetricAggregate,
    QueryGenerationResult,
    QueryRetrievalResult,
)

__all__ = [
    # Retrieval
    "hit_rate_at_k",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "reciprocal_rank",
    "average_precision",
    "dcg_at_k",
    "idcg_at_k",
    "ndcg_at_k",
    # Generation & Grounding
    "normalize_text",
    "exact_match",
    "token_precision_recall_f1",
    "rouge_n",
    "rouge_l",
    "compute_rouge",
    "bleu_score",
    "citation_precision",
    "citation_recall",
    "citation_f1",
    "citation_validity",
    "evidence_span_coverage",
    "refusal_correctness",
    # Evaluators
    "RetrievalEvaluator",
    "GenerationEvaluator",
    "VacxisRAGEvaluator",
    # Schemas
    "QueryRetrievalResult",
    "QueryGenerationResult",
    "MetricAggregate",
    "EvaluationSummary",
]
