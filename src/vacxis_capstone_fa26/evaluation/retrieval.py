"""Retrieval evaluation metrics: Hit Rate, Recall, Precision, MRR, MAP, and nDCG."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from vacxis_capstone_fa26.evaluation.schemas import MetricAggregate, QueryRetrievalResult


def hit_rate_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str] | set[str],
    k: int = 5,
) -> float:
    """Compute Hit Rate @ k (1.0 if at least one relevant item is in top-k, else 0.0)."""
    if k <= 0:
        return 0.0
    rel_set = set(relevant)
    if not rel_set:
        return 1.0 if not retrieved else 0.0
    top_k = retrieved[:k]
    return 1.0 if any(item in rel_set for item in top_k) else 0.0


def precision_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str] | set[str],
    k: int = 5,
) -> float:
    """Compute Precision @ k (proportion of top-k retrieved items that are relevant)."""
    if k <= 0:
        return 0.0
    rel_set = set(relevant)
    if not rel_set:
        return 1.0 if not retrieved[:k] else 0.0
    top_k = retrieved[:k]
    num_hits = sum(1 for item in top_k if item in rel_set)
    return num_hits / float(k)


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str] | set[str],
    k: int = 5,
) -> float:
    """Compute Recall @ k (proportion of all relevant items captured in top-k)."""
    if k <= 0:
        return 0.0
    rel_set = set(relevant)
    if not rel_set:
        return 1.0 if not retrieved[:k] else 0.0
    top_k = retrieved[:k]
    num_hits = sum(1 for item in top_k if item in rel_set)
    return num_hits / float(len(rel_set))


def f1_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str] | set[str],
    k: int = 5,
) -> float:
    """Compute F1 @ k (harmonic mean of Precision@k and Recall@k)."""
    prec = precision_at_k(retrieved, relevant, k)
    rec = recall_at_k(retrieved, relevant, k)
    if prec + rec <= 0.0:
        return 0.0
    return 2.0 * (prec * rec) / (prec + rec)


def reciprocal_rank(
    retrieved: Sequence[str],
    relevant: Sequence[str] | set[str],
    k: int | None = None,
) -> float:
    """Compute Reciprocal Rank (1 / rank of first relevant item, or 0.0)."""
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    candidates = retrieved[:k] if k is not None else retrieved
    for rank_idx, item in enumerate(candidates, start=1):
        if item in rel_set:
            return 1.0 / float(rank_idx)
    return 0.0


def average_precision(
    retrieved: Sequence[str],
    relevant: Sequence[str] | set[str],
    k: int | None = None,
) -> float:
    """Compute Average Precision @ k."""
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    candidates = retrieved[:k] if k is not None else retrieved
    hits = 0
    prec_sum = 0.0
    for idx, item in enumerate(candidates, start=1):
        if item in rel_set:
            hits += 1
            prec_sum += hits / float(idx)
    divisor = min(len(rel_set), len(candidates)) if k is not None else len(rel_set)
    if divisor == 0:
        return 0.0
    return prec_sum / float(divisor)


def dcg_at_k(
    retrieved: Sequence[str],
    relevance_scores: dict[str, float],
    k: int = 5,
) -> float:
    """Compute Discounted Cumulative Gain @ k using graded relevance scores."""
    if k <= 0:
        return 0.0
    candidates = retrieved[:k]
    dcg = 0.0
    for idx, item in enumerate(candidates, start=1):
        rel = float(relevance_scores.get(item, 0.0))
        if rel > 0.0:
            dcg += (math.pow(2.0, rel) - 1.0) / math.log2(idx + 1.0)
    return dcg


def idcg_at_k(
    relevance_scores: dict[str, float],
    k: int = 5,
) -> float:
    """Compute Ideal Discounted Cumulative Gain @ k."""
    if k <= 0:
        return 0.0
    sorted_scores = sorted(relevance_scores.values(), reverse=True)
    top_scores = sorted_scores[:k]
    idcg = 0.0
    for idx, rel in enumerate(top_scores, start=1):
        if rel > 0.0:
            idcg += (math.pow(2.0, float(rel)) - 1.0) / math.log2(idx + 1.0)
    return idcg


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance_scores: dict[str, float],
    k: int = 5,
) -> float:
    """Compute Normalized Discounted Cumulative Gain @ k."""
    idcg = idcg_at_k(relevance_scores, k=k)
    if idcg <= 0.0:
        dcg = dcg_at_k(retrieved, relevance_scores, k=k)
        return 1.0 if dcg <= 0.0 else 0.0
    dcg = dcg_at_k(retrieved, relevance_scores, k=k)
    return min(1.0, max(0.0, dcg / idcg))


def _extract_id(item: Any) -> str:
    """Safely extract chunk or doc ID from str, dict, or object."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        cid = (
            item.get("chunk_id")
            or item.get("CHUNK_ID")
            or item.get("doc_id")
            or item.get("DOC_ID")
        )
        return str(cid) if cid is not None else ""
    if hasattr(item, "chunk_id"):
        return str(item.chunk_id)
    if hasattr(item, "doc_id"):
        return str(item.doc_id)
    return str(item)


def _extract_doc_id(chunk_id: str, item: Any = None) -> str:
    """Extract doc_id from chunk object or chunk ID naming convention."""
    if item is not None:
        if isinstance(item, dict) and (item.get("doc_id") or item.get("DOC_ID")):
            return str(item.get("doc_id") or item.get("DOC_ID"))
        if hasattr(item, "doc_id") and getattr(item, "doc_id"):
            return str(getattr(item, "doc_id"))
    if "_c_" in chunk_id:
        return chunk_id.split("_c_")[0]
    return chunk_id


def evaluate_retrieval_query(
    retrieved: Sequence[Any],
    relevant_chunks: Sequence[Any],
    relevant_docs: Sequence[str] | None = None,
    k_values: Sequence[int] = (1, 3, 5, 10),
    query_id: str = "",
    query_text: str = "",
    metadata: dict[str, Any] | None = None,
) -> QueryRetrievalResult:
    """Evaluate retrieval results for a single query across multiple cutoffs."""
    retrieved_ids = [_extract_id(c) for c in retrieved]

    # Build chunk-level ground truth and relevance score map
    chunk_scores: dict[str, float] = {}
    binary_relevant_chunks: list[str] = []

    for c in relevant_chunks:
        cid = _extract_id(c)
        if not cid:
            continue
        score = 1.0
        if isinstance(c, dict):
            raw_score = c.get("RELEVANCE_SCORE", c.get("relevance_score", 1.0))
            try:
                score = float(raw_score)
            except (ValueError, TypeError):
                score = 1.0
        elif hasattr(c, "relevance_score"):
            score = float(getattr(c, "relevance_score", 1.0))

        chunk_scores[cid] = score
        # Include as binary relevant if score > 0
        if score > 0.0:
            binary_relevant_chunks.append(cid)

    # Calculate metrics for each k
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_ids, binary_relevant_chunks, k=k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, binary_relevant_chunks, k=k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, binary_relevant_chunks, k=k)
        metrics[f"f1@{k}"] = f1_at_k(retrieved_ids, binary_relevant_chunks, k=k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, chunk_scores, k=k)

    metrics["mrr"] = reciprocal_rank(retrieved_ids, binary_relevant_chunks)
    metrics["map"] = average_precision(retrieved_ids, binary_relevant_chunks)

    # Document-level retrieval evaluation
    if relevant_docs:
        gold_docs = set(relevant_docs)
        retrieved_docs: list[str] = []
        for idx, cid in enumerate(retrieved_ids):
            chunk_obj = retrieved[idx] if idx < len(retrieved) else None
            doc_id = _extract_doc_id(cid, chunk_obj)
            if doc_id and doc_id not in retrieved_docs:
                retrieved_docs.append(doc_id)

        for k in k_values:
            metrics[f"doc_hit_rate@{k}"] = hit_rate_at_k(retrieved_docs, gold_docs, k=k)
            metrics[f"doc_recall@{k}"] = recall_at_k(retrieved_docs, gold_docs, k=k)
            metrics[f"doc_precision@{k}"] = precision_at_k(retrieved_docs, gold_docs, k=k)
        metrics["doc_mrr"] = reciprocal_rank(retrieved_docs, gold_docs)

    return QueryRetrievalResult(
        query_id=query_id,
        query=query_text,
        retrieved_chunk_ids=retrieved_ids,
        relevant_chunk_ids=binary_relevant_chunks,
        metrics=metrics,
        metadata=metadata or {},
    )
