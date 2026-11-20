"""Deterministic Information Retrieval (IR) evaluation metrics.

Provides standard rank-based IR metrics (Hit@K, Recall@K, Precision@K,
MRR@K, NDCG@K, MAP@K) for measuring retrieval quality against ground-truth
relevant chunk IDs without requiring an LLM judge.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from vacxis_capstone_fa26.evaluation.schemas import MetricAggregate

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


def _dedup_clean(items: Sequence[Any]) -> list[str]:
    """Deduplicate and normalize string identifiers while preserving order."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        cid = str(item).strip()
        if cid and cid not in seen:
            seen.add(cid)
            cleaned.append(cid)
    return cleaned


def _normalize_aliases(
    chunk_aliases: Mapping[str, Sequence[str]] | None,
) -> dict[str, set[str]]:
    """Normalize primary chunk_id -> covered source chunk_ids mapping."""
    if not chunk_aliases:
        return {}
    normalized: dict[str, set[str]] = {}
    for raw_key, raw_vals in chunk_aliases.items():
        key = str(raw_key).strip()
        if not key:
            continue
        covered = {key}
        if isinstance(raw_vals, (list, tuple, set, frozenset)):
            for val in raw_vals:
                cleaned = str(val).strip()
                if cleaned:
                    covered.add(cleaned)
        normalized[key] = covered
    return normalized


def _evaluate_rank_slots(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
    chunk_aliases: Mapping[str, Sequence[str]] | None = None,
) -> tuple[int, int, int, int | None, float, float, int]:
    """Evaluate ranked slots 1..k with optional rank-preserving merged-chunk alias matching.

    Returns:
        (hit_slots, covered_relevant_count, total_relevant, first_hit_rank,
         dcg, sum_precisions, collapsed_relevant)
    """
    relevant_set = {str(c).strip() for c in relevant if str(c).strip()}
    total_relevant = len(relevant_set)
    if k <= 0 or not retrieved or total_relevant == 0:
        return 0, 0, total_relevant, None, 0.0, 0.0, 0

    top_k = _dedup_clean(retrieved)[:k]
    aliases = _normalize_aliases(chunk_aliases)

    seen_rel: set[str] = set()
    hit_slots = 0
    first_hit_rank: int | None = None
    dcg = 0.0
    sum_precisions = 0.0
    collapsed_relevant = 0

    for idx, cid in enumerate(top_k, start=1):
        covered = aliases.get(cid, {cid})
        newly = (covered & relevant_set) - seen_rel
        if newly:
            seen_rel |= newly
            hit_slots += 1
            collapsed_relevant += len(newly) - 1
            if first_hit_rank is None:
                first_hit_rank = idx
            dcg += 1.0 / math.log2(idx + 1)
            sum_precisions += hit_slots / float(idx)

    return (
        hit_slots,
        len(seen_rel),
        total_relevant,
        first_hit_rank,
        dcg,
        sum_precisions,
        collapsed_relevant,
    )


def hit_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
    *,
    chunk_aliases: Mapping[str, Sequence[str]] | None = None,
) -> float:
    """Return 1.0 if any relevant chunk is covered in retrieved[:k], else 0.0."""
    hit_slots, *_ = _evaluate_rank_slots(retrieved, relevant, k, chunk_aliases)
    return 1.0 if hit_slots > 0 else 0.0


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
    *,
    chunk_aliases: Mapping[str, Sequence[str]] | None = None,
) -> float:
    """Fraction of ground-truth relevant chunks covered by retrieved[:k]."""
    _, covered_count, total_rel, *_ = _evaluate_rank_slots(
        retrieved, relevant, k, chunk_aliases
    )
    return covered_count / float(total_rel) if total_rel > 0 else 0.0


def precision_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
    *,
    chunk_aliases: Mapping[str, Sequence[str]] | None = None,
) -> float:
    """Fraction of retrieved[:k] rank slots covering at least one new relevant chunk."""
    if k <= 0:
        return 0.0
    hit_slots, *_ = _evaluate_rank_slots(retrieved, relevant, k, chunk_aliases)
    return hit_slots / float(k)


def mrr_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
    *,
    chunk_aliases: Mapping[str, Sequence[str]] | None = None,
) -> float:
    """Mean Reciprocal Rank: reciprocal rank of the first relevant slot in top k."""
    _, _, _, first_rank, *_ = _evaluate_rank_slots(retrieved, relevant, k, chunk_aliases)
    return 1.0 / float(first_rank) if first_rank is not None else 0.0


def map_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
    *,
    chunk_aliases: Mapping[str, Sequence[str]] | None = None,
) -> float:
    """Mean Average Precision at k with rank-preserving alias matching."""
    _, _, total_rel, _, _, sum_precisions, collapsed = _evaluate_rank_slots(
        retrieved, relevant, k, chunk_aliases
    )
    if total_rel == 0:
        return 0.0
    effective_rel = max(1, total_rel - collapsed)
    return sum_precisions / float(effective_rel)


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
    *,
    chunk_aliases: Mapping[str, Sequence[str]] | None = None,
) -> float:
    """Normalized Discounted Cumulative Gain at k with binary slot relevance."""
    _, _, total_rel, _, dcg, _, collapsed = _evaluate_rank_slots(
        retrieved, relevant, k, chunk_aliases
    )
    if total_rel == 0 or dcg <= 0.0:
        return 0.0
    effective_rel = max(1, total_rel - collapsed)
    ideal_count = min(k, effective_rel)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_count + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def compute_query_ir_metrics(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    *,
    chunk_aliases: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, float]:
    """Compute all standard IR metrics for a single query across cutoff values."""
    scores: dict[str, float] = {}
    for k in k_values:
        scores[f"Hit@{k}"] = hit_at_k(retrieved, relevant, k, chunk_aliases=chunk_aliases)
        scores[f"Recall@{k}"] = recall_at_k(retrieved, relevant, k, chunk_aliases=chunk_aliases)
        scores[f"Precision@{k}"] = precision_at_k(retrieved, relevant, k, chunk_aliases=chunk_aliases)
        scores[f"MRR@{k}"] = mrr_at_k(retrieved, relevant, k, chunk_aliases=chunk_aliases)
        scores[f"NDCG@{k}"] = ndcg_at_k(retrieved, relevant, k, chunk_aliases=chunk_aliases)
        scores[f"MAP@{k}"] = map_at_k(retrieved, relevant, k, chunk_aliases=chunk_aliases)
    return scores


def aggregate_ir_metrics(
    per_query_metrics: Sequence[dict[str, float]],
    metric_names: Sequence[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, MetricAggregate]:
    """Aggregate per-query IR metrics into MetricAggregate instances."""
    if not per_query_metrics:
        return {}
    if metric_names is None:
        metric_names = list(per_query_metrics[0].keys())

    aggregates: dict[str, MetricAggregate] = {}
    for name in metric_names:
        values: list[float] = []
        passed = 0
        for item in per_query_metrics:
            if name in item:
                val = float(item[name])
                values.append(val)
                if val >= threshold:
                    passed += 1
        count = len(values)
        if count == 0:
            continue
        mean_val = sum(values) / count
        aggregates[name] = MetricAggregate(
            mean=mean_val,
            min=min(values),
            max=max(values),
            count=count,
            pass_rate=passed / count,
        )
    return aggregates
