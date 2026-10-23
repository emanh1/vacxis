"""Generation, citation, refusal, and text-similarity evaluation metrics for RAG answers."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import math
import re
import string
from typing import Any

from vacxis_capstone_fa26.evaluation.schemas import QueryGenerationResult

_CITATION_REGEX = re.compile(r"\[(?:CHUNK_ID:\s*)?([a-zA-Z0-9_\-]+)\]", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str, remove_citations: bool = True) -> str:
    """Normalize text for evaluation: strip citations, lowercase, and clean whitespace."""
    if not text:
        return ""
    if remove_citations:
        text = _CITATION_REGEX.sub(" ", text)
    # Lowercase and strip punctuation
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    # Normalize whitespaces
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """Check if normalized prediction matches normalized reference exactly."""
    norm_pred = normalize_text(prediction)
    norm_ref = normalize_text(reference)
    if not norm_pred and not norm_ref:
        return 1.0
    return 1.0 if norm_pred == norm_ref else 0.0


def token_precision_recall_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    """Compute token-level precision, recall, and F1 (SQuAD style)."""
    norm_pred = normalize_text(prediction)
    norm_ref = normalize_text(reference)

    pred_tokens = norm_pred.split()
    ref_tokens = norm_ref.split()

    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / float(len(pred_tokens))
    recall = num_same / float(len(ref_tokens))
    f1 = (2.0 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _get_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    """Extract n-grams from a token sequence."""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(prediction: str, reference: str, n: int = 1) -> tuple[float, float, float]:
    """Compute ROUGE-N precision, recall, and F1."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens or len(ref_tokens) < n or len(pred_tokens) < n:
        return 0.0, 0.0, 0.0

    pred_ngrams = _get_ngrams(pred_tokens, n)
    ref_ngrams = _get_ngrams(ref_tokens, n)

    overlap = sum((pred_ngrams & ref_ngrams).values())
    pred_total = max(1, sum(pred_ngrams.values()))
    ref_total = max(1, sum(ref_ngrams.values()))

    prec = overlap / float(pred_total)
    rec = overlap / float(ref_total)
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0.0 else 0.0
    return prec, rec, f1


def _lcs_length(seq1: list[str], seq2: list[str]) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(seq1), len(seq2)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def rouge_l(prediction: str, reference: str) -> tuple[float, float, float]:
    """Compute ROUGE-L precision, recall, and F1 via Longest Common Subsequence."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    prec = lcs / float(len(pred_tokens))
    rec = lcs / float(len(ref_tokens))
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0.0 else 0.0
    return prec, rec, f1


def compute_rouge(prediction: str, reference: str) -> dict[str, float]:
    """Compute comprehensive ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)."""
    p1, r1, f1_1 = rouge_n(prediction, reference, n=1)
    p2, r2, f1_2 = rouge_n(prediction, reference, n=2)
    pl, rl, f1_l = rouge_l(prediction, reference)
    return {
        "rouge_1_precision": p1,
        "rouge_1_recall": r1,
        "rouge_1_f1": f1_1,
        "rouge_2_precision": p2,
        "rouge_2_recall": r2,
        "rouge_2_f1": f1_2,
        "rouge_l_precision": pl,
        "rouge_l_recall": rl,
        "rouge_l_f1": f1_l,
    }


def bleu_score(
    prediction: str,
    reference: str,
    max_n: int = 4,
    smooth: bool = True,
) -> dict[str, float]:
    """Compute BLEU-1 to BLEU-4 and cumulative BLEU with brevity penalty and smoothing."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    c = len(pred_tokens)
    r = len(ref_tokens)

    if c == 0 and r == 0:
        return {f"bleu_{i}": 1.0 for i in range(1, max_n + 1)} | {"bleu": 1.0}
    if c == 0 or r == 0:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)} | {"bleu": 0.0}

    # Brevity penalty
    bp = 1.0 if c > r else math.exp(1.0 - (float(r) / float(c)))

    precisions: list[float] = []
    scores: dict[str, float] = {}

    for n in range(1, max_n + 1):
        if c < n:
            p_n = 0.0
        else:
            pred_ngrams = _get_ngrams(pred_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total = sum(pred_ngrams.values())
            if smooth and overlap == 0:
                p_n = 1.0 / (2.0 * total) if total > 0 else 0.0
            else:
                p_n = overlap / float(total) if total > 0 else 0.0

        precisions.append(p_n)
        scores[f"bleu_{n}"] = bp * p_n

    # Cumulative BLEU geometric mean
    if any(p <= 0.0 for p in precisions):
        cum_bleu = 0.0
    else:
        log_mean = sum((1.0 / max_n) * math.log(p) for p in precisions)
        cum_bleu = bp * math.exp(log_mean)

    scores["bleu"] = cum_bleu
    return scores


def citation_precision(
    citations: Sequence[str],
    relevant_chunk_ids: Sequence[str] | set[str],
) -> float:
    """Compute Citation Precision: fraction of generated citations that are relevant."""
    if not citations:
        # If no citations were generated, precision is 1.0 if there were no relevant chunks, else 0.0
        return 1.0 if not relevant_chunk_ids else 0.0
    rel_set = set(relevant_chunk_ids)
    valid_hits = sum(1 for cid in citations if cid in rel_set)
    return valid_hits / float(len(citations))


def citation_recall(
    citations: Sequence[str],
    relevant_chunk_ids: Sequence[str] | set[str],
) -> float:
    """Compute Citation Recall: fraction of ground-truth relevant chunks that were cited."""
    rel_set = set(relevant_chunk_ids)
    if not rel_set:
        return 1.0
    cited_set = set(citations)
    hits = sum(1 for cid in rel_set if cid in cited_set)
    return hits / float(len(rel_set))


def citation_f1(
    citations: Sequence[str],
    relevant_chunk_ids: Sequence[str] | set[str],
) -> float:
    """Compute Citation F1 (harmonic mean of citation precision and recall)."""
    p = citation_precision(citations, relevant_chunk_ids)
    r = citation_recall(citations, relevant_chunk_ids)
    if p + r <= 0.0:
        return 0.0
    return 2.0 * (p * r) / (p + r)


def citation_validity(
    citations: Sequence[str],
    retrieved_chunk_ids: Sequence[str] | set[str],
    invalid_citations: Sequence[str] | None = None,
) -> float:
    """Compute Citation Validity Rate (proportion of citations existing in retrieved context)."""
    inv = list(invalid_citations or [])
    all_citations = list(citations) + inv
    if not all_citations:
        return 1.0
    ret_set = set(retrieved_chunk_ids)
    valid = sum(1 for c in citations if c in ret_set)
    return valid / float(len(all_citations))


def evidence_span_coverage(
    citations: Sequence[str],
    relevant_chunks: Sequence[dict[str, Any]],
) -> float:
    """Fraction of chunks containing evidence spans that were cited."""
    spanned_chunk_ids = [
        c.get("CHUNK_ID") or c.get("chunk_id")
        for c in relevant_chunks
        if isinstance(c, dict) and c.get("EVIDENCE_SPANS")
    ]
    spanned_chunk_ids = [cid for cid in spanned_chunk_ids if cid]
    if not spanned_chunk_ids:
        return 1.0
    cited_set = set(citations)
    covered = sum(1 for cid in spanned_chunk_ids if cid in cited_set)
    return covered / float(len(spanned_chunk_ids))


def refusal_correctness(is_refusal: bool, expected_answerability: str) -> float:
    """Compute whether model refusal matches ground-truth answerability.
    
    1.0 if (is_refusal == True and expected is UNANSWERABLE) or
           (is_refusal == False and expected is ANSWERABLE or PARTIALLY_ANSWERABLE).
    """
    is_unanswerable = expected_answerability.strip().upper() == "UNANSWERABLE"
    return 1.0 if (is_refusal == is_unanswerable) else 0.0


def evaluate_generation_query(
    prediction: str,
    reference: str,
    citations: Sequence[str],
    invalid_citations: Sequence[str] | None = None,
    relevant_chunks: Sequence[Any] | None = None,
    retrieved_chunk_ids: Sequence[str] | None = None,
    is_refusal: bool = False,
    expected_answerability: str = "ANSWERABLE",
    query_id: str = "",
    query_text: str = "",
    latency_seconds: float = 0.0,
    token_usage: dict[str, int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> QueryGenerationResult:
    """Evaluate generation response for a single query."""
    # Extract relevant chunk IDs
    rel_chunk_ids: list[str] = []
    raw_relevant = list(relevant_chunks or [])
    for c in raw_relevant:
        if isinstance(c, str):
            rel_chunk_ids.append(c)
        elif isinstance(c, dict):
            cid = c.get("CHUNK_ID") or c.get("chunk_id")
            score = c.get("RELEVANCE_SCORE", 1)
            try:
                score_num = float(score)
            except (ValueError, TypeError):
                score_num = 1.0
            if cid and score_num > 0:
                rel_chunk_ids.append(str(cid))
        elif hasattr(c, "chunk_id"):
            rel_chunk_ids.append(str(c.chunk_id))

    metrics: dict[str, float] = {}

    # Exact match & token overlap
    metrics["exact_match"] = exact_match(prediction, reference)
    tp, tr, tf1 = token_precision_recall_f1(prediction, reference)
    metrics["token_precision"] = tp
    metrics["token_recall"] = tr
    metrics["token_f1"] = tf1

    # ROUGE
    rouge_dict = compute_rouge(prediction, reference)
    metrics.update(rouge_dict)

    # BLEU
    bleu_dict = bleu_score(prediction, reference)
    metrics.update(bleu_dict)

    # Citations
    metrics["citation_precision"] = citation_precision(citations, rel_chunk_ids)
    metrics["citation_recall"] = citation_recall(citations, rel_chunk_ids)
    metrics["citation_f1"] = citation_f1(citations, rel_chunk_ids)
    metrics["citation_validity"] = citation_validity(
        citations, retrieved_chunk_ids or [], invalid_citations
    )
    metrics["evidence_span_coverage"] = evidence_span_coverage(citations, raw_relevant)

    # Refusal
    metrics["refusal_correctness"] = refusal_correctness(is_refusal, expected_answerability)

    # Latency & tokens
    metrics["latency_seconds"] = latency_seconds
    if token_usage:
        metrics["total_tokens"] = float(token_usage.get("total_tokens", 0))

    return QueryGenerationResult(
        query_id=query_id,
        query=query_text,
        prediction=prediction,
        reference=reference,
        citations=list(citations),
        invalid_citations=list(invalid_citations or []),
        relevant_chunk_ids=rel_chunk_ids,
        is_refusal=is_refusal,
        expected_answerability=expected_answerability,
        metrics=metrics,
        latency_seconds=latency_seconds,
        token_usage=token_usage or {},
        metadata=metadata or {},
    )
