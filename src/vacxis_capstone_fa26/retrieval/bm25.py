"""Dependency-light BM25 retrieval over the existing corpus chunks."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from heapq import nlargest
from math import log1p
import re
from typing import Any
from unicodedata import normalize


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", normalize("NFC", text).casefold())


class BM25Retriever:
    """Rank chunks by lexical overlap with the original question."""

    def __init__(self, chunk_texts: dict[str, str]) -> None:
        self._texts = chunk_texts
        term_counts = {cid: Counter(_tokens(text)) for cid, text in chunk_texts.items()}
        lengths = {cid: sum(counts.values()) for cid, counts in term_counts.items()}
        if not any(lengths.values()):
            raise ValueError("BM25 requires a nonempty, tokenizable chunk corpus.")
        corpus_size = len(lengths)
        average_length = sum(lengths.values()) / corpus_size
        k1, b = 1.2, 0.75
        postings: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for cid, counts in term_counts.items():
            length_penalty = k1 * (1 - b + b * lengths[cid] / average_length)
            for term, count in counts.items():
                postings[term].append((cid, count * (k1 + 1) / (count + length_penalty)))
        self._postings = postings
        self._idf = {
            term: log1p((corpus_size - len(rows) + 0.5) / (len(rows) + 0.5))
            for term, rows in postings.items()
        }

    def retrieve(
        self,
        query: str,
        top_k: int | None = 5,
        *,
        skip_rerank: bool | None = None,
    ) -> list[dict[str, str]]:
        """Return up to top_k matching chunks in descending BM25 score order."""
        limit = 5 if top_k is None else top_k
        scores: dict[str, float] = defaultdict(float)
        for term in dict.fromkeys(_tokens(query)):
            if term in self._idf:
                for cid, weight in self._postings[term]:
                    scores[cid] += self._idf[term] * weight
        ranked_ids = nlargest(limit, scores, key=scores.__getitem__)
        return [{"chunk_id": cid, "text": self._texts[cid]} for cid in ranked_ids]

    def batch_retrieve(
        self,
        queries: Sequence[str],
        top_k: int | None = 5,
        *,
        skip_rerank: bool | None = None,
    ) -> list[list[dict[str, str]]]:
        """Return BM25 rankings for each query in a sequence."""
        return [self.retrieve(query, top_k=top_k, skip_rerank=skip_rerank) for query in queries]

    def close(self) -> None:
        """No-op resource cleanup for context manager compatibility."""

    def __enter__(self) -> BM25Retriever:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
