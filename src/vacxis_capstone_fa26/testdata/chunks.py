"""Chunk loading and lightweight indexing helpers.

The original generator used ``table.slice(i, 1).to_pylist()`` in a loop, which
is fine for 1188 rows but a known antipattern. We read the whole table once
and precompute a few indices that every other module needs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


@dataclass
class Chunk:
    """Lightweight view over a row in chunks.parquet."""

    chunk_id: str
    doc_id: str
    doc_title: str
    category: str
    text: str
    source_file: str

    @property
    def vaccine_name(self) -> str:
        # doc_title looks like "Hướng dẫn sử dụng Vaccine X"
        prefix = "Hướng dẫn sử dụng Vaccine "
        if self.doc_title.startswith(prefix):
            return self.doc_title[len(prefix):]
        return self.doc_title


def load_chunks(path: Path) -> list[Chunk]:
    """Read chunks.parquet into a list of Chunk objects."""
    table = pq.read_table(str(path))
    chunks: list[Chunk] = []
    for row in table.to_pylist():
        metadata = row.get("metadata") or {}
        chunks.append(
            Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                doc_title=row["doc_title"],
                category=row["category"],
                text=row["text"],
                source_file=metadata.get("source_file", ""),
            )
        )
    return chunks


def index_by_category(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Group chunks by their category."""
    index: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        index[c.category].append(c)
    return dict(index)


def index_by_vaccine(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Group chunks by vaccine name (derived from doc_title)."""
    index: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        index[c.vaccine_name].append(c)
    return dict(index)


def chunk_by_id(chunks: list[Chunk]) -> dict[str, Chunk]:
    """Map chunk_id -> Chunk for O(1) lookup."""
    return {c.chunk_id: c for c in chunks}
