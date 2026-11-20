"""Schema definitions for generation pipeline."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
import os
from typing import Any, Protocol


from vacxis_capstone_fa26.resilience import RetryPolicy


class ChunkLike(Protocol):
    """Protocol for chunk objects passed into generator."""

    chunk_id: str
    text: str
    doc_title: str
    category: str
    header: str


@dataclass
class GenerationConfig:
    """Configuration options for generation and LLM inference."""

    model: str = field(
        default_factory=lambda: os.getenv("VACXIS_GENERATION_MODEL", "").strip()
    )
    temperature: float = 0.0
    max_tokens: int | None = None
    enforce_citations: bool = True
    include_metadata: bool = True
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    timeout: float = 120.0
    system_prompt_override: str | None = None
    dry_run: bool = False


def _extract_chunk_id(c: Any) -> str:
    """Safely extract chunk ID from an object or dictionary."""
    if isinstance(c, dict):
        cid = c.get("chunk_id") or c.get("CHUNK_ID")
        return str(cid) if cid is not None else str(c)
    if hasattr(c, "chunk_id"):
        return str(c.chunk_id)
    return str(c)


def _extract_chunk_text(c: Any) -> str:
    """Safely extract nonblank chunk text from an object or dictionary."""
    if isinstance(c, dict):
        text = c.get("text")
        if text is None:
            text = c.get("TEXT")
        return str(text).strip() if text is not None else ""
    if hasattr(c, "text") and c.text is not None:
        return str(c.text).strip()
    return ""


def extract_chunk_aliases(chunks: Sequence[Any]) -> dict[str, list[str]]:
    """Extract primary chunk_id -> merged_chunk_ids aliases for merged table chunks.

    Only records chunks whose ``merged_chunk_ids`` contains at least one source ID
    distinct from its primary ``chunk_id``, keeping outputs empty for unmerged chunks.
    """
    aliases: dict[str, list[str]] = {}
    for chunk in chunks:
        cid = _extract_chunk_id(chunk).strip()
        if not cid:
            continue
        meta = chunk.get("metadata") if isinstance(chunk, dict) else getattr(chunk, "metadata", None)
        raw_merged = meta.get("merged_chunk_ids") if isinstance(meta, dict) else None
        if raw_merged is None and isinstance(chunk, dict):
            raw_merged = chunk.get("merged_chunk_ids")
        if not isinstance(raw_merged, (list, tuple)):
            continue
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in raw_merged:
            item_str = str(item).strip()
            if item_str and item_str not in seen:
                seen.add(item_str)
                cleaned.append(item_str)
        if cid not in seen:
            cleaned.insert(0, cid)
        if any(item != cid for item in cleaned):
            aliases[cid] = cleaned
    return aliases


@dataclass
class GeneratedAnswer:
    """Represents an answer synthesized by the generator."""

    query: str
    answer: str
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    cited_chunks: list[Any] = field(default_factory=list)
    retrieved_chunks: list[Any] = field(default_factory=list)
    is_refusal: bool = False
    model: str = ""
    latency_seconds: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    query_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert answer object to serializable dictionary."""
        retrieved_contexts = [
            text
            for c in self.retrieved_chunks
            if (text := _extract_chunk_text(c))
        ]
        result = {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "invalid_citations": self.invalid_citations,
            "cited_chunk_ids": [_extract_chunk_id(c) for c in self.cited_chunks],
            "retrieved_chunk_ids": [_extract_chunk_id(c) for c in self.retrieved_chunks],
            "retrieved_contexts": retrieved_contexts,
            "is_refusal": self.is_refusal,
            "model": self.model,
            "latency_seconds": round(self.latency_seconds, 4),
            "usage": self.usage,
        }
        aliases = extract_chunk_aliases(self.retrieved_chunks)
        if aliases:
            result["retrieved_chunk_aliases"] = aliases
        if self.query_context:
            result["query_context"] = self.query_context
        return result
