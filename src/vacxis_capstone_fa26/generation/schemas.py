"""Schema definitions for generation pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from typing import Any, Protocol


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
        default_factory=lambda: os.environ.get("VACXIS_LLM_MODEL", "gpt-4o-mini")
    )
    temperature: float = 0.0
    max_tokens: int = 1024
    enforce_citations: bool = True
    include_metadata: bool = True
    timeout: float = 30.0
    system_prompt_override: str | None = None
    dry_run: bool = False


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

    def to_dict(self) -> dict[str, Any]:
        """Convert answer object to serializable dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "invalid_citations": self.invalid_citations,
            "cited_chunk_ids": [getattr(c, "chunk_id", str(c)) for c in self.cited_chunks],
            "retrieved_chunk_ids": [getattr(c, "chunk_id", str(c)) for c in self.retrieved_chunks],
            "is_refusal": self.is_refusal,
            "model": self.model,
            "latency_seconds": round(self.latency_seconds, 4),
            "usage": self.usage,
        }
