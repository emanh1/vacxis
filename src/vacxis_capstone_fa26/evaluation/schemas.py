"""Schema definitions for VacxiS evaluation metrics and reports."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QueryRetrievalResult:
    """Individual query retrieval evaluation results."""

    query_id: str
    query: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


@dataclass
class QueryGenerationResult:
    """Individual query generation evaluation results."""

    query_id: str
    query: str
    prediction: str
    reference: str
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    is_refusal: bool = False
    expected_answerability: str = "ANSWERABLE"
    metrics: dict[str, float] = field(default_factory=dict)
    latency_seconds: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


@dataclass
class MetricAggregate:
    """Statistical summary (mean, std, min, max, count) for a metric."""

    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert aggregate to dictionary."""
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "count": self.count,
        }


@dataclass
class EvaluationSummary:
    """Aggregated evaluation report across all queries and subsets."""

    task: str  # "retrieval", "generation", or "rag"
    num_queries: int = 0
    overall_metrics: dict[str, float] = field(default_factory=dict)
    metric_aggregates: dict[str, MetricAggregate] = field(default_factory=dict)
    by_query_type: dict[str, dict[str, float]] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    by_complexity: dict[str, dict[str, float]] = field(default_factory=dict)
    by_answerability: dict[str, dict[str, float]] = field(default_factory=dict)
    per_query_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete summary to dictionary."""
        return {
            "task": self.task,
            "num_queries": self.num_queries,
            "overall_metrics": {k: round(v, 4) for k, v in self.overall_metrics.items()},
            "metric_aggregates": {k: v.to_dict() for k, v in self.metric_aggregates.items()},
            "by_query_type": {
                k: {mk: round(mv, 4) for mk, mv in v.items()}
                for k, v in self.by_query_type.items()
            },
            "by_difficulty": {
                k: {mk: round(mv, 4) for mk, mv in v.items()}
                for k, v in self.by_difficulty.items()
            },
            "by_complexity": {
                k: {mk: round(mv, 4) for mk, mv in v.items()}
                for k, v in self.by_complexity.items()
            },
            "by_answerability": {
                k: {mk: round(mv, 4) for mk, mv in v.items()}
                for k, v in self.by_answerability.items()
            },
            "metadata": self.metadata,
            "per_query_results": self.per_query_results,
        }

    def render_markdown(self) -> str:
        """Render a clean Markdown summary table."""
        lines: list[str] = []
        lines.append(f"# VacxiS Evaluation Report: {self.task.upper()}")
        lines.append(f"Total evaluated queries: {self.num_queries}")
        lines.append("")
        lines.append("## Overall Metrics")
        lines.append("| Metric | Mean | Std | Min | Max |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for k, v in sorted(self.metric_aggregates.items()):
            lines.append(f"| {k} | {v.mean:.4f} | {v.std:.4f} | {v.min:.4f} | {v.max:.4f} |")

        if self.by_query_type:
            lines.append("")
            lines.append("## Breakdown by Query Type")
            first_metrics = sorted(list(next(iter(self.by_query_type.values())).keys()))[:6]
            header = "| Query Type | Count | " + " | ".join(first_metrics) + " |"
            sep = "| :--- | :--- | " + " | ".join([":---" for _ in first_metrics]) + " |"
            lines.append(header)
            lines.append(sep)
            for qtype, scores in sorted(self.by_query_type.items()):
                cnt = scores.get("count", 0)
                row_vals = [f"{scores.get(m, 0.0):.4f}" for m in first_metrics]
                lines.append(f"| {qtype} | {int(cnt)} | " + " | ".join(row_vals) + " |")

        if self.by_difficulty:
            lines.append("")
            lines.append("## Breakdown by Difficulty")
            first_metrics = sorted(list(next(iter(self.by_difficulty.values())).keys()))[:6]
            header = "| Difficulty | Count | " + " | ".join(first_metrics) + " |"
            sep = "| :--- | :--- | " + " | ".join([":---" for _ in first_metrics]) + " |"
            lines.append(header)
            lines.append(sep)
            for diff, scores in sorted(self.by_difficulty.items()):
                cnt = scores.get("count", 0)
                row_vals = [f"{scores.get(m, 0.0):.4f}" for m in first_metrics]
                lines.append(f"| {diff} | {int(cnt)} | " + " | ".join(row_vals) + " |")

        return "\n".join(lines)
