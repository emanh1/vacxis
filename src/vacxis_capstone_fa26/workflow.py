"""Connect corpus preparation, RAG and evaluation.

The dataset stage is the testdata CLI (`generate-test-dataset`, vLLM on the
GPU host); this workflow prepares the corpus, then benchmarks the pipeline
against a dataset it finds at ``config.dataset_path``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .evaluation.evaluator import VacxisRAGEvaluator, validate_query_ids
from .generation.generator import MedicalGenerator
from .generation.pipeline import VacxisRAGPipeline
from .generation.schemas import GenerationConfig
from .llm import validate_config
from .retrieval.schemas import RetrievalConfig


@dataclass
class WorkflowConfig:
    """Shared paths and model/store settings for every workflow stage."""

    project_root: Path = field(default_factory=Path.cwd)
    chunks_path: Path = Path("chunks_storage/chunks.parquet")
    dataset_path: Path = Path("test_dataset.json")
    results_path: Path = Path("generation_results.json")
    report_json: Path = Path("evaluation_report.json")
    report_md: Path = Path("evaluation_report.md")
    mapping_path: Path | None = None
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    recreate: bool = False

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        for name in ("chunks_path", "dataset_path", "results_path", "report_json", "report_md", "mapping_path"):
            value = getattr(self, name)
            if value is not None:
                value = Path(value)
                setattr(self, name, value if value.is_absolute() else self.project_root / value)
        if self.retrieval.top_k < 1:
            raise ValueError("top_k must be positive.")
        paths = [self.chunks_path, self.dataset_path, self.results_path, self.report_json, self.report_md]
        if len({p.resolve() for p in paths}) != len(paths):
            raise ValueError("Workflow artifact paths must be distinct.")
        if self.mapping_path is not None and self.mapping_path.resolve() in {p.resolve() for p in paths}:
            raise ValueError("Mapping path cannot be overwritten by a workflow artifact.")


def write_json(path: Path, data: Any) -> None:
    """Write UTF-8 JSON after creating the artifact directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(config: WorkflowConfig, **ingestion_dependencies: Any) -> int:
    """Create the shared Parquet corpus and ingest it into the configured store."""
    from .indexing.main import run_indexing_pipeline
    from .indexing.indexer import ingest_to_qdrant

    path = run_indexing_pipeline(config.project_root, config.chunks_path, config.mapping_path)
    count = ingest_to_qdrant(
        parquet_path=path, model_path=config.retrieval.embed_model_path,
        collection_name=config.retrieval.collection_name,
        qdrant_host=config.retrieval.qdrant_host, qdrant_port=config.retrieval.qdrant_port,
        recreate=config.recreate, **ingestion_dependencies,
    )
    if not count:
        raise ValueError("No chunks were indexed; downstream stages cannot run.")
    return count


def build_pipeline(config: WorkflowConfig) -> VacxisRAGPipeline:
    """Build a planner-free pipeline; websites opt into run_turn separately."""
    if not config.generation.dry_run:
        validate_config(config.generation.model)
    return VacxisRAGPipeline(retrieval_config=config.retrieval,
                             generator=MedicalGenerator(config.generation))


def ask(config: WorkflowConfig, query: str) -> dict[str, Any]:
    """Answer a single nonblank question and release retrieval resources."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be blank.")
    with build_pipeline(config) as pipeline:
        return pipeline.answer(query, top_k=config.retrieval.top_k).to_dict()


def benchmark(config: WorkflowConfig) -> Any:
    """Run batch RAG against golden references and export both metric reports."""
    items = json.loads(config.dataset_path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("Dataset must be a nonempty JSON list.")
    validate_query_ids(items)
    queries = [item.get("QUERY") for item in items]
    if any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError("Every dataset item must have a nonblank QUERY.")
    if any(not isinstance(item.get("REFERENCE_ANSWER"), str)
           or not item["REFERENCE_ANSWER"].strip() for item in items):
        raise ValueError("Every benchmark item must have a REFERENCE_ANSWER.")
    with build_pipeline(config) as pipeline:
        answers = pipeline.answer_batch(queries, top_k=config.retrieval.top_k)
    if len(answers) != len(items):
        raise ValueError("Expected one answer per dataset item.")
    results = [dict(item, GENERATED_ANSWER=answer.to_dict())
               for item, answer in zip(items, answers, strict=True)]
    evaluator = VacxisRAGEvaluator()
    summary = evaluator.evaluate_results(items, results)
    write_json(config.results_path, results)
    evaluator.save_json(summary, config.report_json)
    evaluator.save_markdown(summary, config.report_md)
    return summary


def run(config: WorkflowConfig) -> Any:
    """Prepare the corpus, then benchmark against the dataset at dataset_path.

    The dataset itself is produced by the testdata CLI (`generate-test-dataset`,
    vLLM on the GPU host) and must already exist at ``config.dataset_path``.
    """
    if not config.generation.dry_run:
        validate_config(config.generation.model)
    prepare(config)
    return benchmark(config)
