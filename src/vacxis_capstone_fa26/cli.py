"""Unified command-line entrypoint for VacxiS workflows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import workflow
from .generation.schemas import GenerationConfig
from .llm import LLMError
from .retrieval.schemas import RetrievalConfig


def add_retrieval_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the same store and model defaults to orchestration and generation."""
    for flag, default in (
        ("collection-name", RetrievalConfig.collection_name),
        ("qdrant-host", RetrievalConfig.qdrant_host),
        ("model-path", RetrievalConfig.embed_model_path),
        ("reranker-path", RetrievalConfig.reranker_model_path),
    ):
        parser.add_argument("--" + flag, default=default)
    parser.add_argument("--qdrant-port", type=int, default=RetrievalConfig.qdrant_port)


def retrieval_config(args: argparse.Namespace) -> RetrievalConfig:
    """Translate shared CLI arguments into retrieval settings."""
    return RetrievalConfig(
        collection_name=args.collection_name, qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port, embed_model_path=args.model_path,
        reranker_model_path=args.reranker_path, top_k=args.top_k,
        skip_rerank=args.skip_rerank,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VacxiS corpus, RAG and evaluation workflows")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "ask", "benchmark", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--project-root", type=Path, default=Path.cwd())
        add_retrieval_arguments(sub)
        sub.add_argument("--top-k", type=int, default=RetrievalConfig.top_k)
        sub.add_argument("--skip-rerank", action="store_true")
        if name in ("prepare", "run"):
            sub.add_argument("--chunks", type=Path, default=Path("chunks_storage/chunks.parquet"))
            sub.add_argument("--mapping", type=Path)
            sub.add_argument("--recreate", action="store_true")
        if name != "prepare":
            sub.add_argument("--model", help="Override VACXIS_GENERATION_MODEL")
            sub.add_argument("--dry-run", action="store_true", help="Use offline extractive generation")
        if name == "ask":
            sub.add_argument("--query", required=True)
        if name in ("benchmark", "run"):
            sub.add_argument("--dataset", type=Path, default=Path("test_dataset.json"))
            sub.add_argument("--results", type=Path, default=Path("generation_results.json"))
            sub.add_argument("--report-json", type=Path, default=Path("evaluation_report.json"))
            sub.add_argument("--report-md", type=Path, default=Path("evaluation_report.md"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generation = GenerationConfig(dry_run=getattr(args, "dry_run", False))
    if getattr(args, "model", None) is not None:
        generation.model = args.model
    options = {"project_root": args.project_root, "retrieval": retrieval_config(args),
               "generation": generation}
    for flag, field in (("chunks", "chunks_path"), ("mapping", "mapping_path"),
                        ("dataset", "dataset_path"), ("results", "results_path"),
                        ("report_json", "report_json"), ("report_md", "report_md"),
                        ("recreate", "recreate")):
        if hasattr(args, flag):
            options[field] = getattr(args, flag)
    try:
        config = workflow.WorkflowConfig(**options)
        if args.command == "ask":
            print(json.dumps(workflow.ask(config, args.query), ensure_ascii=False, indent=2))
        else:
            result = getattr(workflow, args.command)(config)
            print(f"Indexed {result} chunks." if args.command == "prepare"
                  else f"Evaluated {result.num_queries} queries. Report: {config.report_json}")
    except LLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Workflow {args.command} failed ({type(exc).__name__}). Check inputs and services.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
