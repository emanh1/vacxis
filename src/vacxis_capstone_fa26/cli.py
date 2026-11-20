"""Unified command-line entrypoint for VacxiS workflows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import workflow
from .generation.schemas import GenerationConfig
from .indexing.table_pipeline import DEFAULT_TABLE_CACHE_PATH, VALID_TABLE_STRATEGIES
from .llm import LLMError
from .retrieval.schemas import RetrievalConfig

_RETRIEVER_MODES = ("naive", "advanced", "hybrid", "lightrag", "hybrid_lightrag")
_LIGHTRAG_MODES = ("local", "global", "hybrid", "mix", "naive")
_LIGHTRAG_FUSIONS = ("context_append", "rrf_rerank")


def add_lightrag_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose LightRAG Knowledge Graph retrieval arguments on a parser."""
    parser.add_argument(
        "--lightrag-mode",
        choices=_LIGHTRAG_MODES,
        default=RetrievalConfig.lightrag_mode,
        help="LightRAG query mode",
    )
    parser.add_argument(
        "--lightrag-fusion",
        choices=_LIGHTRAG_FUSIONS,
        default=RetrievalConfig.lightrag_fusion,
        help="Fusion strategy when using hybrid_lightrag mode",
    )
    parser.add_argument(
        "--lightrag-working-dir",
        type=Path,
        default=Path(RetrievalConfig.lightrag_working_dir),
        help="LightRAG working directory containing KV storage",
    )
    parser.add_argument(
        "--lightrag-workspace",
        type=str,
        default=RetrievalConfig.lightrag_workspace,
        help="LightRAG workspace name for Qdrant and Neo4j isolation",
    )
    parser.add_argument(
        "--lightrag-top-k",
        type=int,
        default=RetrievalConfig.lightrag_top_k,
        help="Number of graph entities/relationships retrieved by LightRAG",
    )
    parser.add_argument(
        "--lightrag-chunk-top-k",
        type=int,
        default=RetrievalConfig.lightrag_chunk_top_k,
        help="Number of document chunks retrieved by LightRAG",
    )


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
    parser.add_argument(
        "--retriever-mode",
        choices=_RETRIEVER_MODES,
        default=RetrievalConfig.mode,
        help="Retriever backend mode",
    )
    parser.add_argument(
        "--merge-table-chunks",
        action="store_true",
        default=RetrievalConfig.merge_table_chunks,
        help="Merge split table chunks sharing (doc_id, header) before reranking",
    )
    add_lightrag_arguments(parser)


def retrieval_config(args: argparse.Namespace) -> RetrievalConfig:
    """Translate shared CLI arguments into retrieval settings."""
    mode = getattr(args, "retriever_mode", None) or RetrievalConfig.mode
    return RetrievalConfig(
        collection_name=args.collection_name,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        embed_model_path=args.model_path,
        reranker_model_path=args.reranker_path,
        top_k=args.top_k,
        skip_rerank=args.skip_rerank,
        mode=mode,
        merge_table_chunks=bool(
            getattr(args, "merge_table_chunks", RetrievalConfig.merge_table_chunks)
        ),
        lightrag_working_dir=getattr(
            args, "lightrag_working_dir", RetrievalConfig.lightrag_working_dir
        ),
        lightrag_workspace=getattr(
            args, "lightrag_workspace", RetrievalConfig.lightrag_workspace
        ),
        lightrag_mode=getattr(
            args, "lightrag_mode", RetrievalConfig.lightrag_mode
        ),
        lightrag_top_k=getattr(
            args, "lightrag_top_k", RetrievalConfig.lightrag_top_k
        ),
        lightrag_chunk_top_k=getattr(
            args, "lightrag_chunk_top_k", RetrievalConfig.lightrag_chunk_top_k
        ),
        lightrag_fusion=getattr(
            args, "lightrag_fusion", RetrievalConfig.lightrag_fusion
        ),
        chunks_parquet_path=getattr(args, "chunks", RetrievalConfig.chunks_parquet_path),
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
            sub.add_argument(
                "--table-strategy",
                choices=VALID_TABLE_STRATEGIES,
                default="raw",
                help="Table preprocessing strategy during Qdrant indexing",
            )
            sub.add_argument(
                "--table-cache",
                type=Path,
                default=DEFAULT_TABLE_CACHE_PATH,
                help="Path to JSON cache of table prose summaries",
            )
            sub.add_argument(
                "--table-summary-model",
                type=str,
                default=None,
                help="Model key for table summary cache lookup (defaults to LIGHTRAG_LLM)",
            )
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
        if name == "benchmark":
            sub.add_argument("--max-tokens", type=int, default=None,
                             help="Cap LLM output tokens; default: no cap")
            sub.add_argument("--timeout", type=float, default=GenerationConfig.timeout,
                             help="Per-request LLM timeout in seconds (default: "
                                  f"{GenerationConfig.timeout})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generation = GenerationConfig(dry_run=getattr(args, "dry_run", False))
    if getattr(args, "model", None) is not None:
        generation.model = args.model
    if getattr(args, "max_tokens", None) is not None:
        generation.max_tokens = args.max_tokens
    if getattr(args, "timeout", None) is not None:
        generation.timeout = args.timeout
    options = {"project_root": args.project_root, "retrieval": retrieval_config(args),
               "generation": generation}
    for flag, field in (("chunks", "chunks_path"), ("mapping", "mapping_path"),
                        ("dataset", "dataset_path"), ("results", "results_path"),
                        ("report_json", "report_json"), ("report_md", "report_md"),
                        ("recreate", "recreate"), ("table_strategy", "table_strategy"),
                        ("table_cache", "table_cache_path"),
                        ("table_summary_model", "table_summary_model")):
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
