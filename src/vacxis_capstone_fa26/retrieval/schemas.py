"""Schema definitions for retrieval pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any, Literal

RetrieverMode = Literal["naive", "advanced", "hybrid", "lightrag", "hybrid_lightrag"]
LightRAGQueryMode = Literal["local", "global", "hybrid", "mix", "naive"]
LightRAGFusionMode = Literal["context_append", "rrf_rerank"]


@dataclass
class RetrievedChunk:
    """Represents a chunk retrieved from the vector store."""

    chunk_id: str
    doc_id: str
    doc_title: str
    category: str
    parent_header: str
    header: str
    content_type: str
    text: str
    score: float
    rrf_score: float
    rerank_score: float | None = None
    point_id: int | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for easy serialization."""
        return asdict(self)


@dataclass
class RetrievalConfig:
    """Configuration options for retrieval and reranking pipeline."""

    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "VacxiS_Capstone_FA26")
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    embed_model_path: str | Path = "models/bge-m3"
    reranker_model_path: str | Path = "models/bge-reranker"
    top_k_dense: int = 50
    top_k_sparse: int = 50
    rrf_k: int = 60
    rrf_top_k: int = 30
    top_k: int = 5
    skip_rerank: bool = False
    mode: RetrieverMode = "hybrid"
    merge_table_chunks: bool = False

    # LightRAG Knowledge Graph settings
    lightrag_working_dir: str | Path = "lightrag_storage"
    lightrag_workspace: str = "vacxis"
    lightrag_mode: LightRAGQueryMode = "hybrid"
    lightrag_top_k: int = 20
    lightrag_chunk_top_k: int = 5
    lightrag_max_entity_tokens: int = 1500
    lightrag_max_relation_tokens: int = 1500
    lightrag_include_chunks: bool = True
    lightrag_fusion: LightRAGFusionMode = "context_append"
    chunks_parquet_path: str | Path = "chunks_storage/chunks.parquet"
