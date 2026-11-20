"""Shared local BGE-M3 dense embedder adapter for LightRAG."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import threading
from typing import Any

import numpy as np

logger = logging.getLogger("vacxis.retrieval.lightrag_embedder")

DENSE_DIM = 1024
DEFAULT_MODEL_PATH = Path("models/bge-m3")
DEFAULT_ENCODE_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 8192

_LOCAL_BGE_M3_ALIASES = frozenset({"baai/bge-m3", "bge-m3", "models/bge-m3"})


def is_local_bge_m3_model(embed_model_name: str) -> bool:
    """Return True when the configured LightRAG embedding name targets local BGE-M3."""
    return embed_model_name.strip().lower() in _LOCAL_BGE_M3_ALIASES


class LocalBGEM3Embedder:
    """Thread-safe local BGEM3FlagModel adapter matching LightRAG's EmbeddingFunc contract.

    Accepts an optional pre-loaded ``embed_model`` instance so ``HybridLightRAGRetriever``
    can share a single BGE-M3 model across both VacxiS Qdrant search and LightRAG queries
    without duplicating ~2.2 GB of GPU memory.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        batch_size: int = DEFAULT_ENCODE_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        *,
        embed_model: Any | None = None,
    ) -> None:
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be positive integers.")
        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self._embed_model = embed_model
        self._lock = threading.Lock()

    def _get_model(self) -> Any:
        """Lazily load BGEM3FlagModel if no shared instance was provided."""
        if self._embed_model is None:
            import torch
            from FlagEmbedding import BGEM3FlagModel

            use_fp16 = torch.cuda.is_available()
            logger.info(
                "Loading local BGE-M3 model from '%s' (use_fp16=%s)...",
                self.model_path,
                use_fp16,
            )
            self._embed_model = BGEM3FlagModel(self.model_path, use_fp16=use_fp16)
            logger.info("Local BGE-M3 model loaded.")
        return self._embed_model

    @property
    def embed_model(self) -> Any:
        """Return the underlying BGE-M3 model, initializing it lazily under lock."""
        with self._lock:
            return self._get_model()

    def encode_sync(self, texts: list[str]) -> np.ndarray:
        """Encode texts synchronously into float32 dense vectors of shape (N, 1024)."""
        if not texts:
            return np.empty((0, DENSE_DIM), dtype=np.float32)

        with self._lock:
            output = self._get_model().encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )

        vectors = np.asarray(output["dense_vecs"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(texts), DENSE_DIM):
            raise ValueError(
                f"Expected embedding shape ({len(texts)}, {DENSE_DIM}), got {vectors.shape}."
            )
        return vectors

    async def __call__(self, texts: list[str], *args: Any, **kwargs: Any) -> np.ndarray:
        """Async callable matching LightRAG EmbeddingFunc signature."""
        return await asyncio.to_thread(self.encode_sync, texts)

    def __deepcopy__(self, memo: dict[int, Any]) -> LocalBGEM3Embedder:
        """Return self during ``dataclasses.asdict`` deepcopy inside ``LightRAG.__post_init__``.

        ``LightRAG._build_global_config()`` calls ``dataclasses.asdict(self)`` which
        deep-copies ``EmbeddingFunc.func`` before discarding the copy and restoring
        ``original_embedding_func``. Returning ``self`` avoids attempting to pickle
        ``self._lock`` (``_thread.lock``) or duplicating the ~2.2 GB ``BGEM3FlagModel``.
        """
        memo[id(self)] = self
        return self

