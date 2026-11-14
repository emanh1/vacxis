"""Shared vLLM engine wrapper for the testdata LLM stages.

vLLM is *never* imported at module load time, so the rest of the project
stays usable on CPU-only machines without vLLM installed; the engine is
loaded lazily on first use. One :class:`VLLMClient` is constructed per run
and shared by every stage, so the model is loaded exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

# Lazy, optional vLLM import. vLLM (and torch) are only present in the GPU
# environment (Colab/Kaggle).
try:  # pragma: no cover - exercised only where vLLM is installed
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover
    LLM = None
    SamplingParams = None


# Canonical Qwen instruct model. Overridable via --vllm-model.
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class VLLMError(RuntimeError):
    """Raised when vLLM is requested but unavailable or misconfigured."""


@dataclass
class VLLMClient:
    """Holds the lazily-constructed vLLM engine and sampling settings."""

    model: str = DEFAULT_VLLM_MODEL
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    dtype: str = "float16"
    seed: int | None = None
    llm: Any = None

    def ensure_loaded(self) -> None:
        """Load the model on first use; raise a clear error if no vLLM."""
        if self.llm is not None:
            return
        if LLM is None:
            raise VLLMError(
                "vLLM is not installed. Install it on a GPU host "
                "(Colab/Kaggle) with: uv pip install vllm."
            )
        self.llm = LLM(
            model=self.model,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            trust_remote_code=True,
        )

    def sample(self, **overrides: Any) -> Any:
        if SamplingParams is None:
            raise VLLMError("vLLM SamplingParams unavailable; is vllm installed?")
        kwargs: dict[str, Any] = {"temperature": 0.0, "max_tokens": 512}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        kwargs.update(overrides)
        return SamplingParams(**kwargs)

    def render(self, messages: list[dict[str, str]]) -> str:
        """Render a conversation to a chat-template string for ``generate``.

        ``LLM.generate`` in recent vLLM versions only accepts plain strings
        (or dicts whose ``"prompt"`` is a string), and the convenience
        ``LLM.apply_chat_template`` method has since been removed. The
        tokenizer-level ``apply_chat_template`` is stable across versions,
        so we render through the engine's tokenizer instead.
        """
        self.ensure_loaded()
        rendered = self._tokenizer().apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        # TokenizerPool-style tokenizers return a list for multiple
        # conversations; we always pass one, so a bare string is expected.
        return rendered[0] if isinstance(rendered, list) else rendered

    def _tokenizer(self) -> Any:
        """Resolve the engine's tokenizer across vLLM versions."""
        llm = self.llm
        get_tokenizer = getattr(llm, "get_tokenizer", None)
        if callable(get_tokenizer):
            return get_tokenizer()
        if hasattr(llm, "tokenizer"):
            return llm.tokenizer
        engine = getattr(llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "tokenizer"):
            return engine.tokenizer
        raise VLLMError(
            "Could not locate a tokenizer on the loaded vLLM engine; "
            "update vLLM or pass a chat-template string manually."
        )


class Rewriter(Protocol):
    """Structural type: maps source queries to grounded vaccine queries."""

    def rewrite(self, source_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return one result dict per input query, preserving order.

        Each result carries at least: ``id``, ``vaccine``,
        ``rewritten_query``, ``query_type``, ``answerability``.
        """
        ...
