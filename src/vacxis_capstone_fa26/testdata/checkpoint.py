"""Durable store for the expensive vLLM stages (rewrites, masks, answers).

A single JSON file lets a crashed or killed run (Colab session timeout, GPU
OOM) resume instead of regenerating everything. All stages are keyed by
the stable source-query ``id`` (never by ``QUERY_ID``, which depends on
the shuffle seed), so a ``--limit 50`` smoke run warms the same
checkpoint for the full 1200-query run.

A ``meta`` header (model, seed, prompt version) is checked on load: a
checkpoint from a different model/seed/prompt is rejected loudly rather
than mixed into a new run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from .vllm_client import VLLMError


StoreName = Literal["rewrites", "masks", "answers"]


class VLLMCheckpoint:
    """JSON checkpoint with one sub-store per LLM stage."""

    _STORES: tuple[str, ...] = ("rewrites", "masks", "answers")

    def __init__(
        self,
        path: Path,
        *,
        model: str,
        seed: int | None,
        prompt_version: int,
    ) -> None:
        self.path = Path(path)
        self.meta = {"model": model, "seed": seed, "prompt_version": prompt_version}
        self.stores: dict[str, dict[str, Any]] = {
            name: {} for name in self._STORES
        }

    @staticmethod
    def _key(source_id: Any) -> str:
        return str(source_id)

    def load(self) -> None:
        """Load an existing checkpoint; raise on incompatible meta."""
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        meta = data.get("meta") or {}
        for key, want in self.meta.items():
            if key in meta and meta[key] != want:
                raise VLLMError(
                    f"Checkpoint {self.path} was written with {key}={meta[key]!r}, "
                    f"but this run uses {key}={want!r}. Use a fresh --checkpoint "
                    "path or re-run without it."
                )
        for name in self._STORES:
            rows = data.get(name) or {}
            self.stores[name] = {self._key(k): v for k, v in rows.items()}
        counts = ", ".join(
            f"{n} {len(self.stores[n])}" for n in self._STORES
        )
        print(f"  -> checkpoint resumed: {counts}")

    def save(self) -> None:
        """Atomically persist the current caches (tmp file + rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": self.meta}
        payload.update(self.stores)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, store: StoreName, key: Any) -> Any:
        """Return the cached value for ``key`` in ``store`` (None if absent)."""
        return self.stores[store].get(self._key(key))

    def put(self, store: StoreName, key: Any, value: Any) -> None:
        """Cache ``value`` under ``key`` in ``store``."""
        self.stores[store][self._key(key)] = value
