"""Golden dataset and chunk-text loading for the DeepEval module.

Two dataset shapes are accepted:

* the main evaluation dataset produced by ``scripts/generate_eval_dataset.py``
  (``{"metadata": ..., "goldens": [...]}`` where each golden already has
  ``input``/``expected_output``/``context`` in DeepEval form), and
* the flat unanswerable set ``test_dataset.json`` (a list of records with
  ``QUERY``/``REFERENCE_ANSWER``/``RELEVANT_CHUNKS``/... upper-case keys).

``GoldenRecord`` is the normalized view the evaluator consumes. Chunk text is
loaded from ``chunks.parquet`` so a precomputed result's retrieved chunk ids
can be expanded into real context strings for DeepEval's retrieval metrics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

# Stratification axes carried by both dataset shapes, under different keys.
_ANSWERABILITY_KEYS = ("ANSWERABILITY", "answerability")
_COMPLEXITY_KEYS = ("QUERY_COMPLEXITY", "complexity")
_STYLE_KEYS = ("QUERY_STYLE", "query_style")
_DIFFICULTY_KEYS = ("QUERY_DIFFICULTY", "difficulty")
_TYPE_KEYS = ("QUERY_TYPE", "query_type")


@dataclass
class GoldenRecord:
    """Normalized golden question plus its reference context and stratification."""

    query_id: str
    query: str
    expected_output: str = ""
    context: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    answerability: str = ""
    complexity: str = ""
    query_style: str = ""
    difficulty: str = ""
    query_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record to a plain dictionary."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "expected_output": self.expected_output,
            "context": self.context,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "answerability": self.answerability,
            "complexity": self.complexity,
            "query_style": self.query_style,
            "difficulty": self.difficulty,
            "query_type": self.query_type,
        }


def _pick(d: dict[str, Any], keys: tuple[str, ...], default: str = "") -> str:
    """Return the first present key's string value, else ``default``."""
    for key in keys:
        if key in d and d[key] is not None:
            return str(d[key])
    return default


def _extract_relevant_chunk_ids(item: dict[str, Any]) -> list[str]:
    """Extract ground-truth relevant chunk IDs from both deep and flat dataset formats."""
    extra = item.get("additional_metadata")
    if isinstance(extra, dict):
        ids = extra.get("source_chunk_ids") or extra.get("relevant_chunk_ids")
        if isinstance(ids, list) and ids:
            return [str(cid).strip() for cid in ids if str(cid).strip()]

    for key in ("source_chunk_ids", "relevant_chunk_ids", "gold_chunk_ids"):
        ids = item.get(key)
        if isinstance(ids, list) and ids:
            return [str(cid).strip() for cid in ids if str(cid).strip()]

    raw_chunks = item.get("RELEVANT_CHUNKS") or item.get("relevant_chunks")
    if isinstance(raw_chunks, list):
        ids_from_chunks: list[str] = []
        for c in raw_chunks:
            if isinstance(c, dict):
                cid = c.get("CHUNK_ID", c.get("chunk_id"))
                score = c.get("RELEVANCE_SCORE", c.get("score"))
                try:
                    is_rel = (score is None) or (float(score) > 0)
                except (ValueError, TypeError):
                    is_rel = True
                if cid and is_rel:
                    ids_from_chunks.append(str(cid).strip())
            elif isinstance(c, str) and c.strip():
                ids_from_chunks.append(c.strip())
        if ids_from_chunks:
            return ids_from_chunks

    return []


def _context_strings(chunks: Any) -> list[str]:
    """Normalize a reference context to a list of plain text strings.

    Accepts either a list of pre-formatted context strings (main dataset) or
    a list of chunk dicts carrying a ``TEXT`` field (flat dataset).
    """
    texts: list[str] = []
    for chunk in chunks or []:
        if isinstance(chunk, str):
            if chunk.strip():
                texts.append(chunk)
        elif isinstance(chunk, dict):
            text = _pick(chunk, ("TEXT", "text"))
            if text.strip():
                texts.append(text)
    return texts


def to_record(item: dict[str, Any]) -> GoldenRecord:
    """Build a GoldenRecord from either dataset shape (deep or flat)."""
    extra = item.get("additional_metadata") or {}
    merged = {**extra, **item}
    query_id = _pick(
        merged, ("query_id", "QUERY_ID", "id"), default=_pick(item, ("QUERY_ID",), "unnamed")
    )
    if query_id == "unnamed":
        query_id = str(item.get("QUERY_ID") or item.get("query_id") or "unnamed")
    query = _pick(merged, ("input", "QUERY"), default="")
    expected = _pick(merged, ("expected_output", "REFERENCE_ANSWER"), default="")
    context = _context_strings(item.get("context") or item.get("RELEVANT_CHUNKS") or [])
    return GoldenRecord(
        query_id=query_id,
        query=query,
        expected_output=expected,
        context=context,
        relevant_chunk_ids=_extract_relevant_chunk_ids(item),
        answerability=_pick(merged, _ANSWERABILITY_KEYS),
        complexity=_pick(merged, _COMPLEXITY_KEYS),
        query_style=_pick(merged, _STYLE_KEYS),
        difficulty=_pick(merged, _DIFFICULTY_KEYS),
        query_type=_pick(merged, _TYPE_KEYS),
    )


def load_goldens(path: str | Path) -> list[GoldenRecord]:
    """Load a golden dataset file (main or flat shape) into normalized records."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "goldens" in payload:
        raw_items: list[dict[str, Any]] = payload["goldens"]
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raise ValueError(f"Unrecognized dataset shape in {path}: expected a list or a goldens object.")
    return [to_record(item) for item in raw_items]


def load_chunk_texts(path: str | Path) -> dict[str, str]:
    """Read a chunk_id -> text map from a chunks Parquet file.

    Returns an empty map when the file is missing, so evaluation can still run
    (retrieval context will be empty in that case) rather than hard-failing.
    """
    chunk_path = Path(path)
    if not chunk_path.is_file():
        return {}
    table = pq.read_table(str(chunk_path), columns=["chunk_id", "text"])
    text_mask = pc.is_valid(table["text"])
    table = table.filter(text_mask)
    texts: dict[str, str] = {}
    for row in table.to_pylist():
        raw = row.get("text") or ""
        if raw.strip():
            texts[str(row["chunk_id"])] = raw.strip()
    return texts


def retrieval_context_from_answer(
    answer: dict[str, Any],
    chunk_texts: dict[str, str],
) -> list[str]:
    """Expand a precomputed result's retrieved chunks into context strings.

    Prefers explicit ``retrieved_contexts`` stored on the answer (preserving
    table-aware summaries or graph contexts passed to the generator) and falls
    back to hydrating ``retrieved_chunk_ids`` from ``chunk_texts``.
    """
    raw_contexts = answer.get("retrieved_contexts")
    if isinstance(raw_contexts, list):
        contexts = [t.strip() for t in raw_contexts if isinstance(t, str) and t.strip()]
        if contexts:
            return contexts

    chunk_ids = answer.get("retrieved_chunk_ids") or []
    context: list[str] = []
    for chunk_id in chunk_ids:
        text = chunk_texts.get(str(chunk_id))
        if text:
            context.append(text)
    return context


def answer_fields(answer: dict[str, Any]) -> dict[str, Any]:
    """Extract the generation fields DeepEval test cases need from a result.

    Accepts the flat ``GENERATED_ANSWER`` dict the generation CLI writes
    (``answer``, ``citations``, ``is_refusal``) or an already-unwrapped dict.
    """
    payload = answer.get("GENERATED_ANSWER", answer) if isinstance(answer, dict) else {}
    fields: dict[str, Any] = {
        "actual_output": str(payload.get("answer", "")),
        "is_refusal": bool(payload.get("is_refusal", False)),
        "retrieved_chunk_ids": [str(c) for c in (payload.get("retrieved_chunk_ids") or [])],
    }
    raw_contexts = payload.get("retrieved_contexts")
    if isinstance(raw_contexts, list):
        fields["retrieved_contexts"] = [
            t.strip() for t in raw_contexts if isinstance(t, str) and t.strip()
        ]
    raw_aliases = payload.get("retrieved_chunk_aliases")
    if isinstance(raw_aliases, dict):
        normalized_aliases: dict[str, list[str]] = {}
        for key, vals in raw_aliases.items():
            key_str = str(key).strip()
            if not key_str or not isinstance(vals, (list, tuple)):
                continue
            cleaned_vals = [str(v).strip() for v in vals if str(v).strip()]
            if cleaned_vals:
                normalized_aliases[key_str] = cleaned_vals
        if normalized_aliases:
            fields["retrieved_chunk_aliases"] = normalized_aliases
    return fields
