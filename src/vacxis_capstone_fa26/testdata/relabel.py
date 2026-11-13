"""Post-mask re-labelling and deterministic record verification.

The symptom masker rewrites the *query* of a record but leaves the fields
derived from the old query untouched: ``ANSWERABILITY``,
``REFERENCE_ANSWER`` and ``CLAIM_EVIDENCE_MAP`` still describe the
pre-masking symptoms. A record can therefore carry a label and an answer
that do not match the question it now asks.

This module closes that gap in two steps:

* :func:`apply_verification` writes a verifier's output back into the
  record: the answerability label, the reference answer, the claim/evidence
  map and the evidence completeness.
* :func:`verify_record` checks the result against invariants that need no
  model, and reports whether the record may be kept. A record that violates
  an invariant is dropped rather than shipped with a broken claim map.

The invariants mirror the ones the LLM review of the dataset found broken:
a citation must resolve to a relevant chunk, and an evidence quote must be
a verbatim excerpt of the chunk it cites.
"""
from __future__ import annotations

import re
from typing import Any

ANSWERABILITY_VALUES: frozenset[str] = frozenset(
    {"ANSWERABLE", "PARTIALLY_ANSWERABLE", "UNANSWERABLE"}
)

# Evidence completeness follows the label: an unanswerable record has no
# evidence to be complete about.
COMPLETENESS_BY_ANSWERABILITY: dict[str, str] = {
    "ANSWERABLE": "COMPLETE",
    "PARTIALLY_ANSWERABLE": "PARTIAL",
    "UNANSWERABLE": "NOT_ASSESSED",
}

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_\-]+)\]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text or "").strip()


def relevant_chunks(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The record's chunks with ``RELEVANCE_SCORE >= 3``."""
    return [
        chunk
        for chunk in record.get("RELEVANT_CHUNKS") or []
        if isinstance(chunk, dict) and chunk.get("RELEVANCE_SCORE", 0) >= 3
    ]


def apply_verification(
    record: dict[str, Any],
    *,
    answerability: str,
    reference_answer: str,
    claim_evidence_map: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write a verifier's output into a record and refresh derived fields.

    The answerability label decides whether the record keeps its evidence:
    an unanswerable record carries no chunks, because the evidence does not
    support its question.
    """
    label = answerability if answerability in ANSWERABILITY_VALUES else "ANSWERABLE"
    updated = dict(record)
    updated["ANSWERABILITY"] = label
    updated["REFERENCE_ANSWER"] = reference_answer
    updated["CLAIM_EVIDENCE_MAP"] = list(claim_evidence_map or [])
    updated["EVIDENCE_COMPLETENESS"] = COMPLETENESS_BY_ANSWERABILITY[label]

    if label == "UNANSWERABLE":
        updated["RELEVANT_CHUNKS"] = []
        updated["RELEVANT_DOCS"] = []
    return updated


def verify_record(record: dict[str, Any]) -> tuple[bool, str]:
    """Check a record against the invariants. Returns ``(keep, reason)``.

    The checks are:

    1. the answerability label is a known value;
    2. an unanswerable record carries no relevant chunks, and an answerable
       one carries at least one;
    3. every ``[CHUNK_ID]`` in the reference answer resolves to one of the
       record's relevant chunks;
    4. every claim cites a relevant chunk;
    5. every evidence quote is a verbatim excerpt of the chunk it cites.
    """
    label = str(record.get("ANSWERABILITY") or "")
    if label not in ANSWERABILITY_VALUES:
        return False, f"ANSWERABILITY={label!r} is not a known label"

    chunks = relevant_chunks(record)
    if label == "UNANSWERABLE":
        if chunks:
            return False, "UNANSWERABLE record still carries relevant chunks"
    elif not chunks:
        return False, f"{label} record has no relevant chunk"

    texts = {
        str(chunk.get("CHUNK_ID", "")): str(chunk.get("TEXT", "") or "")
        for chunk in chunks
    }

    answer = str(record.get("REFERENCE_ANSWER") or "")
    unknown = sorted(set(CITATION_PATTERN.findall(answer)) - set(texts))
    if unknown:
        return False, f"reference answer cites unknown chunk(s): {', '.join(unknown)}"

    for index, claim in enumerate(record.get("CLAIM_EVIDENCE_MAP") or []):
        if not isinstance(claim, dict):
            return False, f"claim #{index + 1} is not an object"
        chunk_id = str(claim.get("chunk_id", "")).strip()
        if chunk_id not in texts:
            return False, f"claim #{index + 1} cites unknown chunk {chunk_id!r}"
        quote = _normalize(str(claim.get("evidence_quote", "")))
        if quote and quote not in _normalize(texts[chunk_id]):
            return False, f"claim #{index + 1} quote is not verbatim in {chunk_id}"

    return True, "ok"
