"""Validation and deterministic label application for the generated dataset.

Two layers:

* :class:`ValidationReport` / :func:`validate` -- corpus-level checks over a
  produced query list (hard failures vs soft warnings).
* record invariants (``verify_record``) and the deterministic writer
  ``apply_labels`` -- keep a record's derived fields (label, answer, claim
  map, evidence completeness) consistent after any stage rewrites the
  query.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .chunks import Chunk


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


def apply_labels(
    record: dict[str, Any],
    *,
    answerability: str,
    reference_answer: str,
    claim_evidence_map: list[dict[str, Any]] | None = None,
) -> None:
    """Write a stage's verdict into a record, in place.

    The answerability label decides whether the record keeps its evidence:
    an unanswerable record carries no chunks, because the evidence does not
    support its question.
    """
    label = answerability if answerability in ANSWERABILITY_VALUES else "ANSWERABLE"
    record["ANSWERABILITY"] = label
    record["REFERENCE_ANSWER"] = reference_answer
    record["CLAIM_EVIDENCE_MAP"] = list(claim_evidence_map or [])
    record["EVIDENCE_COMPLETENESS"] = COMPLETENESS_BY_ANSWERABILITY[label]

    if label == "UNANSWERABLE":
        record["RELEVANT_CHUNKS"] = []
        record["RELEVANT_DOCS"] = []


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


@dataclass
class ValidationReport:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        lines: list[str] = []
        lines.append(
            f"Validation: {'OK' if self.ok else f'FAIL ({len(self.failures)})'}; "
            f"warnings={len(self.warnings)}; stats={self.stats}"
        )
        for w in self.warnings:
            lines.append(f"  WARN  {w}")
        for f in self.failures:
            lines.append(f"  FAIL  {f}")
        return "\n".join(lines)


def _chunks_by_id(chunks: Iterable[Chunk]) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}


def validate(
    queries: list[dict],
    chunks: list[Chunk],
    by_category: dict[str, list[Chunk]],
    expected_per_type: int,
    expected_types: list[str],
) -> ValidationReport:
    rpt = ValidationReport()
    by_id = _chunks_by_id(chunks)
    type_counts: Counter[str] = Counter()

    for q in queries:
        qid = q.get("QUERY_ID", "<missing>")
        qtype = q.get("QUERY_TYPE")
        type_counts[qtype] += 1

        relevant = q.get("RELEVANT_CHUNKS", [])
        answerability = q.get("ANSWERABILITY")

        if answerability == "UNANSWERABLE":
            if relevant:
                rpt.failures.append(
                    f"{qid} UNANSWERABLE but RELEVANT_CHUNKS not empty"
                )
            continue

        if not relevant:
            rpt.failures.append(
                f"{qid} {answerability} but RELEVANT_CHUNKS is empty"
            )
            continue

        primary = relevant[0]
        cid = primary.get("CHUNK_ID")
        if cid not in by_id:
            rpt.failures.append(f"{qid} references unknown CHUNK_ID {cid!r}")
            continue

        text = primary.get("TEXT", "")
        if text != by_id[cid].text:
            rpt.warnings.append(
                f"{qid} TEXT in RELEVANT_CHUNKS does not match chunks.parquet"
            )

        # MULTI_HOP / INTERACTION should pull a second doc; other types should
        # not (per the strict pairing rules).
        if q.get("QUERY_COMPLEXITY") == "MULTI_HOP":
            docs = q.get("RELEVANT_DOCS", [])
            if qtype == "INTERACTION":
                if len(docs) < 2:
                    rpt.failures.append(
                        f"{qid} INTERACTION multi-hop needs >= 2 RELEVANT_DOCS, got {docs}"
                    )
            else:
                if len(docs) != 1:
                    rpt.failures.append(
                        f"{qid} {qtype} multi-hop should keep RELEVANT_DOCS==1, got {docs}"
                    )

        for d in relevant[1:]:
            if d.get("CHUNK_ID") not in by_id:
                rpt.failures.append(
                    f"{qid} distractor references unknown CHUNK_ID {d.get('CHUNK_ID')!r}"
                )

        # The answer and its claim map must be grounded in the record's own
        # evidence. These are warnings rather than failures so a dataset
        # without answers still validates.
        keep, reason = verify_record(q)
        if not keep and reason != "ok":
            rpt.warnings.append(f"{qid} answer/claim map check: {reason}")

    for t in expected_types:
        if type_counts.get(t, 0) == 0:
            rpt.failures.append(f"QUERY_TYPE {t} produced 0 queries")

        if expected_per_type > 0 and type_counts.get(t, 0) < expected_per_type:
            rpt.warnings.append(
                f"QUERY_TYPE {t} produced {type_counts.get(t, 0)} of {expected_per_type} requested"
            )

    rpt.stats = {
        "queries": len(queries),
        "by_type": dict(type_counts),
        "chunks": len(chunks),
    }
    return rpt
