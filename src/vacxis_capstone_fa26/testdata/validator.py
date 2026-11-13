"""Post-write validation for the generated dataset.

The validator is read-only: it inspects the produced query list and emits a
: class:`ValidationReport` summarising hard failures (exit non-zero) and soft
warnings. The output JSON schema is unchanged from the original pipeline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .chunks import Chunk
from .relabel import verify_record


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

        # New datasets have null spans; retain bounds checks for legacy datasets.
        for chunk in relevant:
            for span in chunk.get("EVIDENCE_SPANS") or []:
                start, end = span.get("START"), span.get("END")
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(chunk.get("TEXT", ""))):
                    rpt.failures.append(f"{qid} EVIDENCE_SPAN out of bounds")

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
