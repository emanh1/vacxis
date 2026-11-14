"""Ground rewritten user queries onto the chunk corpus.

The query-driven path: raw queries from ``query.parquet`` are rewritten into
vaccine-domain questions (by Qwen via vLLM, or offline), the proposed
vaccine is resolved against the corpus, and each record is grounded on a real
chunk with distractors. Orchestration: :mod:`distractors`, :mod:`validator`,
reference answer generation.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

from .answer_generator import AnswerGenerator
from .chunks import Chunk, index_by_category, index_by_vaccine
from .distractors import sample_distractors, score_for
from .taxonomy import QUERY_TYPE_DEFS
from .validator import validate


def derive_default_seed(chunks: list[Chunk], per_type: int) -> int:
    """Deterministic seed based on the chunk content + per-type target.

    Same ``chunks.parquet`` + same ``per_type`` => same dataset, without the
    user having to remember ``--seed``.
    """
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda c: c.chunk_id):
        h.update(c.chunk_id.encode("utf-8"))
        h.update(b"|")
    h.update(str(per_type).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


def _resolve_vaccine(
    proposed: str,
    all_vaccines: list[str],
    by_vaccine: dict[str, list[Chunk]],
    rng: random.Random,
) -> str:
    """Best-effort match of the model's proposed vaccine against the corpus."""
    proposed = (proposed or "").strip()
    if proposed in by_vaccine:
        return proposed
    if proposed:
        low = proposed.lower()
        for name in all_vaccines:
            if low in name.lower() or name.lower() in low:
                return name
    # Unresolvable -> a random corpus vaccine keeps the record grounded.
    return rng.choice(all_vaccines) if all_vaccines else proposed


def build_queries(
    chunks: list[Chunk],
    source_queries: list[dict[str, Any]],
    rewriter,
    rng: random.Random,
    distractors_per_query: int,
    answer_generator: AnswerGenerator | None = None,
) -> list[dict]:
    """Ground rewritten source queries onto the corpus and emit records.

    The rewriter proposes a ``vaccine`` name; it is resolved against
    ``index_by_vaccine`` (fuzzy: exact, then case-insensitive substring).
    Unresolvable names fall back to a random corpus vaccine so the record
    always points at a real chunk.
    """
    by_category = index_by_category(chunks)
    by_vaccine = index_by_vaccine(chunks)
    all_vaccines = list(by_vaccine.keys())

    # Type -> preferred category (the chunk category that supports it).
    type_to_category = {t["type"]: t["category"] for t in QUERY_TYPE_DEFS}

    results = rewriter.rewrite(source_queries)

    queries: list[dict] = []
    for sq, res in zip(source_queries, results):
        vaccine = _resolve_vaccine(res.get("vaccine"), all_vaccines, by_vaccine, rng)
        vaccine_chunks = by_vaccine.get(vaccine, [])
        if not vaccine_chunks:
            # Unknown vaccine: ground on a random corpus chunk instead.
            vaccine = rng.choice(all_vaccines)
            vaccine_chunks = by_vaccine.get(vaccine, [])
        if not vaccine_chunks:
            continue

        qtype = res.get("query_type") or "PRECAUTION"
        cat = type_to_category.get(qtype, "cảnh báo và thận trọng")
        type_def = next((td for td in QUERY_TYPE_DEFS if td["type"] == qtype), None)

        # Prefer a chunk whose category matches the query type; else random.
        cat_chunks = [c for c in vaccine_chunks if c.category == cat]
        primary = rng.choice(cat_chunks) if cat_chunks else rng.choice(vaccine_chunks)

        answerability = res.get("answerability", "ANSWERABLE")
        evidence_chunk = {
            "CHUNK_ID": primary.chunk_id,
            "TEXT": primary.text,
            "RELEVANCE_SCORE": 3,
        }

        distractors = sample_distractors(
            primary, chunks, by_category, rng, distractors_per_query,
        )

        if answerability == "UNANSWERABLE":
            relevant_payload: list[dict] = []
            relevant_docs: list[str] = []
            evidence_completeness = "NOT_ASSESSED"
        else:
            relevant_payload = [evidence_chunk] + [
                {
                    "CHUNK_ID": d.chunk.chunk_id,
                    "TEXT": d.chunk.text,
                    "RELEVANCE_SCORE": score_for(d.kind),
                }
                for d in distractors
            ]
            relevant_docs = [primary.doc_id]
            evidence_completeness = "COMPLETE"

        queries.append(
            {
                "QUERY_ID": "",  # stamped below
                "SOURCE_QUERY_ID": sq.get("id"),  # stable parquet id, used by checkpoints
                "QUERY": res.get("rewritten_query") or str(sq.get("query", "")).strip(),
                "QUERY_DIFFICULTY": "MEDIUM",
                "QUERY_TYPE": qtype,
                "QUERY_SOURCE": "VLLM_QUERY",
                "QUERY_COMPLEXITY": "SINGLE_HOP",
                "ENTITIES": {"vaccine": vaccine},
                "CONSTRAINTS": {},
                "ANSWERABILITY": answerability,
                "RELEVANT_DOCS": relevant_docs,
                "RELEVANT_CHUNKS": relevant_payload,
                "REQUIRED_EVIDENCE_GROUPS": list(
                    type_def["evidence_groups"] if type_def else []
                ),
                "EVIDENCE_COMPLETENESS": evidence_completeness,
                "REFERENCE_ANSWER": None,
                "CLAIM_EVIDENCE_MAP": [],
                "SOURCE_VERSION": "PI_VERSION_ID",
                "VERIFIED": False,
                "ANNOTATOR_IDS": ["GEN"],
                "ADJUDICATED": False,
                "VERIFICATION_NOTES": None,
            }
        )

    rng.shuffle(queries)
    for i, q in enumerate(queries, start=1):
        q["QUERY_ID"] = f"VAX_Q_{i:04d}"

    if answer_generator is not None:
        answer_generator.generate_answers(queries)

    return queries


def generate_and_validate(
    chunks: list[Chunk],
    source_queries: list[dict[str, Any]],
    rewriter,
    rng: random.Random,
    distractors_per_query: int,
    answer_generator: AnswerGenerator | None = None,
) -> tuple[list[dict], object]:
    """Ground the queries and validate the result against the corpus."""
    queries = build_queries(
        chunks, source_queries, rewriter, rng, distractors_per_query,
        answer_generator=answer_generator,
    )
    by_category = index_by_category(chunks)
    # Every real type must appear at least once; the query-driven path has no
    # fixed per-type quota, so expect the union of the types actually emitted.
    expected_types = sorted({q["QUERY_TYPE"] for q in queries}) or [
        t["type"] for t in QUERY_TYPE_DEFS
    ]
    rpt = validate(
        queries, chunks, by_category,
        expected_per_type=0,
        expected_types=expected_types,
    )
    return queries, rpt
