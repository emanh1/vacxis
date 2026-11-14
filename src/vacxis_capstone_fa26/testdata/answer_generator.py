"""Reference answer generator and claim-evidence mapper for Vaccine RAG.

The offline extractive generator (``ExtractiveAnswerGenerator``) is the
deterministic fallback for ``--dry-run-answers``: it composes the answer
from the record's score-3 chunks without any model call, then commits the
record's derived fields via ``validator.apply_labels``. The LLM path
(``QwenAnswerStage`` in :mod:`stages`) reuses the shared
``REFERENCE_ANSWER_FORMAT`` and ``UNANSWERABLE_FALLBACK_ANSWER`` below so
both paths emit byte-compatible records.

Both generate:
- ``REFERENCE_ANSWER``: concise, factual Vietnamese medical answers with inline
  bracketed citations e.g. ``[CHUNK_ID]``.
- ``CLAIM_EVIDENCE_MAP``: atomic claims mapped to the specific chunk ID and quote.
"""
from __future__ import annotations

from typing import Any, Protocol


UNANSWERABLE_FALLBACK_ANSWER = (
    "Tài liệu hướng dẫn sử dụng không cung cấp thông tin cho câu hỏi này. "
    "Bạn nên tham khảo ý kiến bác sĩ hoặc chuyên viên y tế tại các cơ sở tiêm chủng để được tư vấn chính xác."
)


REFERENCE_ANSWER_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "reference_answer", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "reference_answer": {"type": "string", "description": "Vietnamese answer grounded in provided chunks, with [CHUNK_ID] citations."},
                "claim_evidence_map": {
                    "type": "array", "description": "Claims and their source evidence; empty for refusal.",
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "claim": {"type": "string", "description": "One factual claim in the answer."},
                            "chunk_id": {"type": "string", "description": "Source chunk identifier."},
                            "evidence_quote": {"type": "string", "description": "Verbatim evidence from the source chunk."},
                        },
                        "required": ["claim", "chunk_id", "evidence_quote"],
                    },
                },
            },
            "required": ["reference_answer", "claim_evidence_map"],
        },
    },
}

class AnswerStage(Protocol):
    """Fills label/answer/claim map on records, in place."""

    def answer(self, records: list[dict[str, Any]]) -> None: ...


class ExtractiveAnswerGenerator:
    """Offline deterministic answer generator over the record's score-3 chunks.

    Implements :class:`AnswerStage`. After composing the answer it commits
    the record's derived fields via :func:`validator.apply_labels` so the
    extractive path and the LLM path agree on ``EVIDENCE_COMPLETENESS`` and
    evidence clearing for UNANSWERABLE records.
    """

    def answer(self, records: list[dict[str, Any]]) -> None:
        from .validator import apply_labels

        for record in records:
            self._generate_single(record)
            apply_labels(
                record,
                answerability=str(record.get("ANSWERABILITY") or "ANSWERABLE"),
                reference_answer=str(record.get("REFERENCE_ANSWER") or ""),
                claim_evidence_map=record.get("CLAIM_EVIDENCE_MAP") or [],
            )

    def _generate_single(self, q: dict[str, Any]) -> None:
        answerability = q.get("ANSWERABILITY")
        if answerability == "UNANSWERABLE":
            q["REFERENCE_ANSWER"] = UNANSWERABLE_FALLBACK_ANSWER
            q["CLAIM_EVIDENCE_MAP"] = []
            return

        relevant = [
            c for c in q.get("RELEVANT_CHUNKS", [])
            if c.get("RELEVANCE_SCORE", 0) >= 3
        ]
        if not relevant:
            q["REFERENCE_ANSWER"] = UNANSWERABLE_FALLBACK_ANSWER
            q["CLAIM_EVIDENCE_MAP"] = []
            return

        claims: list[dict[str, str]] = []
        answer_sentences: list[str] = []

        for c in relevant:
            cid = c.get("CHUNK_ID", "")
            text = c.get("TEXT", "")

            # First substantive line of the chunk: skip the synthetic
            # [CHỦ ĐỀ: ...] header.
            lines = [
                ln.strip() for ln in text.splitlines()
                if ln.strip() and not ln.strip().startswith("[CHỦ ĐỀ:")
            ]
            evidence_text = (lines[0] if lines else text[:120].strip())

            # Clean leading/trailing artifacts
            evidence_text = evidence_text.lstrip("\n- *").strip()
            if evidence_text:
                answer_sentences.append(f"{evidence_text} [{cid}]")
                claims.append({
                    "claim": evidence_text,
                    "chunk_id": cid,
                    "evidence_quote": evidence_text,
                })

        if answer_sentences:
            prefix = "Theo hướng dẫn sử dụng: "
            q["REFERENCE_ANSWER"] = prefix + " ".join(answer_sentences)
            q["CLAIM_EVIDENCE_MAP"] = claims
        else:
            q["REFERENCE_ANSWER"] = UNANSWERABLE_FALLBACK_ANSWER
            q["CLAIM_EVIDENCE_MAP"] = []
