"""Tests for the post-mask re-labelling and record verification."""
from __future__ import annotations

from vacxis_capstone_fa26.testdata.relabel import (
    apply_verification,
    verify_record,
)


def _record(**overrides) -> dict:
    record = {
        "QUERY_ID": "VAX_Q_0001",
        "QUERY": "Sau khi tiêm Vax A tôi bị sốt, có phải do vaccine không?",
        "ANSWERABILITY": "ANSWERABLE",
        "REFERENCE_ANSWER": "Theo tài liệu [c1].",
        "CLAIM_EVIDENCE_MAP": [
            {"claim": "có thể gây sốt", "chunk_id": "c1", "evidence_quote": "sốt"}
        ],
        "RELEVANT_CHUNKS": [
            {"CHUNK_ID": "c1", "TEXT": "Triệu chứng: sốt, tiêu chảy.", "RELEVANCE_SCORE": 3}
        ],
        "RELEVANT_DOCS": ["doc_a"],
        "EVIDENCE_COMPLETENESS": "COMPLETE",
    }
    record.update(overrides)
    return record


def test_verify_accepts_a_clean_record():
    keep, reason = verify_record(_record())
    assert keep, reason
    assert reason == "ok"


def test_verify_rejects_unknown_citation():
    record = _record(REFERENCE_ANSWER="Theo tài liệu [nope].")
    keep, reason = verify_record(record)
    assert not keep
    assert "nope" in reason


def test_verify_rejects_non_verbatim_quote():
    record = _record(
        CLAIM_EVIDENCE_MAP=[
            {"claim": "c", "chunk_id": "c1", "evidence_quote": "gây sốt nhẹ"}
        ]
    )
    keep, reason = verify_record(record)
    assert not keep
    assert "verbatim" in reason


def test_verify_rejects_claim_citing_unknown_chunk():
    record = _record(
        CLAIM_EVIDENCE_MAP=[
            {"claim": "c", "chunk_id": "c9", "evidence_quote": "sốt"}
        ]
    )
    keep, reason = verify_record(record)
    assert not keep
    assert "c9" in reason


def test_verify_rejects_unknown_label():
    keep, reason = verify_record(_record(ANSWERABILITY="MAYBE"))
    assert not keep
    assert "MAYBE" in reason


def test_verify_requires_empty_chunks_for_unanswerable():
    keep, reason = verify_record(_record(ANSWERABILITY="UNANSWERABLE"))
    assert not keep
    assert "chunks" in reason


def test_verify_requires_chunks_for_answerable():
    keep, reason = verify_record(
        _record(RELEVANT_CHUNKS=[], REFERENCE_ANSWER="", CLAIM_EVIDENCE_MAP=[])
    )
    assert not keep
    assert "no relevant chunk" in reason


def test_apply_verification_writes_label_and_derived_fields():
    updated = apply_verification(
        _record(),
        answerability="PARTIALLY_ANSWERABLE",
        reference_answer="chỉ một phần [c1]",
        claim_evidence_map=[
            {"claim": "c", "chunk_id": "c1", "evidence_quote": "sốt"}
        ],
    )
    assert updated["ANSWERABILITY"] == "PARTIALLY_ANSWERABLE"
    assert updated["REFERENCE_ANSWER"] == "chỉ một phần [c1]"
    assert updated["EVIDENCE_COMPLETENESS"] == "PARTIAL"
    # The chunks survive a partially answerable record.
    assert updated["RELEVANT_CHUNKS"]


def test_apply_verification_clears_evidence_for_unanswerable():
    updated = apply_verification(
        _record(),
        answerability="UNANSWERABLE",
        reference_answer="tài liệu không cung cấp thông tin",
        claim_evidence_map=[],
    )
    assert updated["ANSWERABILITY"] == "UNANSWERABLE"
    assert updated["EVIDENCE_COMPLETENESS"] == "NOT_ASSESSED"
    assert updated["RELEVANT_CHUNKS"] == []
    assert updated["RELEVANT_DOCS"] == []
    keep, reason = verify_record(updated)
    assert keep, reason


def test_apply_verification_does_not_mutate_the_input():
    original = _record()
    apply_verification(
        original,
        answerability="UNANSWERABLE",
        reference_answer="x",
        claim_evidence_map=[],
    )
    assert original["ANSWERABILITY"] == "ANSWERABLE"
    assert original["RELEVANT_CHUNKS"]
