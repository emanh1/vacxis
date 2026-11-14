"""Unit tests for the answer generator and claim-evidence mapper."""
from __future__ import annotations

from vacxis_capstone_fa26.testdata.answer_generator import (
    ExtractiveAnswerGenerator,
    UNANSWERABLE_FALLBACK_ANSWER,
)
from vacxis_capstone_fa26.testdata.validator import validate

from .conftest import make_chunk


CHUNKS = [
    make_chunk("a_c0", "doc_a", "chỉ định", "Chỉ định tiêm cho người lớn.", vaccine="Vax A"),
    make_chunk("a_c1", "doc_a", "chống chỉ định", "Chống chỉ định phụ nữ có thai.", vaccine="Vax A"),
]


def test_extractive_answer_unanswerable() -> None:
    gen = ExtractiveAnswerGenerator()
    q = {
        "QUERY_ID": "Q1",
        "QUERY": "Giá vaccine là bao nhiêu?",
        "ANSWERABILITY": "UNANSWERABLE",
        "RELEVANT_CHUNKS": [],
        "REFERENCE_ANSWER": None,
        "CLAIM_EVIDENCE_MAP": [],
    }
    gen.generate_answers([q])
    assert q["REFERENCE_ANSWER"] == UNANSWERABLE_FALLBACK_ANSWER
    assert q["CLAIM_EVIDENCE_MAP"] == []


def test_extractive_answer_single_hop() -> None:
    gen = ExtractiveAnswerGenerator()
    q = {
        "QUERY_ID": "Q2",
        "QUERY": "Vaccine Vax A dùng cho ai?",
        "ANSWERABILITY": "ANSWERABLE",
        "RELEVANT_CHUNKS": [
            {
                "CHUNK_ID": "a_c0",
                "TEXT": CHUNKS[0].text,
                "RELEVANCE_SCORE": 3,
            }
        ],
        "REFERENCE_ANSWER": None,
        "CLAIM_EVIDENCE_MAP": [],
    }
    gen.generate_answers([q])
    assert q["REFERENCE_ANSWER"] is not None
    assert "[a_c0]" in q["REFERENCE_ANSWER"]
    assert len(q["CLAIM_EVIDENCE_MAP"]) == 1
    assert q["CLAIM_EVIDENCE_MAP"][0]["chunk_id"] == "a_c0"


def test_extractive_answer_multi_hop() -> None:
    gen = ExtractiveAnswerGenerator()
    q = {
        "QUERY_ID": "Q3",
        "QUERY": "Chỉ định và chống chỉ định của Vax A?",
        "ANSWERABILITY": "ANSWERABLE",
        "RELEVANT_CHUNKS": [
            {
                "CHUNK_ID": "a_c0",
                "TEXT": CHUNKS[0].text,
                "RELEVANCE_SCORE": 3,
            },
            {
                "CHUNK_ID": "a_c1",
                "TEXT": CHUNKS[1].text,
                "RELEVANCE_SCORE": 3,
            },
        ],
        "REFERENCE_ANSWER": None,
        "CLAIM_EVIDENCE_MAP": [],
    }
    gen.generate_answers([q])
    assert "[a_c0]" in q["REFERENCE_ANSWER"]
    assert "[a_c1]" in q["REFERENCE_ANSWER"]
    assert len(q["CLAIM_EVIDENCE_MAP"]) == 2


def test_validator_with_reference_answer() -> None:
    q = {
        "QUERY_ID": "Q5",
        "QUERY": "Chỉ định?",
        "QUERY_TYPE": "INDICATION",
        "QUERY_COMPLEXITY": "SINGLE_HOP",
        "ANSWERABILITY": "ANSWERABLE",
        "RELEVANT_DOCS": ["doc_a"],
        "RELEVANT_CHUNKS": [
            {
                "CHUNK_ID": "a_c0",
                "TEXT": CHUNKS[0].text,
                "RELEVANCE_SCORE": 3,
            }
        ],
        "REFERENCE_ANSWER": "Chỉ định cho người lớn [a_c0].",
        "CLAIM_EVIDENCE_MAP": [{"claim": "Chỉ định người lớn", "chunk_id": "a_c0", "evidence_quote": "..."}],
    }
    rpt = validate([q], CHUNKS, by_category={"chỉ định": [CHUNKS[0]]}, expected_per_type=1, expected_types=["INDICATION"])
    assert rpt.ok, rpt.render()


def test_extractive_answer_skips_synthetic_header():
    q = {"ANSWERABILITY": "ANSWERABLE", "RELEVANT_CHUNKS": [
        {"CHUNK_ID": "c1", "TEXT": "[CHỦ ĐỀ: Vax A]\nEvidence.",
         "RELEVANCE_SCORE": 3},
    ]}
    ExtractiveAnswerGenerator().generate_answers([q])
    assert q["REFERENCE_ANSWER"] == "Theo hướng dẫn sử dụng: Evidence. [c1]"
