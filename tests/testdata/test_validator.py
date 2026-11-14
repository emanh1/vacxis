"""Unit tests for the post-write validator."""
from __future__ import annotations

from vacxis_capstone_fa26.testdata.validator import validate

from .conftest import make_chunk


CHUNKS = [
    make_chunk("a_c0", "doc_a", "chỉ định", "Chỉ định Vax A."),
    make_chunk("a_c1", "doc_a", "chống chỉ định", "Chống chỉ định Vax A."),
    make_chunk("b_c0", "doc_b", "chỉ định", "Chỉ định Vax B."),
]


def _query(**overrides):
    base = {
        "QUERY_ID": "VAX_Q_0001",
        "QUERY": "Test?",
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
    }
    base.update(overrides)
    return base


def test_answerable_passes() -> None:
    rpt = validate(
        [_query()], CHUNKS, by_category={"chỉ định": [CHUNKS[0]]},
        expected_per_type=1, expected_types=["INDICATION"],
    )
    assert rpt.ok, rpt.render()


def test_unanswerable_must_have_empty_chunks() -> None:
    q = _query(ANSWERABILITY="UNANSWERABLE", RELEVANT_CHUNKS=[])
    rpt = validate(
        [q], CHUNKS, by_category={"chỉ định": [CHUNKS[0]]},
        expected_per_type=1, expected_types=["INDICATION"],
    )
    assert rpt.ok, rpt.render()


def test_unknown_chunk_id_fails() -> None:
    q = _query()
    q["RELEVANT_CHUNKS"][0]["CHUNK_ID"] = "does_not_exist"
    rpt = validate(
        [q], CHUNKS, by_category={"chỉ định": [CHUNKS[0]]},
        expected_per_type=1, expected_types=["INDICATION"],
    )
    assert not rpt.ok


def test_interaction_multihop_requires_two_docs() -> None:
    q = _query(
        QUERY_TYPE="INTERACTION", QUERY_COMPLEXITY="MULTI_HOP",
        RELEVANT_DOCS=["doc_a"], RELEVANT_CHUNKS=[
            {
                "CHUNK_ID": "a_c0",
                "TEXT": CHUNKS[0].text,
                "RELEVANCE_SCORE": 3,
            }
        ],
    )
    rpt = validate(
        [q], CHUNKS, by_category={"chỉ định": [CHUNKS[0]]},
        expected_per_type=1, expected_types=["INTERACTION"],
    )
    assert not rpt.ok
    assert any("INTERACTION" in f for f in rpt.failures)
