"""Offline tests for the query-driven (source) dataset path.

These exercise the rule-based rewriter, grounding, record schema and the
vLLM JSON-parsing helpers *without* a GPU: the real vLLM path is covered by
the marked integration test, which is skipped by default.
"""
from __future__ import annotations

import json

import pytest

from vacxis_capstone_fa26.testdata.generator import (
    RuleBasedRewriter,
    generate_and_validate_from_source,
)
from vacxis_capstone_fa26.testdata.vllm_rewrite import (
    QwenAnswerGenerator,
    QwenRewriter,
    VLLMCheckpoint,
    VLLMClient,
    VLLMError,
    _extract_json,
    _normalize_result,
)

from .conftest import make_chunk


REQUIRED_KEYS = {
    "QUERY_ID", "QUERY", "QUERY_DIFFICULTY", "QUERY_TYPE", "QUERY_SOURCE",
    "QUERY_COMPLEXITY", "ENTITIES", "CONSTRAINTS", "ANSWERABILITY",
    "RELEVANT_DOCS", "RELEVANT_CHUNKS", "REQUIRED_EVIDENCE_GROUPS",
    "EVIDENCE_COMPLETENESS", "REFERENCE_ANSWER", "CLAIM_EVIDENCE_MAP",
    "SOURCE_VERSION", "VERIFIED", "ANNOTATOR_IDS", "ADJUDICATED",
    "VERIFICATION_NOTES",
}


def _corpus() -> list:
    return [
        make_chunk("a_c0", "doc_a", "chỉ định", "Chỉ định Vax A.", vaccine="Vax A"),
        make_chunk("a_c1", "doc_a", "cảnh báo và thận trọng", "Thận trọng Vax A.", vaccine="Vax A"),
        make_chunk("b_c0", "doc_b", "chống chỉ định", "Chống chỉ định Vax B.", vaccine="Vax B"),
    ]


def test_rule_based_rewriter_shape():
    import random

    src = [{"id": 1, "query": "răng đau"}, {"id": 2, "query": "sốt"}]
    out = RuleBasedRewriter(_corpus(), random.Random(0)).rewrite(src)
    assert [o["id"] for o in out] == [1, 2]
    for o in out:
        assert o["vaccine"] in {"Vax A", "Vax B"}
        assert o["rewritten_query"] in {"răng đau", "sốt"}
        assert o["query_type"] in {
            "INDICATION", "DOSAGE", "ADMINISTRATION_ROUTE", "CONTRAINDICATION",
            "ADVERSE_EFFECT", "COMPOSITION", "STORAGE", "PRECAUTION",
            "INTERACTION", "SPECIAL_POPULATION",
        }


def test_query_path_records_are_schema_valid_and_grounded():
    import random

    src = [{"id": i, "query": f"question {i}"} for i in range(5)]
    queries, rpt = generate_and_validate_from_source(
        _corpus(),
        src,
        RuleBasedRewriter(_corpus(), random.Random(0)),
        random.Random(1),
        distractors_per_query=1,
    )
    assert len(queries) == 5
    assert rpt.ok, rpt.failures
    chunk_ids = {c.chunk_id for c in _corpus()}
    for q in queries:
        assert REQUIRED_KEYS <= set(q)
        assert q["QUERY_SOURCE"] == "VLLM_QUERY"
        if q["ANSWERABILITY"] != "UNANSWERABLE":
            # Every relevant chunk must point at a real corpus chunk.
            assert q["RELEVANT_CHUNKS"]
            for rc in q["RELEVANT_CHUNKS"]:
                assert rc["CHUNK_ID"] in chunk_ids


def test_extract_json_tolerates_fences_and_trailing_text():
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('prefix {"a": 1} suffix') == {"a": 1}
    assert _extract_json("no json here") is None
    assert _extract_json('[1, 2]') is None  # not a dict


def test_normalize_result_defaults_on_garbage():
    r = _normalize_result({}, fallback_id=7)
    assert r["id"] == 7
    assert r["query_type"] == "PRECAUTION"
    assert r["answerability"] == "ANSWERABLE"
    good = _normalize_result(
        {"vaccine": "X", "query_type": "dosage", "answerability": "UNANSWERABLE"},
        fallback_id=7,
    )
    assert good["query_type"] == "DOSAGE"  # upper-cased
    assert good["answerability"] == "UNANSWERABLE"
    assert good["vaccine"] == "X"


class _FakeCompletion:
    """Mimics vLLM's RequestOutput: .outputs is a list, [0] has .text."""

    def __init__(self, text: str) -> None:
        self._completion = _Completion(text)
        self.outputs = [self._completion]


class _Completion:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeLLM:
    """Minimal stand-in for vllm.LLM so the vLLM path runs without a GPU."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self._tokenizer = _FakeTokenizer()

    def generate(self, prompts, sampling):
        self.calls.append(prompts)
        return [_FakeCompletion('{"a": 1}') for _ in prompts]

    def get_tokenizer(self):
        return self._tokenizer


class _FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "<rendered>"


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "ckpt.json"
    cp = VLLMCheckpoint(path, model="m", seed=1)
    cp.put_rewrite(1, {"vaccine": "V", "rewritten_query": "q"})
    cp.put_answer(1, "ans", [])
    cp.save()

    cp2 = VLLMCheckpoint(path, model="m", seed=1)
    cp2.load()
    assert cp2.get_rewrite("1")["vaccine"] == "V"
    assert cp2.get_answer("1")["reference_answer"] == "ans"


def test_checkpoint_meta_mismatch_raises(tmp_path):
    path = tmp_path / "ckpt.json"
    VLLMCheckpoint(path, model="m", seed=1).save()
    with pytest.raises(VLLMError):
        VLLMCheckpoint(path, model="other", seed=1).load()


def test_rewriter_uses_checkpoint_cache(tmp_path, monkeypatch):
    import vacxis_capstone_fa26.testdata.vllm_rewrite as vr

    monkeypatch.setattr(vr, "SamplingParams", lambda **kw: kw)
    cp = VLLMCheckpoint(tmp_path / "ckpt.json", model="m", seed=1)
    cp.put_rewrite(2, {"id": 2, "vaccine": "Vax A", "rewritten_query": "c",
                       "query_type": "PRECAUTION", "answerability": "ANSWERABLE"})
    llm = _FakeLLM()
    client = VLLMClient(llm=llm, model="m")
    out = QwenRewriter(client, vaccine_names=["Vax A"], checkpoint=cp).rewrite(
        [{"id": 2, "query": "q"}, {"id": 3, "query": "r"}]
    )
    assert out[0]["vaccine"] == "Vax A"  # from cache, no LLM call
    assert len(llm.calls[0]) == 1  # only the uncached id=3 hit the model


def test_rewriter_chunks_llm_calls(tmp_path, monkeypatch):
    """--rewrite-batch-size drives both the LLM call size and save points."""
    import vacxis_capstone_fa26.testdata.vllm_rewrite as vr

    monkeypatch.setattr(vr, "SamplingParams", lambda **kw: kw)
    cp = VLLMCheckpoint(tmp_path / "ckpt.json", model="m", seed=1)
    llm = _FakeLLM()
    client = VLLMClient(llm=llm, model="m")
    out = QwenRewriter(
        client,
        vaccine_names=["Vax A"],
        checkpoint=cp,
        rewrite_batch_size=2,
    ).rewrite([{"id": i, "query": f"q{i}"} for i in range(5)])
    assert [o["id"] for o in out] == [0, 1, 2, 3, 4]  # order preserved
    # 5 queries in chunks of 2 -> 3 LLM calls of size 2, 2, 1.
    assert [len(c) for c in llm.calls] == [2, 2, 1]
    # One save per chunk, all five cached.
    assert len(cp.rewrites) == 5


def test_resume_restores_cached_answers(tmp_path, monkeypatch):
    """A resumed run must emit the cached answer text, not None (gate B1)."""
    import vacxis_capstone_fa26.testdata.vllm_rewrite as vr

    monkeypatch.setattr(vr, "SamplingParams", lambda **kw: kw)
    cp = VLLMCheckpoint(tmp_path / "ckpt.json", model="m", seed=1)
    cp.put_answer(7, "cached answer text", [{"claim": "c"}])
    cp.save()

    cp2 = VLLMCheckpoint(tmp_path / "ckpt.json", model="m", seed=1)
    cp2.load()
    llm = _FakeLLM()
    client = VLLMClient(llm=llm, model="m")
    gen = QwenAnswerGenerator(client, checkpoint=cp2, batch_size=2)
    queries = [
        {"SOURCE_QUERY_ID": 7, "ANSWERABILITY": "ANSWERABLE",
         "RELEVANT_CHUNKS": [{"RELEVANCE_SCORE": 3, "CHUNK_ID": "c", "TEXT": "t"}]},
        {"SOURCE_QUERY_ID": 8, "ANSWERABILITY": "ANSWERABLE",
         "RELEVANT_CHUNKS": [{"RELEVANCE_SCORE": 3, "CHUNK_ID": "c", "TEXT": "t"}]},
    ]
    gen.generate_answers(queries)

    # The cached id keeps its text and is not re-requested from the model.
    assert queries[0]["REFERENCE_ANSWER"] == "cached answer text"
    assert queries[0]["CLAIM_EVIDENCE_MAP"] == [{"claim": "c"}]
    # Only the uncached id went to the model.
    assert len(llm.calls) == 1
    assert len(llm.calls[0]) == 1
    # No record may end up with a missing answer.
    assert all(q["REFERENCE_ANSWER"] for q in queries)


def test_answer_generator_batches_and_checkpoints(tmp_path, monkeypatch):
    import vacxis_capstone_fa26.testdata.vllm_rewrite as vr

    monkeypatch.setattr(vr, "SamplingParams", lambda **kw: kw)
    cp = VLLMCheckpoint(tmp_path / "ckpt.json", model="m", seed=1)
    llm = _FakeLLM()
    client = VLLMClient(llm=llm, model="m")
    gen = QwenAnswerGenerator(client, checkpoint=cp, batch_size=2)
    queries = [
        {"SOURCE_QUERY_ID": i, "ANSWERABILITY": "ANSWERABLE",
         "RELEVANT_CHUNKS": [{"RELEVANCE_SCORE": 3, "CHUNK_ID": "c", "TEXT": "t"}]}
        for i in range(3)
    ]
    gen.generate_answers(queries)
    assert len(llm.calls) == 2  # batches of 2 + 1
    assert llm.calls[0][0] == "<rendered>"  # rendered strings, not dicts
    assert all(q["REFERENCE_ANSWER"] for q in queries)


@pytest.mark.integration
def test_vllm_path_smoke(monkeypatch):
    """Exercises the real vLLM classes on a GPU host; skipped by default."""
    from vacxis_capstone_fa26.testdata.vllm_rewrite import QwenRewriter, VLLMClient

    client = VLLMClient(model="Qwen/Qwen2.5-7B-Instruct")
    client.ensure_loaded()
    rw = QwenRewriter(client, vaccine_names=["Vax A"])
    out = rw.rewrite([{"id": 1, "query": "phụ nữ mang thai có tiêm được không"}])
    assert out[0]["vaccine"]


class _MaskFakeLLM:
    """Fake LLM returning a fixed symptom-replacement payload."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[list[str]] = []
        self._tokenizer = _FakeTokenizer()

    def generate(self, prompts, sampling):
        self.calls.append(prompts)
        return [_FakeCompletion(self.payload) for _ in prompts]

    def get_tokenizer(self):
        return self._tokenizer


def _mask_record(query: str, chunk_text: str, source_id: int = 1) -> dict:
    return {
        "QUERY_ID": "VAX_Q_0001",
        "SOURCE_QUERY_ID": source_id,
        "QUERY": query,
        "RELEVANT_CHUNKS": [
            {"CHUNK_ID": "c1", "TEXT": chunk_text, "RELEVANCE_SCORE": 3}
        ],
    }


def test_symptom_masker_uses_llm_and_validates(tmp_path, monkeypatch):
    """The model proposes pairs; only pairs grounded in the chunk are applied."""
    import vacxis_capstone_fa26.testdata.vllm_rewrite as vr
    from vacxis_capstone_fa26.testdata.postprocess import apply_replacements

    monkeypatch.setattr(vr, "SamplingParams", lambda **kw: kw)
    payload = json.dumps(
        {
            "replacements": [
                {"original": "đau đầu", "replacement": "tiêu chảy"},
                {"original": "hắt hơi", "replacement": "sốt"},
            ]
        }
    )
    llm = _MaskFakeLLM(payload)
    client = VLLMClient(llm=llm, model="m")
    cp = VLLMCheckpoint(tmp_path / "ckpt.json", model="m", seed=1)
    masker = vr.QwenSymptomMasker(client, checkpoint=cp, batch_size=8)

    rec = _mask_record(
        "Sau khi tiêm tôi bị đau đầu dữ dội?", "Triệu chứng: sốt, tiêu chảy."
    )
    pairs = masker.mask([rec])
    assert len(llm.calls) == 1
    assert pairs[0][0]["original"] == "đau đầu"

    updated, report = apply_replacements(rec, pairs[0])
    assert report["applied"] == ["đau đầu"]
    assert "tiêu chảy" in updated["QUERY"]
    assert "hắt hơi" not in updated["QUERY"]

    assert cp.get_mask(1) == pairs[0]


def test_symptom_masker_resumes_from_checkpoint(tmp_path, monkeypatch):
    import vacxis_capstone_fa26.testdata.vllm_rewrite as vr

    monkeypatch.setattr(vr, "SamplingParams", lambda **kw: kw)
    cp = VLLMCheckpoint(tmp_path / "ckpt.json", model="m", seed=1)
    cp.put_mask(1, [{"original": "a", "replacement": "b"}])
    cp.save()

    cp2 = VLLMCheckpoint(tmp_path / "ckpt.json", model="m", seed=1)
    cp2.load()
    llm = _MaskFakeLLM("{}")
    client = VLLMClient(llm=llm, model="m")
    masker = vr.QwenSymptomMasker(client, checkpoint=cp2, batch_size=8)
    pairs = masker.mask([_mask_record("q", "chunk", source_id=1)])
    assert pairs[0] == [{"original": "a", "replacement": "b"}]
    assert llm.calls == []


def test_symptom_masker_tolerates_bad_payload(tmp_path, monkeypatch):
    import vacxis_capstone_fa26.testdata.vllm_rewrite as vr

    monkeypatch.setattr(vr, "SamplingParams", lambda **kw: kw)
    llm = _MaskFakeLLM("not json at all")
    client = VLLMClient(llm=llm, model="m")
    masker = vr.QwenSymptomMasker(client, batch_size=8)
    pairs = masker.mask([_mask_record("q", "chunk")])
    assert pairs == [[]]


class _VerifyFakeLLM:
    """Fake LLM returning a fixed verification payload."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[list[str]] = []
        self._tokenizer = _FakeTokenizer()

    def generate(self, prompts, sampling):
        self.calls.append(prompts)
        return [_FakeCompletion(self.payload) for _ in prompts]

    def get_tokenizer(self):
        return self._tokenizer


def _verify_record(query: str, chunk_text: str, source_id: int = 1) -> dict:
    return {
        "QUERY_ID": "VAX_Q_0001",
        "SOURCE_QUERY_ID": source_id,
        "QUERY": query,
        "ANSWERABILITY": "ANSWERABLE",
        "RELEVANT_CHUNKS": [
            {"CHUNK_ID": "c1", "TEXT": chunk_text, "RELEVANCE_SCORE": 3}
        ],
        "RELEVANT_DOCS": ["doc"],
        "EVIDENCE_COMPLETENESS": "COMPLETE",
    }


def test_verifier_uses_llm_and_checkpoints(tmp_path, monkeypatch):
    import vacxis_capstone_fa26.testdata.vllm_rewrite as vr

    monkeypatch.setattr(vr, "SamplingParams", lambda **kw: kw)
    payload = json.dumps(
        {
            "answerability": "PARTIALLY_ANSWERABLE",
            "reference_answer": "chỉ một phần [c1]",
            "claim_evidence_map": [
                {"claim": "chỉ một phần", "chunk_id": "c1",
                 "evidence_quote": "sốt"}
            ],
        }
    )
    llm = _VerifyFakeLLM(payload)
    client = VLLMClient(llm=llm, model="m")
    cp = VLLMCheckpoint(tmp_path / "ckpt.json", model="m", seed=1)
    verifier = vr.QwenLabelVerifier(client, checkpoint=cp, batch_size=8)

    results = verifier.verify([_verify_record("q", "có sốt")])
    assert len(llm.calls) == 1
    assert results[0]["answerability"] == "PARTIALLY_ANSWERABLE"
    assert results[0]["reference_answer"] == "chỉ một phần [c1]"
    assert cp.get_verification(1)["reference_answer"] == "chỉ một phần [c1]"


def test_verifier_unanswerable_drops_claims(tmp_path, monkeypatch):
    import vacxis_capstone_fa26.testdata.vllm_rewrite as vr

    monkeypatch.setattr(vr, "SamplingParams", lambda **kw: kw)
    payload = json.dumps(
        {
            "answerability": "UNANSWERABLE",
            "reference_answer": "tài liệu không có thông tin",
            "claim_evidence_map": [{"claim": "x", "chunk_id": "c1"}],
        }
    )
    llm = _VerifyFakeLLM(payload)
    client = VLLMClient(llm=llm, model="m")
    verifier = vr.QwenLabelVerifier(client, batch_size=8)
    results = verifier.verify([_verify_record("q", "text")])
    # An unanswerable record must not keep claims, even if the model sent some.
    assert results[0]["claim_evidence_map"] == []


def test_verifier_falls_back_on_bad_payload(tmp_path, monkeypatch):
    import vacxis_capstone_fa26.testdata.vllm_rewrite as vr

    monkeypatch.setattr(vr, "SamplingParams", lambda **kw: kw)
    llm = _VerifyFakeLLM("not json")
    client = VLLMClient(llm=llm, model="m")
    verifier = vr.QwenLabelVerifier(client, batch_size=8)
    results = verifier.verify([_verify_record("q", "text")])
    assert results[0]["answerability"] == "UNANSWERABLE"
    assert results[0]["claim_evidence_map"] == []


def test_verifier_resumes_from_checkpoint(tmp_path, monkeypatch):
    import vacxis_capstone_fa26.testdata.vllm_rewrite as vr

    monkeypatch.setattr(vr, "SamplingParams", lambda **kw: kw)
    cp = VLLMCheckpoint(tmp_path / "ckpt.json", model="m", seed=1)
    cp.put_verification(
        1,
        answerability="ANSWERABLE",
        reference_answer="cached",
        claim_map=[{"claim": "c", "chunk_id": "c1", "evidence_quote": "e"}],
    )
    cp.save()

    cp2 = VLLMCheckpoint(tmp_path / "ckpt.json", model="m", seed=1)
    cp2.load()
    llm = _VerifyFakeLLM("{}")
    client = VLLMClient(llm=llm, model="m")
    verifier = vr.QwenLabelVerifier(client, checkpoint=cp2, batch_size=8)
    results = verifier.verify([_verify_record("q", "text", source_id=1)])
    assert results[0]["reference_answer"] == "cached"
    assert llm.calls == []


def test_checkpoint_adds_verification_additively(tmp_path):
    """A checkpoint without the verifications key still loads."""
    import vacxis_capstone_fa26.testdata.vllm_rewrite as vr

    path = tmp_path / "old.json"
    # Simulate a checkpoint written before the verification stage existed.
    import json as _json
    path.write_text(
        _json.dumps(
            {
                "meta": {"model": "m", "seed": 1, "prompt_version": 2},
                "rewrites": {},
                "answers": {},
                "masks": {},
            }
        ),
        encoding="utf-8",
    )
    cp = VLLMCheckpoint(path, model="m", seed=1)
    cp.load()
    assert cp.verifications == {}
    cp.put_verification(9, answerability="ANSWERABLE", reference_answer="a", claim_map=[])
    cp.save()
    cp2 = VLLMCheckpoint(path, model="m", seed=1)
    cp2.load()
    assert cp2.get_verification("9")["reference_answer"] == "a"
