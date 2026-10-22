"""Tests for generation package (MedicalGenerator, Prompts, RAG Pipeline)."""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from vacxis_capstone_fa26.generation.generator import MedicalGenerator
from vacxis_capstone_fa26.generation.pipeline import VacxisRAGPipeline
from vacxis_capstone_fa26.generation.prompts import (
    DEFAULT_MEDICAL_SYSTEM_PROMPT,
    build_messages,
    format_context,
)
from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer, GenerationConfig


@dataclass
class DummyChunk:
    chunk_id: str
    text: str
    doc_title: str = "Vaccine Gardasil 9"
    category: str = "liều lượng"
    header: str = "Liều lượng khuyến cáo"


def _make_chunk(cid: str = "doc_c_0001", text: str = "Gardasil 9 tiêm 0.5 ml.") -> DummyChunk:
    return DummyChunk(chunk_id=cid, text=text)


def test_format_context_empty():
    """Verify formatting when no chunks are provided."""
    ctx = format_context([])
    assert "Không có đoạn tài liệu" in ctx


def test_format_context_with_chunks():
    """Verify chunk text and headers are formatted properly."""
    chunks = [
        _make_chunk("c1", "Liều dùng 0.5ml tiêm bắp."),
        _make_chunk("c2", "Bảo quản ở 2-8°C."),
    ]
    ctx = format_context(chunks)
    assert "CHUNK_ID: c1" in ctx
    assert "CHUNK_ID: c2" in ctx
    assert "Vắc-xin: Vaccine Gardasil 9" in ctx
    assert "Liều dùng 0.5ml tiêm bắp." in ctx
    assert "Bảo quản ở 2-8°C." in ctx


def test_build_messages_structure():
    """Verify build_messages output structure for LiteLLM."""
    chunks = [_make_chunk("c1", "Nội dung vắc xin.")]
    messages = build_messages("Gardasil dùng thế nào?", chunks)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert DEFAULT_MEDICAL_SYSTEM_PROMPT in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Gardasil dùng thế nào?" in messages[1]["content"]
    assert "CHUNK_ID: c1" in messages[1]["content"]


def test_citation_validation():
    """Verify valid vs hallucinated citations are distinguished."""
    generator = MedicalGenerator()
    chunks = [_make_chunk("c_real_1"), _make_chunk("c_real_2")]

    answer_text = (
        "Gardasil 9 được tiêm bắp 0.5ml [c_real_1]. "
        "Ngoài ra có thể gặp tác dụng phụ [c_fake_999]."
    )

    valid, invalid, cited = generator._validate_citations(answer_text, chunks)
    assert valid == ["c_real_1"]
    assert invalid == ["c_fake_999"]
    assert len(cited) == 1
    assert cited[0].chunk_id == "c_real_1"


def test_refusal_detection():
    """Verify refusal phrases are detected accurately."""
    generator = MedicalGenerator()
    refusal_ans = "Tài liệu hướng dẫn sử dụng hiện tại không cung cấp thông tin này. Bạn nên hỏi bác sĩ."
    normal_ans = "Vaccine được chỉ định tiêm phòng uốn ván [doc_c_0001]."

    assert generator._detect_refusal(refusal_ans) is True
    assert generator._detect_refusal(normal_ans) is False


def test_extractive_fallback():
    """Verify extractive fallback works without LLM API calls."""
    config = GenerationConfig(dry_run=True)
    generator = MedicalGenerator(config=config)
    chunks = [
        _make_chunk("c1", "[CHỦ ĐỀ: Gardasil]\nLiều dùng khuyến cáo là 0.5ml tiêm bắp."),
    ]

    res = generator.generate("Gardasil tiêm liều bao nhiêu?", chunks)
    assert isinstance(res, GeneratedAnswer)
    assert res.query == "Gardasil tiêm liều bao nhiêu?"
    assert "[c1]" in res.answer
    assert "c1" in res.citations
    assert res.model == "extractive-fallback"
    assert res.is_refusal is False


def test_medical_generator_mocked_litellm():
    """Verify generator with mocked LiteLLM completion."""
    config = GenerationConfig(model="mock-gpt", dry_run=False)
    generator = MedicalGenerator(config=config)
    chunks = [_make_chunk("c1", "Tiêm 3 liều vào tháng 0, 1, 6.")]

    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content="Phác đồ gồm 3 liều tiêm vào tháng 0, 1 và 6 [c1]."))
    ]
    mock_resp.usage.prompt_tokens = 45
    mock_resp.usage.completion_tokens = 22
    mock_resp.usage.total_tokens = 67

    with patch.dict("os.environ", {"OPENAI_API_KEY": "fake_key"}):
        with patch("litellm.completion", return_value=mock_resp) as mock_comp:
            res = generator.generate("Lịch tiêm ra sao?", chunks)

            mock_comp.assert_called_once()
            assert "[c1]" in res.answer
            assert res.citations == ["c1"]
            assert res.invalid_citations == []
            assert res.usage["total_tokens"] == 67
            assert res.model == "mock-gpt"


def test_generated_answer_to_dict():
    """Verify serialization to dictionary."""
    chunk = _make_chunk("c1")
    ans = GeneratedAnswer(
        query="Câu hỏi test",
        answer="Câu trả lời [c1]",
        citations=["c1"],
        invalid_citations=[],
        cited_chunks=[chunk],
        retrieved_chunks=[chunk],
        is_refusal=False,
        model="gpt-4o-mini",
        latency_seconds=0.1234,
        usage={"total_tokens": 50},
    )
    d = ans.to_dict()
    assert d["query"] == "Câu hỏi test"
    assert d["citations"] == ["c1"]
    assert d["cited_chunk_ids"] == ["c1"]
    assert d["retrieved_chunk_ids"] == ["c1"]
    assert d["is_refusal"] is False
    assert d["latency_seconds"] == 0.1234


def test_pipeline_with_mocked_retriever():
    """Verify end-to-end pipeline wiring with mocked retriever and generator."""
    mock_retriever = MagicMock()
    sample_chunk = _make_chunk("c_top1", "Chỉ định phòng ngừa ung thư cổ tử cung.")
    mock_retriever.retrieve.return_value = [sample_chunk]

    mock_generator = MagicMock()
    mock_generator.generate.return_value = GeneratedAnswer(
        query="Chỉ định là gì?",
        answer="Phòng ngừa ung thư cổ tử cung [c_top1].",
        citations=["c_top1"],
        cited_chunks=[sample_chunk],
        retrieved_chunks=[sample_chunk],
        model="mock",
    )

    pipeline = VacxisRAGPipeline(retriever=mock_retriever, generator=mock_generator)
    res = pipeline.answer("Chỉ định là gì?", top_k=3)

    mock_retriever.retrieve.assert_called_once_with(query="Chỉ định là gì?", top_k=3, skip_rerank=None)
    mock_generator.generate.assert_called_once_with(query="Chỉ định là gì?", chunks=[sample_chunk])
    assert res.citations == ["c_top1"]


def test_pipeline_override_chunks():
    """Verify overriding chunks bypasses retriever call."""
    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    custom_chunk = _make_chunk("c_override", "Custom text")

    pipeline = VacxisRAGPipeline(retriever=mock_retriever, generator=mock_generator)
    pipeline.answer("Query", override_chunks=[custom_chunk])

    mock_retriever.retrieve.assert_not_called()
    mock_generator.generate.assert_called_once_with(query="Query", chunks=[custom_chunk])


def test_pipeline_answer_batch_with_mocked_retriever():
    """Verify answer_batch calls batch_retrieve on retriever and generate_batch on generator."""
    mock_retriever = MagicMock()
    chunk1 = _make_chunk("c1", "Text 1")
    chunk2 = _make_chunk("c2", "Text 2")
    mock_retriever.batch_retrieve.return_value = [[chunk1], [chunk2]]

    mock_generator = MagicMock()
    mock_generator.generate_batch.return_value = [
        GeneratedAnswer(query="q1", answer="a1", model="mock"),
        GeneratedAnswer(query="q2", answer="a2", model="mock"),
    ]

    pipeline = VacxisRAGPipeline(retriever=mock_retriever, generator=mock_generator)
    results = pipeline.answer_batch(
        queries=["q1", "q2"],
        top_k=5,
        skip_rerank=True,
    )

    mock_retriever.batch_retrieve.assert_called_once_with(
        queries=["q1", "q2"],
        top_k=5,
        skip_rerank=True,
    )
    mock_generator.generate_batch.assert_called_once_with(
        queries=["q1", "q2"],
        chunks_per_query=[[chunk1], [chunk2]],
    )
    assert len(results) == 2


def test_pipeline_answer_batch_override_chunks():
    """Verify override_chunks_per_query bypasses retrieval in answer_batch."""
    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    custom_batch = [[_make_chunk("c1")], [_make_chunk("c2")]]

    pipeline = VacxisRAGPipeline(retriever=mock_retriever, generator=mock_generator)
    pipeline.answer_batch(queries=["q1", "q2"], override_chunks_per_query=custom_batch)

    mock_retriever.batch_retrieve.assert_not_called()
    mock_generator.generate_batch.assert_called_once_with(
        queries=["q1", "q2"],
        chunks_per_query=custom_batch,
    )


def test_format_context_with_dict_chunks():
    """Verify format_context properly formats dictionary-shaped chunks."""
    dict_chunks = [
        {"chunk_id": "c_dict_1", "text": "Đoạn văn bản 1.", "doc_title": "Vaccine A", "category": "chỉ định", "header": "Mục 1"},
        {"CHUNK_ID": "c_dict_2", "TEXT": "Đoạn văn bản 2."},
    ]
    ctx = format_context(dict_chunks)
    assert "CHUNK_ID: c_dict_1" in ctx
    assert "Vắc-xin: Vaccine A" in ctx
    assert "Đoạn văn bản 1." in ctx
    assert "CHUNK_ID: c_dict_2" in ctx
    assert "Đoạn văn bản 2." in ctx


def test_generated_answer_to_dict_with_dict_chunks():
    """Verify GeneratedAnswer.to_dict handles dictionary chunks without raw dict stringification."""
    ans = GeneratedAnswer(
        query="q",
        answer="a",
        cited_chunks=[{"chunk_id": "c_cited", "text": "t"}],
        retrieved_chunks=[{"CHUNK_ID": "c_retrieved", "text": "t"}],
    )
    d = ans.to_dict()
    assert d["cited_chunk_ids"] == ["c_cited"]
    assert d["retrieved_chunk_ids"] == ["c_retrieved"]


def test_format_context_and_build_messages_flags():
    """Verify include_metadata and enforce_citations configuration flags."""
    chunks = [_make_chunk("c1", "Nội dung vắc xin.")]

    # include_metadata=False
    ctx_no_meta = format_context(chunks, include_metadata=False)
    assert "CHUNK_ID: c1" in ctx_no_meta
    assert "Vắc-xin:" not in ctx_no_meta
    assert "Chuyên mục:" not in ctx_no_meta

    # enforce_citations=False in build_messages
    msgs_no_cit = build_messages("q", chunks, enforce_citations=False)
    user_prompt = msgs_no_cit[1]["content"]
    assert "trích dẫn mã [CHUNK_ID] đầy đủ" not in user_prompt
    assert "Hãy trả lời câu hỏi dựa trên các tài liệu trên." in user_prompt

    # enforce_citations=True in build_messages
    msgs_cit = build_messages("q", chunks, enforce_citations=True)
    assert "trích dẫn mã [CHUNK_ID] đầy đủ sau mỗi câu khẳng định." in msgs_cit[1]["content"]

def test_generator_litellm_exception_sets_extractive_fallback_model():
    """Verify that when litellm raises an exception, model is updated to extractive-fallback."""
    config = GenerationConfig(model="gpt-4o", dry_run=False)
    generator = MedicalGenerator(config=config)
    chunks = [_make_chunk("c1", "Nội dung vắc xin [c1].")]

    with patch.dict("os.environ", {"OPENAI_API_KEY": "fake_key"}):
        with patch("litellm.completion", side_effect=RuntimeError("LiteLLM connection error")):
            res = generator.generate("Câu hỏi?", chunks)
            assert res.model == "extractive-fallback"
            assert "[c1]" in res.answer


def test_generator_unified_llm_api_key():
    """Verify LLM_API_KEY is accepted and passed to litellm.completion kwargs."""
    config = GenerationConfig(model="custom-model", dry_run=False)
    generator = MedicalGenerator(config=config)
    chunks = [_make_chunk("c1", "Nội dung.")]

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Đáp án [c1]."))]
    mock_resp.usage.prompt_tokens = 10
    mock_resp.usage.completion_tokens = 10
    mock_resp.usage.total_tokens = 20

    with patch.dict("os.environ", {"LLM_API_KEY": "unified_secret_key"}, clear=True):
        with patch("litellm.completion", return_value=mock_resp) as mock_comp:
            res = generator.generate("Query?", chunks)
            mock_comp.assert_called_once()
            _, kwargs = mock_comp.call_args
            assert kwargs.get("api_key") == "unified_secret_key"
            assert res.model == "custom-model"


def test_citation_validation_advanced_formats():
    """Verify extraction of merged citations, bracket prefixes, and multiple ids."""
    generator = MedicalGenerator()
    chunks = [_make_chunk("c1"), _make_chunk("c2"), _make_chunk("c3")]

    # Multi-citation inside single bracket
    valid, invalid, cited = generator._validate_citations(
        "Chỉ định ngừa bệnh [c1, c2]. Tác dụng phụ [CHUNK_ID: c3]. Lỗi [c_fake].",
        chunks,
    )
    assert "c1" in valid
    assert "c2" in valid
    assert "c3" in valid
    assert "c_fake" in invalid
    assert len(cited) == 3

    # Semicolon separated citations
    valid_semi, _, _ = generator._validate_citations("Chỉ định [c1; c2].", chunks)
    assert valid_semi == ["c1", "c2"]

    # Header style prefix [Tài liệu 1 | CHUNK_ID: c1]
    valid_prefix, _, _ = generator._validate_citations("Chỉ định [TÀI LIỆU 1 | CHUNK_ID: c1].", chunks)
    assert valid_prefix == ["c1"]


def test_cli_null_safety_and_env_model():
    """Verify cli safe parsing of null values and respect for VACXIS_LLM_MODEL."""
    from vacxis_capstone_fa26.generation.cli import _GroundTruthChunk
    import os

    # Test null ENTITIES and null RELEVANCE_SCORE
    item = {
        "QUERY": "q",
        "ENTITIES": None,
        "RELEVANT_CHUNKS": [
            {"CHUNK_ID": "c1", "TEXT": "t", "RELEVANCE_SCORE": None}
        ]
    }
    raw_chunks = item.get("RELEVANT_CHUNKS") or []
    chunks = [
        _GroundTruthChunk(
            chunk_id=c.get("CHUNK_ID", "") if isinstance(c, dict) else getattr(c, "chunk_id", ""),
            doc_title=(item.get("ENTITIES") or {}).get("vaccine", "") if isinstance(item, dict) else "",
            text=c.get("TEXT", "") if isinstance(c, dict) else getattr(c, "text", ""),
            score=float((c.get("RELEVANCE_SCORE") if isinstance(c, dict) else getattr(c, "score", 0.0)) or 0.0),
        )
        for c in raw_chunks
    ]
    assert len(chunks) == 1
    assert chunks[0].doc_title == ""
    assert chunks[0].score == 0.0

    # Test VACXIS_LLM_MODEL env var
    with patch.dict("os.environ", {"VACXIS_LLM_MODEL": "gemini-2.5-pro"}):
        args_model = None
        model_name = args_model if args_model is not None else os.environ.get("VACXIS_LLM_MODEL", "gpt-4o-mini")
        cfg = GenerationConfig(model=model_name)
        assert cfg.model == "gemini-2.5-pro"

