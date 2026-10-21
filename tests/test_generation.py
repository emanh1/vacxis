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
