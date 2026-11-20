"""Medical LLM Generator with Grounding, Verification, and Citation Parsing."""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Sequence

from vacxis_capstone_fa26.llm import completion

from vacxis_capstone_fa26.generation.prompts import (
    DEFAULT_MEDICAL_SYSTEM_PROMPT,
    REFUSAL_TRIGGER_PHRASES,
    build_messages,
    get_chunk_field,
)
from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer, GenerationConfig

logger = logging.getLogger("vacxis.generation.generator")

_BRACKET_RE = re.compile(r"\[(.*?)\]")
_ID_TOKEN_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")
_IGNORE_TOKENS = {
    "chunk", "chunks", "chunk_id", "chunk_ids", "id", "ids",
    "tai_lieu", "tai", "lieu", "document", "doc", "vắc-xin", "vacxin", "vaccine",
}


class MedicalGenerator:
    """Orchestrates LLM inference with grounding guardrails and citation verification."""

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()

    def generate(
        self,
        query: str,
        chunks: Sequence[Any],
    ) -> GeneratedAnswer:
        """Generate a grounded, cited medical answer for a query given retrieved chunks."""
        if not query or not query.strip():
            return GeneratedAnswer(
                query="",
                answer="Không có câu hỏi được cung cấp.",
                model=self.config.model,
                is_refusal=True,
            )

        start_time = time.perf_counter()

        if self.config.dry_run:
            answer_text = self._extractive_fallback(query, chunks)
            latency = time.perf_counter() - start_time
            valid_cits, invalid_cits, cited_chunks = self._validate_citations(answer_text, chunks)
            is_refusal = self._detect_refusal(answer_text)

            return GeneratedAnswer(
                query=query,
                answer=answer_text,
                citations=valid_cits,
                invalid_citations=invalid_cits,
                cited_chunks=cited_chunks,
                retrieved_chunks=list(chunks),
                is_refusal=is_refusal,
                model="extractive-fallback",
                latency_seconds=latency,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        messages = build_messages(
            query=query,
            chunks=chunks,
            system_prompt=self.config.system_prompt_override or DEFAULT_MEDICAL_SYSTEM_PROMPT,
            include_metadata=self.config.include_metadata,
            enforce_citations=self.config.enforce_citations,
        )

        model_used = self.config.model
        kwargs: dict[str, Any] = {}
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        response = completion(
            model=self.config.model, messages=messages,
            temperature=self.config.temperature, timeout=self.config.timeout,
            retry_policy=self.config.retry_policy,
            **kwargs,
        )
        answer_text = response.choices[0].message.content.strip()
        usage = {
            name: getattr(getattr(response, "usage", None), name, 0) or 0
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        }

        latency = time.perf_counter() - start_time
        valid_cits, invalid_cits, cited_chunks = self._validate_citations(answer_text, chunks)
        is_refusal = self._detect_refusal(answer_text)

        return GeneratedAnswer(
            query=query,
            answer=answer_text,
            citations=valid_cits,
            invalid_citations=invalid_cits,
            cited_chunks=cited_chunks,
            retrieved_chunks=list(chunks),
            is_refusal=is_refusal,
            model=model_used,
            latency_seconds=latency,
            usage=usage,
        )

    def generate_batch(
        self,
        queries: Sequence[str],
        chunks_per_query: Sequence[Sequence[Any]],
    ) -> list[GeneratedAnswer]:
        """Generate answers for a batch of queries."""
        if len(queries) != len(chunks_per_query):
            raise ValueError("Expected one chunk group per query.")
        return [
            self.generate(q, c_list)
            for q, c_list in zip(queries, chunks_per_query)
        ]

    def _validate_citations(
        self,
        answer_text: str,
        chunks: Sequence[Any],
    ) -> tuple[list[str], list[str], list[Any]]:
        """Extract citations and distinguish between valid and hallucinated chunk IDs."""
        extracted_cids: list[str] = []
        for match in _BRACKET_RE.finditer(answer_text):
            content = match.group(1).strip()
            # Remove any leading metadata like "TÀI LIỆU 1 | CHUNK_ID:" or "CHUNK_ID:"
            content_clean = re.sub(r"^.*?(?:CHUNK_ID|chunk_id)\s*[:\-]\s*", "", content, flags=re.IGNORECASE)
            parts = re.split(r"[,;|\s]+", content_clean)
            for part in parts:
                clean_part = part.strip().strip(":").strip("'\"").strip()
                if not clean_part:
                    continue
                if ":" in clean_part:
                    clean_part = clean_part.split(":")[-1].strip()
                if _ID_TOKEN_RE.match(clean_part):
                    if clean_part.lower() not in _IGNORE_TOKENS and clean_part not in extracted_cids:
                        extracted_cids.append(clean_part)

        chunk_by_id = {str(get_chunk_field(c, "chunk_id", str(c))): c for c in chunks}
        valid_citations: list[str] = []
        invalid_citations: list[str] = []
        cited_chunks: list[Any] = []

        for cid in extracted_cids:
            if cid in chunk_by_id:
                valid_citations.append(cid)
                cited_chunks.append(chunk_by_id[cid])
            else:
                invalid_citations.append(cid)

        return valid_citations, invalid_citations, cited_chunks

    def _detect_refusal(self, answer_text: str) -> bool:
        """Detect whether the answer is a refusal/unanswerable response."""
        lower = answer_text.lower()
        return any(phrase in lower for phrase in REFUSAL_TRIGGER_PHRASES)

    def _extractive_fallback(
        self,
        query: str,
        chunks: Sequence[Any],
    ) -> str:
        """Synthesize answer extractively from top chunks without external LLM calls."""
        if not chunks:
            return (
                "Tài liệu hướng dẫn sử dụng hiện tại không cung cấp thông tin cho câu hỏi này. "
                "Bạn nên tham khảo ý kiến bác sĩ hoặc chuyên viên y tế tại cơ sở tiêm chủng để được tư vấn chính xác."
            )

        snippets: list[str] = []
        for c in chunks[:3]:
            cid = str(get_chunk_field(c, "chunk_id", ""))
            raw_text = str(get_chunk_field(c, "text", ""))
            lines = [
                ln.strip() for ln in raw_text.splitlines()
                if ln.strip() and not ln.strip().startswith("[CHỦ ĐỀ:")
            ]
            first_sentence = lines[0] if lines else raw_text[:120].strip()
            first_sentence = first_sentence.lstrip("\n- *").strip()
            if first_sentence:
                snippets.append(f"{first_sentence} [{cid}]")

        if snippets:
            return "Theo hướng dẫn sử dụng: " + " ".join(snippets)

        return (
            "Tài liệu hướng dẫn sử dụng hiện tại không cung cấp thông tin cho câu hỏi này. "
            "Bạn nên tham khảo ý kiến bác sĩ hoặc chuyên viên y tế để được tư vấn chính xác."
        )
