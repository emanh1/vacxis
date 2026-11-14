"""The three LLM stages over the shared batched runner.

Each stage is thin: it only builds per-item prompts and parses outputs.
All checkpoint/restore, batching, and save-per-batch mechanics live in
:func:`run_batched_stage`, so a crash mid-stage loses at most one batch.

Stages:

* :class:`QwenRewriter` -- rewrites out-of-domain queries into grounded
  vaccine-domain questions.
* :class:`QwenSymptomMasker` -- replaces query symptoms that do not occur in
  the record's evidence with symptoms that do (for the masked tail).
* :class:`QwenAnswerStage` -- the unified answer+label pass: one call per
  record that derives ``ANSWERABILITY``, ``REFERENCE_ANSWER`` and
  ``CLAIM_EVIDENCE_MAP`` from the *final* (possibly masked) query and its
  evidence. This replaced the old separate answer generator and the
  post-mask label verifier, which existed only to repair the inconsistency
  the mask introduced.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Literal

from .answer_generator import REFERENCE_ANSWER_FORMAT, UNANSWERABLE_FALLBACK_ANSWER
from .checkpoint import VLLMCheckpoint
from .taxonomy import QUERY_TYPE_DEFS
from .vllm_client import VLLMClient, VLLMError

# Bump when the stage prompts change so a stale checkpoint is rejected rather
# than silently mixing incompatible outputs.
PROMPT_VERSION = 3

# The real query types, pulled from the taxonomy so the model must pick from
# exactly the values the rest of the pipeline understands.
QUERY_TYPE_VALUES: tuple[str, ...] = tuple(t["type"] for t in QUERY_TYPE_DEFS)

ANSWERABILITY_VALUES: tuple[str, ...] = (
    "ANSWERABLE",
    "PARTIALLY_ANSWERABLE",
    "UNANSWERABLE",
)


# JSON-ish block the model may wrap structured output in.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a model JSON payload, tolerating ``` fences."""
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# The shared batched primitive.
# ---------------------------------------------------------------------------
def run_batched_stage(
    items: list[Any],
    *,
    client: VLLMClient,
    checkpoint: VLLMCheckpoint | None,
    store: Literal["rewrites", "masks", "answers"],
    key_of: Callable[[Any], Any | None],
    build_prompt: Callable[[Any], str],
    parse: Callable[[str, Any], Any | None],
    default: Callable[[Any], Any],
    batch_size: int,
    max_tokens: int,
) -> list[Any]:
    """Run one LLM stage over ``items``, returning one value per input item.

    Values come back in input order. Per item:

    * checkpoint load is owned by the caller; this function only reads and
      saves.
    * ``key = key_of(item)``; if a checkpoint is present and the key is not
      ``None`` and a value is cached, it is used and the item is skipped.
      Items with ``key is None`` always hit the LLM and are never written.
    * pending items are chunked in ``batch_size`` slices; each slice is one
      ``client.llm.generate`` call and one ``checkpoint.save``.
    * ``parse(raw_text, item)`` returns the value, or ``None`` for an
      unparseable payload; a ``None`` result uses ``default(item)`` and is
      *not* cached, so a transient model malfunction is retried on resume.
    """
    if not items:
        return []
    client.ensure_loaded()

    results: list[Any] = [None] * len(items)
    pending: list[tuple[Any, int]] = []
    for i, item in enumerate(items):
        key = key_of(item)
        if checkpoint is not None and key is not None:
            cached = checkpoint.get(store, key)
            if cached is not None:
                results[i] = cached
                continue
        pending.append((item, i))

    for start in range(0, len(pending), max(1, batch_size)):
        batch = pending[start : start + max(1, batch_size)]
        prompts = [build_prompt(item) for item, _ in batch]
        sampling = client.sample(max_tokens=max_tokens)
        outputs = client.llm.generate(prompts, sampling)
        if len(outputs) != len(batch):
            raise VLLMError(
                f"LLM returned {len(outputs)} outputs for {len(batch)} prompts; "
                "refusing to ship short records."
            )
        for (item, idx), out in zip(batch, outputs):
            raw_text = out.outputs[0].text if out and out.outputs else ""
            value = parse(raw_text, item)
            if value is None:
                value = default(item)
            else:
                key = key_of(item)
                if checkpoint is not None and key is not None:
                    checkpoint.put(store, key, value)
            results[idx] = value
        if checkpoint is not None:
            checkpoint.save()

    return results


# ---------------------------------------------------------------------------
# Rewrite stage.
# ---------------------------------------------------------------------------
_REWRITE_SYSTEM = """You are a medical query adapter. The user gives a Vietnamese
medical question that is NOT necessarily about vaccines. Rewrite it into a
natural Vietnamese question ABOUT VACCINES that preserves the user's original
clinical intent and any named symptoms/conditions. The rewritten question MUST
name exactly one real vaccine and target one of the listed query types.
Respond with a single JSON object and nothing else."""

_REWRITE_USER_TMPL = """Available vaccines:
{vaccines}

Allowed query_type values: {qtypes}
Allowed answerability values: ANSWERABLE, PARTIALLY_ANSWERABLE, UNANSWERABLE

Source query (id={qid}):
"""


def _normalize_result(raw: dict[str, Any], fallback_id: Any) -> dict[str, Any]:
    """Coerce a raw model payload into the rewriter result contract."""
    query_type = str(raw.get("query_type", "")).strip().upper()
    if query_type not in QUERY_TYPE_VALUES:
        query_type = "PRECAUTION"  # safe, broad category for unparseable output

    answerability = str(raw.get("answerability", "ANSWERABLE")).strip().upper()
    if answerability not in ANSWERABILITY_VALUES:
        answerability = "ANSWERABLE"

    return {
        "id": raw.get("id", fallback_id),
        "vaccine": str(raw.get("vaccine", "")).strip(),
        "rewritten_query": str(raw.get("rewritten_query", "")).strip(),
        "query_type": query_type,
        "answerability": answerability,
        "rationale": str(raw.get("rationale", "")).strip(),
    }


class QwenRewriter:
    """Rewrites out-of-domain queries into grounded vaccine-domain queries."""

    def __init__(
        self,
        client: VLLMClient,
        *,
        vaccine_names: list[str],
        checkpoint: VLLMCheckpoint | None = None,
        rewrite_batch_size: int = 256,
    ) -> None:
        self.client = client
        self.vaccine_names = list(vaccine_names)
        self.checkpoint = checkpoint
        self.rewrite_batch_size = max(1, rewrite_batch_size)

    def _build_prompt(self, sq: dict[str, Any], vaccines: str) -> str:
        """Render the rewrite conversation for one source query."""
        user = _REWRITE_USER_TMPL.format(
            vaccines=vaccines,
            qtypes=", ".join(QUERY_TYPE_VALUES),
            qid=sq.get("id"),
        )
        user += f"{sq.get('query', '').strip()}\n\n"
        user += (
            "Return JSON: {\"id\": <int>, \"vaccine\": \"<name from the "
            "available list>\", \"rewritten_query\": \"<Vietnamese "
            "vaccine question>\", \"query_type\": \"<allowed value>\", "
            "\"answerability\": \"<allowed value>\", \"rationale\": "
            "\"<short reason>\"}"
        )
        return self.client.render(
            [
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": user},
            ]
        )

    def rewrite(self, source_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not source_queries:
            return []
        vaccines = ", ".join(self.vaccine_names) if self.vaccine_names else "(none)"

        def build_prompt(sq: dict[str, Any]) -> str:
            return self._build_prompt(sq, vaccines)

        def parse(raw_text: str, sq: dict[str, Any]) -> dict[str, Any]:
            data = _extract_json(raw_text)
            return _normalize_result(data or {}, sq.get("id"))

        def default(sq: dict[str, Any]) -> dict[str, Any]:
            return _normalize_result({}, sq.get("id"))

        return run_batched_stage(
            source_queries,
            client=self.client,
            checkpoint=self.checkpoint,
            store="rewrites",
            key_of=lambda sq: sq.get("id"),
            build_prompt=build_prompt,
            parse=parse,
            default=default,
            batch_size=self.rewrite_batch_size,
            max_tokens=512,
        )


# ---------------------------------------------------------------------------
# Symptom-masking stage.
# ---------------------------------------------------------------------------
_MASK_SYSTEM = """Bạn là người hiệu chỉnh câu hỏi y tế bằng tiếng Việt. Với một
câu hỏi và các đoạn CHỨNG CỨ đi kèm, tìm các triệu chứng được nêu trong
câu hỏi mà KHÔNG xuất hiện trong chứng cứ. Thay mỗi triệu chứng đó bằng một
triệu chứng có mặt trong chứng cứ sao cho câu hỏi vẫn tự nhiên. Chỉ trả JSON,
không dùng Markdown."""

_MASK_USER_TMPL = """CÂU HỎI: {query}

CÁC ĐOẠN CHỨNG CỨ (chỉ các chunk có điểm liên quan cao):
{chunks}

Trả JSON: {{"replacements": [{{"original": "triệu chứng trong câu hỏi",
"replacement": "triệu chứng trong chứng cứ"}}, ...]}}. Nếu không có gì
cần thay, trả {{"replacements": []}}."""


class QwenSymptomMasker:
    """LLM-backed symptom replacement for masked dataset records.

    Records with no relevant chunk text (the UNANSWERABLE records) are
    skipped deterministically: there is no evidence to ground a
    replacement against, and the caller would drop the pair anyway, so an
    LLM call on a "(không có chứng cứ)" prompt can only hallucinate.
    """

    def __init__(
        self,
        client: VLLMClient,
        *,
        checkpoint: VLLMCheckpoint | None = None,
        batch_size: int = 64,
    ) -> None:
        self.client = client
        self.checkpoint = checkpoint
        self.batch_size = max(1, batch_size)

    def mask(self, records: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
        """Return one list of ``{original, replacement}`` pairs per record."""
        if not records:
            return []

        # Partition: only records with a score-3 chunk text reach the LLM.
        from .postprocess import relevant_chunk_texts

        needs_llm: list[tuple[int, dict[str, Any]]] = []
        for i, record in enumerate(records):
            if not relevant_chunk_texts(record):
                continue
            needs_llm.append((i, record))

        def build_prompt(_: tuple[int, dict[str, Any]]) -> str:
            _idx, record = _
            chunks_context = "\n\n---\n\n".join(
                t for t in relevant_chunk_texts(record) if t
            )
            user = _MASK_USER_TMPL.format(
                query=record.get("QUERY", ""),
                chunks=chunks_context or "(không có chứng cứ)",
            )
            return self.client.render(
                [
                    {"role": "system", "content": _MASK_SYSTEM},
                    {"role": "user", "content": user},
                ]
            )

        def parse(raw_text: str, item: tuple[int, dict[str, Any]]) -> list[dict[str, str]]:
            _idx, record = item
            data = _extract_json(raw_text) or {}
            raw_pairs = data.get("replacements")
            cleaned: list[dict[str, str]] = []
            if isinstance(raw_pairs, list):
                for entry in raw_pairs:
                    if isinstance(entry, dict):
                        cleaned.append(
                            {
                                "original": str(entry.get("original", "")).strip(),
                                "replacement": str(
                                    entry.get("replacement", "")
                                ).strip(),
                            }
                        )
            return cleaned

        def default(_item: tuple[int, dict[str, Any]]) -> list[dict[str, str]]:
            return []

        values = run_batched_stage(
            [pair for pair in needs_llm],
            client=self.client,
            checkpoint=self.checkpoint,
            store="masks",
            key_of=lambda item: item[1].get("SOURCE_QUERY_ID"),
            build_prompt=build_prompt,
            parse=parse,
            default=default,
            batch_size=self.batch_size,
            max_tokens=512,
        )

        # No-chunk records are skipped; the LLM was never called, so there is
        # nothing to cache and nothing to resume — they are not part of
        # ``needs_llm`` and default to ``[]`` here.
        results: list[list[dict[str, str]]] = [[] for _ in records]
        for (idx, _record), value in zip(needs_llm, values):
            results[idx] = value
        return results


# ---------------------------------------------------------------------------
# Unified answer + label stage.
# ---------------------------------------------------------------------------
_JSON_SCHEMA = REFERENCE_ANSWER_FORMAT["json_schema"]["schema"]

# The stage decides the answerability label too, so its schema extends the
# reference-answer schema with that one enum field.
_UNIFIED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answerability": {"type": "string", "enum": list(ANSWERABILITY_VALUES)},
        **_JSON_SCHEMA["properties"],
    },
    "required": ["answerability", *_JSON_SCHEMA["required"]],
}
_UNIFIED_SCHEMA_HINT = json.dumps(_UNIFIED_SCHEMA)

_ANSWER_SYSTEM = """Bạn là chuyên gia tổng hợp và kiểm định một bản ghi trong bộ dữ
liệu đánh giá RAG về vắc xin. Bạn nhận MỘT câu hỏi cùng các đoạn CHỨNG CỨ
(chunks) là bằng chứng duy nhất được phép dùng.

Hãy làm hai việc:

1. Chọn ANSWERABILITY cho câu hỏi, dựa DUY NHẤT vào mức độ mà các chunks trả
lời được câu hỏi:
   - ANSWERABLE: các chunks trả lời đầy đủ câu hỏi.
   - PARTIALLY_ANSWERABLE: các chunks chỉ trả lời được một phần của câu hỏi.
   - UNANSWERABLE: các chunks không trả lời được câu hỏi.

2. Viết REFERENCE_ANSWER trả lời câu hỏi, chỉ dùng thông tin trong các chunks.
   Sau mỗi khẳng định, chèn [CHUNK_ID] và cung cấp claim cùng evidence_quote
   là trích dẫn NGUYÊN VĂN từ chunk tương ứng (evidence_quote phải là một
   đoạn con nguyên văn, không diễn giải lại).
   - Nếu UNANSWERABLE, trả lời rằng tài liệu không cung cấp thông tin và để
     claim_evidence_map rỗng.
   - Nếu PARTIALLY_ANSWERABLE, chỉ trả lời phần có bằng chứng và nêu rõ phần
     thiếu.

Giữ nguyên phạm vi câu hỏi, không tự suy diễn. Chỉ trả JSON, không dùng
Markdown."""


class QwenAnswerStage:
    """Derives the label, reference answer and claim map from the final query.

    Records with no relevant chunk text (``RELEVANCE_SCORE >= 3``) are
    answered deterministically (UNANSWERABLE fallback, evidence cleared)
    without an LLM call, and are not cached. Every other record is run
    through the model in batches; the result is written back in place via
    :func:`validator.apply_labels`. An unparseable payload falls back to the
    UNANSWERABLE answer for this run but is not cached, so a resumed run
    retries it.
    """

    def __init__(
        self,
        client: VLLMClient,
        *,
        checkpoint: VLLMCheckpoint | None = None,
        batch_size: int = 64,
    ) -> None:
        self.client = client
        self.checkpoint = checkpoint
        self.batch_size = max(1, batch_size)

    @staticmethod
    def _has_evidence(record: dict[str, Any]) -> bool:
        return any(
            c.get("RELEVANCE_SCORE", 0) >= 3
            for c in record.get("RELEVANT_CHUNKS", [])
            if isinstance(c, dict)
        )

    def answer(self, records: list[dict[str, Any]]) -> None:
        """Fill label/answer/claim map on each record, in place."""
        from .validator import apply_labels

        if not records:
            return

        needs_llm = [r for r in records if self._has_evidence(r)]
        for r in records:
            if not self._has_evidence(r):
                apply_labels(
                    r,
                    answerability="UNANSWERABLE",
                    reference_answer=UNANSWERABLE_FALLBACK_ANSWER,
                    claim_evidence_map=[],
                )

        if not needs_llm:
            return

        values = run_batched_stage(
            needs_llm,
            client=self.client,
            checkpoint=self.checkpoint,
            store="answers",
            key_of=lambda r: r.get("SOURCE_QUERY_ID"),
            build_prompt=self._build_prompt,
            parse=self._parse,
            default=self._fallback,
            batch_size=self.batch_size,
            max_tokens=1024,
        )
        for r, v in zip(needs_llm, values):
            apply_labels(
                r,
                answerability=v["answerability"],
                reference_answer=v["reference_answer"],
                claim_evidence_map=v["claim_evidence_map"],
            )

    def _fallback(self, _record: dict[str, Any]) -> dict[str, Any]:
        return {
            "answerability": "UNANSWERABLE",
            "reference_answer": UNANSWERABLE_FALLBACK_ANSWER,
            "claim_evidence_map": [],
        }

    def _parse(
        self, raw_text: str, record: dict[str, Any]
    ) -> dict[str, Any] | None:
        data = _extract_json(raw_text)
        answerability = str((data or {}).get("answerability", "")).strip().upper()
        if not data or answerability not in ANSWERABILITY_VALUES:
            return None  # unparseable: fall back for this run, do not cache
        if answerability == "UNANSWERABLE":
            answer = str((data or {}).get("reference_answer", "")).strip()
            return {
                "answerability": answerability,
                "reference_answer": answer or UNANSWERABLE_FALLBACK_ANSWER,
                "claim_evidence_map": [],
            }
        return {
            "answerability": answerability,
            "reference_answer": str(
                (data or {}).get("reference_answer", "")
            ).strip()
            or UNANSWERABLE_FALLBACK_ANSWER,
            "claim_evidence_map": (data or {}).get("claim_evidence_map", []) or [],
        }

    def _build_prompt(self, record: dict[str, Any]) -> str:
        chunks_context = "\n\n---\n\n".join(
            f"CHUNK_ID: {c.get('CHUNK_ID')}\nNỘI DUNG:\n{c.get('TEXT')}"
            for c in record.get("RELEVANT_CHUNKS", []) or []
            if isinstance(c, dict) and c.get("RELEVANCE_SCORE", 0) >= 3
        )
        user = (
            f"CÂU HỎI: {record.get('QUERY', '')}\n\n"
            f"CÁC CHUNKS BẰNG CHỨNG:\n{chunks_context or '(không có chứng cứ)'}\n\n"
            f"Trả JSON theo schema:\n{_UNIFIED_SCHEMA_HINT}"
        )
        return self.client.render(
            [
                {"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user", "content": user},
            ]
        )
