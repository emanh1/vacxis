"""Qwen2.5-vLLM backed query rewriter and reference answer generator.

This module is the GPU path of the query-driven dataset generator. It is
designed to run on Colab/Kaggle where a CUDA-enabled vLLM and the
Qwen2.5-7B-Instruct weights are available, while remaining importable
everywhere: vLLM is *never* imported at module load time, so the rest of the
project stays usable on CPU-only machines without vLLM installed.

Two capabilities are exposed:

* :class:`QwenRewriter` -- rewrites raw (out-of-domain) user queries into
  vaccine-domain questions that name a real target vaccine from the corpus.
* :class:`QwenAnswerGenerator` -- fills ``REFERENCE_ANSWER`` and
  ``CLAIM_EVIDENCE_MAP`` from the grounded ``RELEVANT_CHUNKS`` using the same
  JSON contract as the OpenRouter generator in :mod:`answer_generator`.

Both share a single :class:`VLLMClient` so the model is loaded exactly once
per run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .answer_generator import REFERENCE_ANSWER_FORMAT, SYSTEM_PROMPT, UNANSWERABLE_FALLBACK_ANSWER
from .taxonomy import QUERY_TYPE_DEFS

# Lazy, optional vLLM import. vLLM (and torch) are only present in the GPU
# environment (Colab/Kaggle). Keeping this import guarded means
# ``import vacxis_capstone_fa26.testdata`` stays cheap and offline-safe.
try:  # pragma: no cover - exercised only where vLLM is installed
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover
    LLM = None
    SamplingParams = None


# Canonical Qwen instruct model. Overridable via --vllm-model.
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# The ten real query types, pulled from the taxonomy so the model must pick
# from exactly the values the rest of the pipeline understands.
QUERY_TYPE_VALUES: tuple[str, ...] = tuple(t["type"] for t in QUERY_TYPE_DEFS)

# Bump when the rewrite/answer prompts change so a stale checkpoint is
# rejected rather than silently mixing incompatible outputs.
PROMPT_VERSION = 2

# Bump when the verification prompt changes.
VERIFY_VERSION = 1

ANSWERABILITY_VALUES: tuple[str, ...] = (
    "ANSWERABLE",
    "PARTIALLY_ANSWERABLE",
    "UNANSWERABLE",
)


class VLLMError(RuntimeError):
    """Raised when vLLM is requested but unavailable or misconfigured."""


class VLLMCheckpoint:
    """Durable store for the expensive vLLM stages (rewrites, answers, masks).

    A single JSON file lets a crashed or killed run (Colab session timeout,
    GPU OOM) resume instead of regenerating everything. Both stages are
    keyed by the stable source-query ``id`` (never by ``QUERY_ID``, which
    depends on the shuffle seed), so a ``--limit 50`` smoke run warms the
    same checkpoint for the full 1200-query run.

    A ``meta`` header (model, seed, prompt version) is checked on load: a
    checkpoint from a different model/seed/prompt is rejected loudly rather
    than mixed into a new run.

    The ``verifications`` store and ``verify_version`` meta key are optional
    and additive: a checkpoint that predates them still loads, it just
    starts without any cached verification results.
    """

    def __init__(
        self,
        path: Path,
        *,
        model: str,
        seed: int | None,
    ) -> None:
        self.path = Path(path)
        self.meta = {
            "model": model,
            "seed": seed,
            "prompt_version": PROMPT_VERSION,
            "verify_version": VERIFY_VERSION,
        }
        self.rewrites: dict[str, dict[str, Any]] = {}
        self.answers: dict[str, dict[str, Any]] = {}
        self.masks: dict[str, list[dict[str, str]]] = {}
        self.verifications: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _key(source_id: Any) -> str:
        return str(source_id)

    def load(self) -> None:
        """Load an existing checkpoint; raise on incompatible meta."""
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        meta = data.get("meta") or {}
        for key, want in self.meta.items():
            if key in meta and meta[key] != want:
                raise VLLMError(
                    f"Checkpoint {self.path} was written with {key}={meta[key]!r}, "
                    f"but this run uses {key}={want!r}. Use a fresh --checkpoint "
                    "path or re-run without it."
                )
        self.rewrites = {
            self._key(k): v for k, v in (data.get("rewrites") or {}).items()
        }
        self.answers = {
            self._key(k): v for k, v in (data.get("answers") or {}).items()
        }
        self.masks = {
            self._key(k): v for k, v in (data.get("masks") or {}).items()
        }
        self.verifications = {
            self._key(k): v for k, v in (data.get("verifications") or {}).items()
        }
        print(
            f"  -> checkpoint resumed: {len(self.rewrites)} rewrites, "
            f"{len(self.answers)} answers, {len(self.masks)} masks, "
            f"{len(self.verifications)} verifications cached"
        )

    def save(self) -> None:
        """Atomically persist the current cache (tmp file + rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self.meta,
            "rewrites": self.rewrites,
            "answers": self.answers,
            "masks": self.masks,
            "verifications": self.verifications,
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    # Convenience accessors used by the rewriter / answer generator.
    def get_rewrite(self, source_id: Any) -> dict[str, Any] | None:
        return self.rewrites.get(self._key(source_id))

    def put_rewrite(self, source_id: Any, result: dict[str, Any]) -> None:
        self.rewrites[self._key(source_id)] = result

    def get_answer(self, source_id: Any) -> dict[str, Any] | None:
        return self.answers.get(self._key(source_id))

    def put_answer(
        self, source_id: Any, reference_answer: str, claim_map: list
    ) -> None:
        self.answers[self._key(source_id)] = {
            "reference_answer": reference_answer,
            "claim_evidence_map": claim_map,
        }

    def get_mask(self, source_id: Any) -> list[dict[str, str]] | None:
        return self.masks.get(self._key(source_id))

    def put_mask(self, source_id: Any, pairs: list[dict[str, str]]) -> None:
        self.masks[self._key(source_id)] = pairs

    def get_verification(self, source_id: Any) -> dict[str, Any] | None:
        return self.verifications.get(self._key(source_id))

    def put_verification(
        self,
        source_id: Any,
        *,
        answerability: str,
        reference_answer: str,
        claim_map: list[dict[str, Any]],
    ) -> None:
        self.verifications[self._key(source_id)] = {
            "answerability": answerability,
            "reference_answer": reference_answer,
            "claim_evidence_map": claim_map,
        }


class Rewriter(Protocol):
    """Structural type: maps source queries to grounded vaccine queries."""

    def rewrite(self, source_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return one result dict per input query, preserving order.

        Each result carries at least: ``id``, ``vaccine``,
        ``rewritten_query``, ``query_type``, ``answerability``.
        """
        ...


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
        # Fall back to the first balanced {...} block, if any.
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


@dataclass
class VLLMClient:
    """Holds the lazily-constructed vLLM engine and sampling settings.

    Constructed once and shared between the rewriter and the answer
    generator so the (large) model is loaded a single time.
    """

    model: str = DEFAULT_VLLM_MODEL
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    dtype: str = "float16"
    seed: int | None = None
    llm: Any = None

    def ensure_loaded(self) -> None:
        """Load the model on first use; raise a clear error if no vLLM."""
        if self.llm is not None:
            return
        if LLM is None:
            raise VLLMError(
                "vLLM is not installed. Install it on a GPU host "
                "(Colab/Kaggle) with: pip install vllm, or pass "
                "--no-vllm to use the offline rule-based path."
            )
        self.llm = LLM(
            model=self.model,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            trust_remote_code=True,
        )

    def sample(self, **overrides: Any) -> Any:
        if SamplingParams is None:
            raise VLLMError("vLLM SamplingParams unavailable; is vllm installed?")
        kwargs: dict[str, Any] = {"temperature": 0.0, "max_tokens": 512}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        kwargs.update(overrides)
        return SamplingParams(**kwargs)

    def render(self, messages: list[dict[str, str]]) -> str:
        """Render a conversation to a chat-template string for ``generate``.

        ``LLM.generate`` in recent vLLM versions only accepts plain strings
        (or dicts whose ``"prompt"`` is a string), and the convenience
        ``LLM.apply_chat_template`` method has since been removed. The
        tokenizer-level ``apply_chat_template`` is stable across versions,
        so we render through the engine's tokenizer instead.
        """
        self.ensure_loaded()
        rendered = self._tokenizer().apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        # TokenizerPool-style tokenizers return a list for multiple
        # conversations; we always pass one, so a bare string is expected.
        return rendered[0] if isinstance(rendered, list) else rendered

    def _tokenizer(self) -> Any:
        """Resolve the engine's tokenizer across vLLM versions."""
        llm = self.llm
        get_tokenizer = getattr(llm, "get_tokenizer", None)
        if callable(get_tokenizer):
            return get_tokenizer()
        if hasattr(llm, "tokenizer"):
            return llm.tokenizer
        engine = getattr(llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "tokenizer"):
            return engine.tokenizer
        raise VLLMError(
            "Could not locate a tokenizer on the loaded vLLM engine; "
            "update vLLM or pass a chat-template string manually."
        )


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


class QwenRewriter:
    """Rewrites out-of-domain queries into grounded vaccine-domain queries.

    Grounding is *post-hoc*: the model proposes a ``vaccine`` name and the
    caller (:func:`generator.build_queries_from_source`) resolves it against
    ``index_by_vaccine`` so every record points at a real chunk. An
    unknown/empty vaccine degrades gracefully to a random corpus vaccine.
    """

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
        self.client.ensure_loaded()

        vaccines = ", ".join(self.vaccine_names) if self.vaccine_names else "(none)"
        pending: list[tuple[dict[str, Any], int]] = []
        results: list[dict[str, Any] | None] = [None] * len(source_queries)

        for i, sq in enumerate(source_queries):
            if self.checkpoint is not None:
                cached = self.checkpoint.get_rewrite(sq.get("id"))
                if cached is not None:
                    results[i] = cached
                    continue
            pending.append((sq, i))

        # Chunk the pending prompts so each chunk is both one LLM call and a
        # checkpoint save point: a crash mid-rewrite loses at most one
        # chunk instead of the whole stage.
        for start in range(0, len(pending), self.rewrite_batch_size):
            chunk = pending[start : start + self.rewrite_batch_size]
            prompts = [self._build_prompt(sq, vaccines) for sq, _ in chunk]
            sampling = self.client.sample()
            outputs = self.client.llm.generate(prompts, sampling)
            for (sq, idx), out in zip(chunk, outputs):
                raw_text = out.outputs[0].text if out and out.outputs else ""
                parsed = _extract_json(raw_text)
                res = _normalize_result(parsed or {}, sq.get("id"))
                results[idx] = res
                if self.checkpoint is not None:
                    self.checkpoint.put_rewrite(sq.get("id"), res)
            if self.checkpoint is not None:
                self.checkpoint.save()

        return [r if r is not None else {} for r in results]


# Answers reuse the exact JSON schema and system prompt of the OpenRouter
# generator so the two paths produce byte-compatible record fields.
_JSON_SCHEMA_HINT = json.dumps(REFERENCE_ANSWER_FORMAT["json_schema"]["schema"])

# The verifier decides the answerability label too, so its schema extends the
# reference-answer schema with that one enum field.
_VERIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answerability": {"type": "string", "enum": list(ANSWERABILITY_VALUES)},
        **REFERENCE_ANSWER_FORMAT["json_schema"]["schema"]["properties"],
    },
    "required": [
        "answerability",
        *REFERENCE_ANSWER_FORMAT["json_schema"]["schema"]["required"],
    ],
}
_VERIFY_SCHEMA_HINT = json.dumps(_VERIFY_SCHEMA)


class QwenAnswerGenerator:
    """Offline-capable, vLLM-backed reference answer generator.

    Implements the :class:`AnswerGenerator` protocol
    (``generate_answers(queries) -> queries``). For UNANSWERABLE queries or
    queries without a score>=3 chunk it mirrors
    :class:`ExtractiveAnswerGenerator` (fallback answer, empty claim map).

    LLM-backed answers are computed in batches (``batch_size``) so 1200
    queries do not issue 1200 single-prompt calls. Each query must carry a
    ``SOURCE_QUERY_ID`` (the stable parquet ``id``) for checkpointing to
    apply; without it, checkpointing is simply bypassed.
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

    def _answerable_with_evidence(self, q: dict[str, Any]) -> bool:
        return q.get("ANSWERABILITY") != "UNANSWERABLE" and any(
            c.get("RELEVANCE_SCORE", 0) >= 3 for c in q.get("RELEVANT_CHUNKS", [])
        )

    def generate_answers(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for q in queries:
            if not self._answerable_with_evidence(q):
                q["REFERENCE_ANSWER"] = UNANSWERABLE_FALLBACK_ANSWER
                q["CLAIM_EVIDENCE_MAP"] = []

        # Restore answers already computed by an earlier (crashed) run so a
        # resumed run emits the cached text instead of leaving None.
        if self.checkpoint is not None:
            for q in queries:
                if self._answerable_with_evidence(q):
                    cached = self.checkpoint.get_answer(q.get("SOURCE_QUERY_ID"))
                    if cached is not None:
                        q["REFERENCE_ANSWER"] = cached["reference_answer"]
                        q["CLAIM_EVIDENCE_MAP"] = cached["claim_evidence_map"]

        to_call: list[dict[str, Any]] = [
            q
            for q in queries
            if self._answerable_with_evidence(q)
            and (
                self.checkpoint is None
                or self.checkpoint.get_answer(q.get("SOURCE_QUERY_ID")) is None
            )
        ]

        for i in range(0, len(to_call), self.batch_size):
            batch = to_call[i : i + self.batch_size]
            self._generate_batch(batch)
            if self.checkpoint is not None:
                self.checkpoint.save()

        return queries

    def _generate_batch(self, batch: list[dict[str, Any]]) -> None:
        self.client.ensure_loaded()
        prompts: list[str] = []
        for q in batch:
            prompts.append(self._build_answer_prompt(q))

        sampling = self.client.sample(max_tokens=1024)
        outputs = self.client.llm.generate(prompts, sampling)

        for q, out in zip(batch, outputs):
            text = out.outputs[0].text if out and out.outputs else ""
            data = _extract_json(text)
            if data and "reference_answer" in data:
                answer = str(data.get("reference_answer", "")).strip()
                claims = data.get("claim_evidence_map", []) or []
            else:
                answer = UNANSWERABLE_FALLBACK_ANSWER
                claims = []
            q["REFERENCE_ANSWER"] = answer
            q["CLAIM_EVIDENCE_MAP"] = claims
            if self.checkpoint is not None:
                self.checkpoint.put_answer(q.get("SOURCE_QUERY_ID"), answer, claims)

    def _build_answer_prompt(self, q: dict[str, Any]) -> str:
        relevant = [
            c
            for c in q.get("RELEVANT_CHUNKS", [])
            if c.get("RELEVANCE_SCORE", 0) >= 3
        ]
        chunks_context = "\n\n---\n\n".join(
            f"CHUNK_ID: {c.get('CHUNK_ID')}\nNỘI DUNG:\n{c.get('TEXT')}"
            for c in relevant
        )
        user = (
            f"ANSWERABILITY: {q.get('ANSWERABILITY')}\nCÂU HỎI: {q.get('QUERY')}\n\n"
            f"CÁC ĐOẠN CHUNKS CUNG CẤP:\n{chunks_context}\n\n"
            "Trả JSON theo schema:\n"
            f"{_JSON_SCHEMA_HINT}"
        )
        return self.client.render(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
        )


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

    The model chooses which symptoms to replace (it sees the query and the
    relevant chunks); the caller validates each pair against the record via
    :func:`postprocess.apply_replacements`, so an invented replacement is
    dropped rather than shipped. Output is cached in the checkpoint like
    the other two LLM stages.
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
        results: list[list[dict[str, str]] | None] = [None] * len(records)
        pending: list[tuple[int, dict[str, Any]]] = []
        for i, record in enumerate(records):
            if self.checkpoint is not None:
                cached = self.checkpoint.get_mask(record.get("SOURCE_QUERY_ID"))
                if cached is not None:
                    results[i] = cached
                    continue
            pending.append((i, record))

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            pairs = self._generate_batch(batch)
            for (idx, record), result in zip(batch, pairs):
                results[idx] = result
                if self.checkpoint is not None:
                    self.checkpoint.put_mask(
                        record.get("SOURCE_QUERY_ID"), result
                    )
            if self.checkpoint is not None:
                self.checkpoint.save()

        return [p if p is not None else [] for p in results]

    def _generate_batch(
        self, batch: list[tuple[int, dict[str, Any]]]
    ) -> list[list[dict[str, str]]]:
        from .postprocess import relevant_chunk_texts

        self.client.ensure_loaded()
        prompts: list[str] = []
        for _, record in batch:
            chunks_context = "\n\n---\n\n".join(
                t for t in relevant_chunk_texts(record) if t
            )
            user = _MASK_USER_TMPL.format(
                query=record.get("QUERY", ""),
                chunks=chunks_context or "(không có chứng cứ)",
            )
            prompts.append(
                self.client.render(
                    [
                        {"role": "system", "content": _MASK_SYSTEM},
                        {"role": "user", "content": user},
                    ]
                )
            )

        sampling = self.client.sample(max_tokens=512)
        outputs = self.client.llm.generate(prompts, sampling)

        pairs: list[list[dict[str, str]]] = []
        for out in outputs:
            text = out.outputs[0].text if out and out.outputs else ""
            data = _extract_json(text) or {}
            raw_pairs = data.get("replacements")
            cleaned: list[dict[str, str]] = []
            if isinstance(raw_pairs, list):
                for item in raw_pairs:
                    if isinstance(item, dict):
                        cleaned.append(
                            {
                                "original": str(item.get("original", "")).strip(),
                                "replacement": str(
                                    item.get("replacement", "")
                                ).strip(),
                            }
                        )
            pairs.append(cleaned)
        return pairs


_VERIFY_SYSTEM = """Bạn là chuyên gia kiểm định một bản ghi trong bộ dữ liệu đánh giá RAG về vắc xin.

Bạn nhận MỘT câu hỏi cùng các đoạn CHỨNG CỨ (chunks) là bằng chứng duy nhất được phép dùng.

Hãy làm hai việc:

1. Chọn ANSWERABILITY cho câu hỏi, dựa DUY NHẤT vào mức độ mà các chunks trả lời được câu hỏi:
   - ANSWERABLE: các chunks trả lời đầy đủ câu hỏi.
   - PARTIALLY_ANSWERABLE: các chunks chỉ trả lời được một phần của câu hỏi.
   - UNANSWERABLE: các chunks không trả lời được câu hỏi (kể cả khi có chunk liên quan đến cùng loại vaccine).

2. Viết REFERENCE_ANSWER trả lời câu hỏi, chỉ dùng thông tin trong các chunks. Sau mỗi khẳng định, chèn [CHUNK_ID] và cung cấp claim cùng evidence_quote là trích dẫn NGUYÊN VĂN từ chunk tương ứng.
   - Nếu ANSWERABILITY là UNANSWERABLE, trả lời rằng tài liệu không cung cấp thông tin và để claim_evidence_map rỗng.
   - Nếu là PARTIALLY_ANSWERABLE, chỉ trả lời phần có bằng chứng và nêu rõ phần thiếu.
   - evidence_quote phải là một đoạn con nguyên văn của chunk, không được diễn giải lại.

Chỉ trả JSON, không dùng Markdown."""

_VERIFY_USER_TMPL = """CÂU HỎI: {query}

CÁC CHUNKS BẰNG CHỨNG:
{chunks}

Trả JSON theo schema:
{schema}"""


class QwenLabelVerifier:
    """Re-derives the answerability label and the reference answer.

    The symptom masker rewrites the query but leaves the label and the
    answer describing the pre-masking question. This stage runs after the
    masking step and re-derives both from the *final* query and its
    evidence, so the shipped record is internally consistent.

    The model returns the answerability label, the reference answer and the
    claim/evidence map in one call. The caller applies the result with
    :func:`relabel.apply_verification` and then checks it with
    :func:`relabel.verify_record`.

    Results are cached in the checkpoint like the other LLM stages, so a
    resumed run does not recompute them.
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

    def verify(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return one verification result per record, in input order."""
        results: list[dict[str, Any] | None] = [None] * len(records)
        pending: list[tuple[int, dict[str, Any]]] = []

        for i, record in enumerate(records):
            if self.checkpoint is not None:
                cached = self.checkpoint.get_verification(record.get("SOURCE_QUERY_ID"))
                if cached is not None:
                    results[i] = cached
                    continue
            pending.append((i, record))

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            for (idx, record), result in zip(batch, self._generate_batch(batch)):
                results[idx] = result
                if self.checkpoint is not None:
                    self.checkpoint.put_verification(
                        record.get("SOURCE_QUERY_ID"),
                        answerability=result["answerability"],
                        reference_answer=result["reference_answer"],
                        claim_map=result["claim_evidence_map"],
                    )
            if self.checkpoint is not None:
                self.checkpoint.save()

        return [r if r is not None else self._fallback() for r in results]

    def _fallback(self) -> dict[str, Any]:
        return {
            "answerability": "UNANSWERABLE",
            "reference_answer": UNANSWERABLE_FALLBACK_ANSWER,
            "claim_evidence_map": [],
        }

    def _generate_batch(
        self, batch: list[tuple[int, dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        self.client.ensure_loaded()
        prompts = [self._build_prompt(record) for _, record in batch]

        sampling = self.client.sample(max_tokens=1024)
        outputs = self.client.llm.generate(prompts, sampling)

        results: list[dict[str, Any]] = []
        for out in outputs:
            text = out.outputs[0].text if out and out.outputs else ""
            data = _extract_json(text) or {}
            answerability = str(data.get("answerability", "")).strip().upper()
            if answerability not in ANSWERABILITY_VALUES:
                answerability = "UNANSWERABLE"
            if answerability == "UNANSWERABLE":
                results.append(
                    {
                        "answerability": answerability,
                        "reference_answer": str(
                            data.get("reference_answer", "")
                        ).strip()
                        or UNANSWERABLE_FALLBACK_ANSWER,
                        "claim_evidence_map": [],
                    }
                )
                continue
            results.append(
                {
                    "answerability": answerability,
                    "reference_answer": str(
                        data.get("reference_answer", "")
                    ).strip()
                    or UNANSWERABLE_FALLBACK_ANSWER,
                    "claim_evidence_map": data.get("claim_evidence_map", []) or [],
                }
            )
        return results

    def _build_prompt(self, record: dict[str, Any]) -> str:
        from .postprocess import relevant_chunk_texts

        chunks_context = "\n\n---\n\n".join(
            f"CHUNK_ID: {c.get('CHUNK_ID')}\nNỘI DUNG:\n{c.get('TEXT')}"
            for c in record.get("RELEVANT_CHUNKS", []) or []
            if isinstance(c, dict) and c.get("RELEVANCE_SCORE", 0) >= 3
        )
        user = _VERIFY_USER_TMPL.format(
            query=record.get("QUERY", ""),
            chunks=chunks_context or "(không có chứng cứ)",
            schema=_VERIFY_SCHEMA_HINT,
        )
        return self.client.render(
            [
                {"role": "system", "content": _VERIFY_SYSTEM},
                {"role": "user", "content": user},
            ]
        )
