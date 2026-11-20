"""OpenRouter-based judge LLM adapter for DeepEval metrics.

DeepEval's LLM-judged metrics (Answer Relevancy, Faithfulness, the Contextual
family, GEval) need a judge model. The project calls OpenRouter through
``OPENROUTER_API_KEY`` + ``VACXIS_LLM_BASE_URL``, so this adapter drives DeepEval
with the same endpoint instead of the library's OpenAI default. The model name
defaults to ``VACXIS_EVALUATION_MODEL`` and falls back to the generation model.
"""
from __future__ import annotations

import os
from typing import Any

from json_repair import json_repair
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ValidationError

from deepeval.models import DeepEvalBaseLLM
import time
import asyncio
from vacxis_capstone_fa26.resilience import RetryPolicy

JUDGE_API_KEY_ENV = "OPENROUTER_API_KEY"
JUDGE_BASE_URL_ENV = "VACXIS_LLM_BASE_URL"
JUDGE_MODEL_ENV = "VACXIS_EVALUATION_MODEL"
_FALLBACK_MODEL_ENV = "VACXIS_GENERATION_MODEL"
_DEFAULT_JUDGE_MODEL = "gpt-oss-120b"


def resolve_judge_model(model: str | None = None) -> str:
    """Pick the judge model: explicit arg, then eval var, then generation var.

    Falls back to a sane default so constructing an evaluator never raises on a
    missing model; a missing API key still fails when a judge call actually runs.
    """
    explicit = (model or "").strip()
    if explicit:
        return explicit
    eval_model = os.getenv(JUDGE_MODEL_ENV, "").strip()
    if eval_model:
        return eval_model
    fallback = os.getenv(_FALLBACK_MODEL_ENV, "").strip()
    if fallback:
        return fallback
    return _DEFAULT_JUDGE_MODEL


def require_judge_credentials(model: str | None = None) -> None:
    """Ensure the OpenRouter key and base URL needed by the judge are present."""
    key = os.getenv(JUDGE_API_KEY_ENV, "").strip()
    if not key:
        raise ValueError(f"Set {JUDGE_API_KEY_ENV} to run DeepEval LLM-judged metrics.")
    resolve_judge_model(model)


class OpenRouterJudgeLLM(DeepEvalBaseLLM):
    """DeepEvalBaseLLM backed by an OpenAI-compatible OpenRouter endpoint."""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        schema_retries: int = 3,
        retry_policy: RetryPolicy | None = None,
        timeout: float = 360,
    ) -> None:
        self.retry_policy = retry_policy or RetryPolicy()
        self.timeout = timeout
        self.model_name = resolve_judge_model(model)
        self.api_key = (api_key or os.getenv(JUDGE_API_KEY_ENV, "")).strip()
        self.base_url = (
            base_url or os.getenv(JUDGE_BASE_URL_ENV, "https://openrouter.ai/api/v1")
        ).rstrip("/")
        self.temperature = temperature
        self.schema_retries = max(0, schema_retries)
        self._sync_client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None
        super().__init__(model=self.model_name)

    def load_model(self) -> OpenAI:
        if self._sync_client is None:
            # A missing key is surfaced as an empty string so the OpenAI client
            # can still be constructed; the actual request then fails with a
            # clear "401" rather than a construction-time credentials error.
            self._sync_client = OpenAI(
                api_key=self.api_key or "OPENROUTER_API_KEY_MISSING",
                base_url=self.base_url,
                max_retries=0,
                timeout=self.timeout,
            )
        return self._sync_client

    def _load_async_model(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self.api_key or "OPENROUTER_API_KEY_MISSING",
                base_url=self.base_url,
                max_retries=0,
                timeout=self.timeout,
            )
        return self._async_client

    def get_model_name(self) -> str:
        return self.model_name

    def _build_schema_retry_prompt(
        self,
        original_prompt: str,
        raw_response: str,
        schema: type[BaseModel],
        error: ValidationError,
    ) -> str:
        """Ask the judge to correct an invalid structured response."""

        validation_errors = error.errors(
            include_url=False,
            include_context=False,
        )

        return f"""
    Your previous response did not conform to the required output schema.

    Original task:
    {original_prompt}

    Required JSON schema:
    {schema.model_json_schema()}

    Previous invalid response:
    {raw_response}

    Validation errors:
    {validation_errors}

    Please correct your previous response.

    Requirements:
    - Return a complete, valid JSON object.
    - Include ALL required fields.
    - Ensure every nested object conforms to the schema.
    - Do not omit any items from arrays.
    - Do not include explanations, markdown fences, or additional text.
    - Preserve the original meaning and information wherever possible.

    Return ONLY the corrected JSON.
    """.strip()

    @staticmethod
    def _parse_schema(raw: str, schema: type[BaseModel]) -> BaseModel:
        """Parse the judge's text into ``schema``, tolerating stray prose.

        DeepEval passes structured-output schemas; Harmony-style endpoints
        (e.g. gpt-oss) can corrupt forced JSON, so parse with json_repair and
        wrap a bare string when the schema only carries a ``response`` field.
        """
        parsed = json_repair.loads(raw)
        if (
            not isinstance(parsed, dict)
            and list(schema.model_fields.keys()) == ["response"]
        ):
            return schema.model_validate({"response": raw.strip()})
        return schema.model_validate(parsed)

    def _build_request_kwargs(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        client = self.load_model()

        if schema is None:
            response = self.retry_policy.run(lambda: client.chat.completions.create(
                **self._build_request_kwargs(prompt)
            ))
            return response.choices[0].message.content or ""

        current_prompt = prompt

        for attempt in range(self.schema_retries + 1):
            response = self.retry_policy.run(lambda: client.chat.completions.create(
                **self._build_request_kwargs(current_prompt)
            ))
            content = response.choices[0].message.content or ""

            try:
                return self._parse_schema(content, schema)

            except ValidationError as exc:

                if attempt >= self.schema_retries:
                    break

                current_prompt = self._build_schema_retry_prompt(
                    original_prompt=prompt,
                    raw_response=content,
                    schema=schema,
                    error=exc,
                )
                time.sleep(self.retry_policy.delay(attempt))

        raise ValueError("Judge schema correction retries exhausted.")

    async def a_generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        client = self._load_async_model()

        if schema is None:
            response = await self.retry_policy.arun(lambda: client.chat.completions.create(
                **self._build_request_kwargs(prompt)
            ))
            return response.choices[0].message.content or ""

        current_prompt = prompt

        for attempt in range(self.schema_retries + 1):
            response = await self.retry_policy.arun(lambda: client.chat.completions.create(
                **self._build_request_kwargs(current_prompt)
            ))
            content = response.choices[0].message.content or ""

            try:
                return self._parse_schema(content, schema)

            except ValidationError as exc:

                if attempt >= self.schema_retries:
                    break

                current_prompt = self._build_schema_retry_prompt(
                    original_prompt=prompt,
                    raw_response=content,
                    schema=schema,
                    error=exc,
                )
                await asyncio.sleep(self.retry_policy.delay(attempt))

        raise ValueError("Judge schema correction retries exhausted.")
