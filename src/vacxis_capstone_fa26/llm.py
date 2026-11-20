"""Shared OpenRouter chat transport; configuration comes from the process environment."""
from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

BASE_URL = os.getenv("VACXIS_LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")


class LLMError(RuntimeError):
    """An error with credential-redacted diagnostics and optional response previews."""


def resolve_model(model: str | None = None, *, planner: bool = False) -> str:
    """Resolve an explicit override before the component's environment variable."""
    variable = "VACXIS_QUERY_PLANNER_MODEL" if planner else "VACXIS_GENERATION_MODEL"
    value = model if model is not None else os.getenv(variable, "")
    if not isinstance(value, str) or not value.strip():
        raise LLMError(f"Set {variable} or pass an explicit model.")
    return value.strip()


def validate_config(model: str | None = None, *, planner: bool = False) -> str:
    """Validate online settings without creating a network client."""
    resolved = resolve_model(model, planner=planner)
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        raise LLMError("Set OPENROUTER_API_KEY for online LLM calls.")
    return resolved


def _response_detail(response: Any, choice: Any = None) -> str:
    """Bound diagnostics; response text is opt-in and credentials are redacted."""
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    reasoning = getattr(message, "reasoning", None)
    refusal = getattr(message, "refusal", None)
    usage = getattr(response, "usage", None)
    details = {
        "model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "content_type": type(content).__name__,
        "content_chars": len(content) if isinstance(content, str) else None,
        "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else None,
        "has_refusal": bool(refusal),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
    try:
        preview_chars = max(0, min(4000, int(os.getenv("VACXIS_LLM_ERROR_PREVIEW_CHARS", "0"))))
    except ValueError:
        preview_chars = 0

    def redact(value):
        text = str(value)
        key = os.getenv("OPENROUTER_API_KEY", "")
        return text.replace(key, "[REDACTED]") if key else text

    if preview_chars:
        for name, value in (("content_preview", content), ("refusal_preview", refusal),
                            ("reasoning_preview", reasoning)):
            if isinstance(value, str):
                details[name] = redact(value)[:preview_chars]
    # Bound individual metadata values too, and redact before JSON escaping.
    details = {key: redact(value)[:200] if isinstance(value, str) and not key.endswith("_preview")
               else value for key, value in details.items()}
    return json.dumps(details, ensure_ascii=False)


def validate_response(response: Any) -> Any:
    """Reject invalid completions with actionable, bounded diagnostics."""
    try:
        choice = response.choices[0]
        message = choice.message
        content = message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMError("OpenRouter returned an invalid response: " + _response_detail(response)) from exc
    failures = []
    if choice.finish_reason not in (None, "stop"):
        failures.append("unexpected finish reason")
    if getattr(message, "refusal", None):
        failures.append("provider refusal")
    if not isinstance(content, str):
        failures.append("content is not text")
    elif not content.strip():
        failures.append("content is empty")
    if failures:
        raise LLMError("OpenRouter response rejected (" + ", ".join(failures) + "): " +
                       _response_detail(response, choice))
    return response


def completion(*, model: str, messages: list[dict[str, str]], timeout: float = 30.0,
               extra_body: dict[str, Any] | None = None, retry_policy=None, **kwargs: Any) -> Any:
    """Make one bounded SDK request and close the client on every exit path."""
    resolved = validate_config(model)
    try:
        with OpenAI(base_url=BASE_URL, api_key=os.getenv("OPENROUTER_API_KEY"),
                    timeout=timeout, max_retries=0) as client:
            from .resilience import RetryPolicy
            response = (retry_policy or RetryPolicy()).run(lambda: client.chat.completions.create(
                model=resolved, messages=messages, extra_body=extra_body or {}, **kwargs,
            ))
    except Exception as exc:
        raise LLMError(f"OpenRouter request failed ({safe_detail(exc)}).") from exc
    return validate_response(response)


def safe_detail(exc: BaseException, limit: int = 300) -> str:
    """One-line, credential-safe description of a transport failure.

    ``LLMError`` must never carry provider payloads or the API key, so the
    detail is limited to the exception class and HTTP status.
    """
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__}" + (f" HTTP {status}" if status is not None else "")
