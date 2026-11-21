"""Serve VacxiS generation with vLLM's Python LLM API on an NVIDIA GPU.

This small HTTP adapter exposes the non-streaming OpenAI Chat Completions shape
used by VacxiS while loading and running the model directly through ``vllm.LLM``.
"""
from __future__ import annotations

import argparse
import hmac
import os
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a vLLM model in this Python process and expose VacxiS's chat API. "
            "This launcher targets NVIDIA GPUs; TPU runtimes need a compatible "
            "vLLM TPU backend and separate launch configuration."
        ),
    )
    parser.add_argument("--model", default=os.getenv("MODEL_ID", "Qwen/Qwen3.8-27B"))
    parser.add_argument("--served-model-name", default=os.getenv("SERVED_MODEL_NAME"))
    parser.add_argument("--host", default=os.getenv("VLLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VLLM_PORT", "8000")))
    parser.add_argument("--max-model-len", type=int, default=int(os.getenv("MAX_MODEL_LEN", "16384")))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90")),
    )
    parser.add_argument("--default-max-tokens", type=int, default=16384)
    return parser.parse_args()


def create_app(llm: Any, *, model_name: str, api_key: str, default_max_tokens: int):
    """Build the small OpenAI-compatible API surface VacxiS needs."""
    from fastapi import FastAPI, Header, HTTPException
    from starlette.concurrency import run_in_threadpool
    from vllm import SamplingParams

    app = FastAPI(title="VacxiS vLLM generation endpoint")

    @app.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)):
        require_key(authorization, api_key, HTTPException)
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "vllm"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: dict[str, Any],
        authorization: str | None = Header(default=None),
    ):
        require_key(authorization, api_key, HTTPException)
        if request.get("model") not in (None, model_name):
            raise HTTPException(status_code=404, detail=f"Unknown model: {request.get('model')}")
        if request.get("stream"):
            raise HTTPException(status_code=400, detail="Streaming is not supported by this endpoint.")

        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list.")
        try:
            max_tokens = request.get("max_tokens", request.get("max_completion_tokens"))
            sampling = SamplingParams(
                temperature=float(request.get("temperature", 0.0)),
                max_tokens=int(max_tokens or default_max_tokens),
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid sampling parameters.") from exc

        outputs = await run_in_threadpool(
            llm.chat,
            [messages],
            sampling_params=sampling,
            use_tqdm=False,
        )
        result = outputs[0]
        completion = result.outputs[0]
        if completion.finish_reason == "length":
            raise HTTPException(
                status_code=500,
                detail="Generation reached its token limit; increase --default-max-tokens.",
            )

        prompt_tokens = len(result.prompt_token_ids or [])
        completion_tokens = len(completion.token_ids or [])
        return {
            "id": f"chatcmpl-vacxis-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": completion.text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def require_key(authorization: str | None, api_key: str, http_exception: Any) -> None:
    expected = f"Bearer {api_key}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise http_exception(status_code=401, detail="Invalid or missing API key.")


def main() -> None:
    args = parse_args()
    api_key = os.getenv("VLLM_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set VLLM_API_KEY before launching the server.")

    try:
        import uvicorn
        from vllm import LLM
    except ImportError as exc:
        raise SystemExit(
            "Install vLLM and its FastAPI/uvicorn dependencies for this NVIDIA GPU runtime first."
        ) from exc

    model_name = args.served_model_name or args.model
    llm = LLM(
        model=args.model,
        dtype="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        generation_config="vllm",
    )
    app = create_app(
        llm,
        model_name=model_name,
        api_key=api_key,
        default_max_tokens=args.default_max_tokens,
    )
    print(f"Serving {model_name} on {args.host}:{args.port}")
    print(f"Set VacxiS VACXIS_GENERATION_MODEL={model_name}")
    print("Set VacxiS VACXIS_GENERATION_BASE_URL=<reachable-server-address>/v1")
    print("Set VacxiS OPENROUTER_API_KEY to the same value as VLLM_API_KEY")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
