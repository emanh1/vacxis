"""Bounded retry policy and atomic, fingerprinted checkpoints."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from threading import Lock

class _RequestPacer:
    """Space request starts across sync and async callers in this process."""

    def __init__(self):
        self._lock = Lock()
        self._last_start = None

    def _remaining(self, interval):
        with self._lock:
            now = time.monotonic()
            remaining = 0 if self._last_start is None else interval - (now - self._last_start)
            if remaining <= 0:
                self._last_start = now
                return 0
            return remaining

    def wait(self, interval):
        if interval == 0:
            return
        while (remaining := self._remaining(interval)) > 0:
            time.sleep(remaining)

    async def await_turn(self, interval):
        if interval == 0:
            return
        while (remaining := self._remaining(interval)) > 0:
            await asyncio.sleep(remaining)


_request_pacer = _RequestPacer()


@dataclass(frozen=True)
class RetryPolicy:
    retries: int = 5
    initial: float = 1.0
    maximum: float = 60.0
    call_delay: float = 0.0

    def __post_init__(self):
        if self.retries < 0 or any(not math.isfinite(x) or x < 0 for x in (self.initial, self.maximum, self.call_delay)):
            raise ValueError("Retry counts and delays must be finite and nonnegative.")

    def delay(self, attempt):
        return min(self.maximum, self.initial * 2 ** min(attempt, 30)) * random.uniform(0.8, 1.0)

    @staticmethod
    def transient(exc):
        from openai import APIConnectionError, APIStatusError
        return isinstance(exc, APIConnectionError) or (
            isinstance(exc, APIStatusError) and (exc.status_code in (408, 409, 429) or exc.status_code >= 500)
        )

    def run(self, call):
        for attempt in range(self.retries + 1):
            _request_pacer.wait(self.call_delay)
            try:
                return call()
            except Exception as exc:
                if attempt == self.retries or not self.transient(exc):
                    raise
                time.sleep(self.delay(attempt))

    async def arun(self, call):
        for attempt in range(self.retries + 1):
            await _request_pacer.await_turn(self.call_delay)
            try:
                return await call()
            except Exception as exc:
                if attempt == self.retries or not self.transient(exc):
                    raise
                await asyncio.sleep(self.delay(attempt))

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def add_retry_arguments(parser):
    parser.add_argument("--llm-call-delay", type=float, default=0.0,
                        help="Minimum seconds between LLM request starts in this process (including retries)")
    parser.add_argument("--retries", type=int, default=5, help="Additional transient API attempts")
    parser.add_argument("--retry-initial", type=float, default=1.0, help="Initial exponential backoff in seconds")
    parser.add_argument("--retry-max", type=float, default=60.0, help="Maximum backoff in seconds")

def retry_from_args(args):
    return RetryPolicy(args.retries, args.retry_initial, args.retry_max, args.llm_call_delay)
