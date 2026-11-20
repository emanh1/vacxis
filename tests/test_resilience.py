"""Offline retry and checkpoint behavior, without retrieval models."""
import asyncio
from unittest.mock import Mock
import httpx
import pytest
from openai import APIConnectionError, PermissionDeniedError
from vacxis_capstone_fa26.resilience import RetryPolicy, atomic_json


def test_transient_retry_and_exhaustion(monkeypatch):
    waits = []
    monkeypatch.setattr('vacxis_capstone_fa26.resilience.time.sleep', waits.append)
    error = APIConnectionError(request=httpx.Request('POST', 'https://example.com'))
    call = Mock(side_effect=[error, error, 'ok'])
    assert RetryPolicy(2, 1, 2).run(call) == 'ok'
    assert len(waits) == 2
    assert 0.8 <= waits[0] <= 1
    assert 1.6 <= waits[1] <= 2
    with pytest.raises(APIConnectionError):
        RetryPolicy(0).run(Mock(side_effect=error))


def test_permanent_error_not_retried():
    response = httpx.Response(403, request=httpx.Request('POST', 'https://example.com'))
    call = Mock(side_effect=PermissionDeniedError('blocked', response=response, body=None))
    with pytest.raises(PermissionDeniedError):
        RetryPolicy().run(call)
    assert call.call_count == 1


def test_async_retry(monkeypatch):
    from unittest.mock import AsyncMock
    sleep = AsyncMock()
    monkeypatch.setattr('vacxis_capstone_fa26.resilience.asyncio.sleep', sleep)
    error = APIConnectionError(request=httpx.Request('POST', 'https://example.com'))
    call = AsyncMock(side_effect=[error, 'ok'])
    assert asyncio.run(RetryPolicy(1).arun(call)) == 'ok'
    sleep.assert_awaited_once()


def test_atomic_checkpoint(tmp_path):
    target = tmp_path / 'nested' / 'report.json'
    atomic_json(target, {'complete': False})
    atomic_json(target, {'complete': True})
    assert 'true' in target.read_text()
    assert not target.with_suffix('.json.tmp').exists()


@pytest.fixture
def paced_clock(monkeypatch):
    from vacxis_capstone_fa26 import resilience
    clock = [0.0]
    monkeypatch.setattr(resilience, '_request_pacer', resilience._RequestPacer())
    monkeypatch.setattr(resilience.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(resilience.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    return clock


def test_spacing_shared_across_policies_and_retries(paced_clock):
    error = APIConnectionError(request=httpx.Request('POST', 'https://example.com'))
    starts = []
    def call():
        starts.append(paced_clock[0])
        if len(starts) == 2:
            raise error
        return 'ok'
    assert RetryPolicy(call_delay=3).run(call) == 'ok'
    assert RetryPolicy(1, 0, 0, call_delay=3).run(call) == 'ok'
    assert starts == [0, 3, 6]


def test_async_concurrent_requests_are_spaced(paced_clock, monkeypatch):
    from vacxis_capstone_fa26 import resilience
    original_sleep = asyncio.sleep
    async def sleep(seconds):
        paced_clock[0] += seconds
        await original_sleep(0)
    monkeypatch.setattr(resilience.asyncio, 'sleep', sleep)
    starts = []
    async def call():
        starts.append(paced_clock[0])
        return 'ok'
    async def run():
        return await asyncio.gather(*(RetryPolicy(call_delay=2).arun(call) for _ in range(3)))
    assert asyncio.run(run()) == ['ok'] * 3
    assert len(starts) == 3
    assert all(b - a >= 2 for a, b in zip(starts, starts[1:]))


def test_sync_and_async_share_pacing(paced_clock, monkeypatch):
    from vacxis_capstone_fa26 import resilience
    async def sleep(seconds):
        paced_clock[0] += seconds
    monkeypatch.setattr(resilience.asyncio, 'sleep', sleep)
    starts = []
    RetryPolicy(call_delay=2).run(lambda: starts.append(paced_clock[0]))
    async def call():
        starts.append(paced_clock[0])
    asyncio.run(RetryPolicy(call_delay=2).arun(call))
    assert starts == [0, 2]


@pytest.mark.parametrize('delay', [-1, float('nan'), float('inf')])
def test_invalid_call_delay(delay):
    with pytest.raises(ValueError):
        RetryPolicy(call_delay=delay)
