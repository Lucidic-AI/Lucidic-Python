"""LUC-901 — 429/503 retry + backoff + Retry-After.

Retry sleeps are patched out (and recorded) so the tests assert the retry
count and computed delays without actually waiting.
"""
import httpx
import pytest
import respx

from lucidicai.core.errors import NotFoundError, ServiceUnavailableError


@pytest.fixture
def sleeps(monkeypatch):
    """Record + skip sync retry sleeps."""
    recorded = []
    monkeypatch.setattr("lucidicai.api.client.time.sleep", lambda s: recorded.append(s))
    return recorded


@pytest.fixture
def asleeps(monkeypatch):
    """Record + skip async retry sleeps."""
    recorded = []

    async def _fake(s):
        recorded.append(s)

    monkeypatch.setattr("lucidicai.api.client.asyncio.sleep", _fake)
    return recorded


_URL = "https://stub.lucidic.test/agents"


class TestRetry:
    @respx.mock
    def test_503_then_success_is_retried(self, http, sleeps):
        route = respx.get(_URL).mock(side_effect=[
            httpx.Response(503, json={"error": "degraded"}),
            httpx.Response(200, json={"ok": True}),
        ])
        assert http.get("agents") == {"ok": True}
        assert route.call_count == 2
        # First backoff = backoff_factor * 2**0 = 0.01.
        assert sleeps == [pytest.approx(0.01)]

    @respx.mock
    def test_exponential_backoff_across_two_retries(self, http, sleeps):
        route = respx.get(_URL).mock(side_effect=[
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, json={"ok": 1}),
        ])
        http.get("agents")
        assert route.call_count == 3
        assert sleeps == [pytest.approx(0.01), pytest.approx(0.02)]

    @respx.mock
    def test_retry_after_header_overrides_backoff(self, http, sleeps):
        respx.get(_URL).mock(side_effect=[
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(200, json={"ok": 1}),
        ])
        http.get("agents")
        assert sleeps == [pytest.approx(2.0)]

    @respx.mock
    def test_exhausts_retries_then_raises_typed(self, http, sleeps):
        # max_retries=3 -> 1 initial + 3 retries = 4 attempts, all 503.
        route = respx.get(_URL).mock(return_value=httpx.Response(503, json={"error": "down"}))
        with pytest.raises(ServiceUnavailableError):
            http.get("agents")
        assert route.call_count == 4
        assert len(sleeps) == 3

    @respx.mock
    def test_non_retryable_status_not_retried(self, http, sleeps):
        route = respx.get(_URL).mock(return_value=httpx.Response(404, json={"error": "x"}))
        with pytest.raises(NotFoundError):
            http.get("agents")
        assert route.call_count == 1
        assert sleeps == []

    @respx.mock
    def test_post_not_retried_may_have_committed(self, http, sleeps):
        # A POST may have committed a write before the 503 (e.g. the EvoSim
        # kickoff commits the run row, then 503s), so it must NOT be retried —
        # retrying would duplicate. Fails fast, like pre-C0.
        route = respx.post("https://stub.lucidic.test/sdk/thing").mock(
            return_value=httpx.Response(503, json={"error": "down"}))
        with pytest.raises(ServiceUnavailableError):
            http.post("sdk/thing", {"a": 1})
        assert route.call_count == 1
        assert sleeps == []

    @respx.mock
    def test_put_is_retried(self, http, sleeps):
        # PUT is idempotent -> safe to replay.
        route = respx.put("https://stub.lucidic.test/sdk/thing").mock(side_effect=[
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ])
        assert http.put("sdk/thing", {"a": 1})["ok"] is True
        assert route.call_count == 2

    @respx.mock
    def test_retry_after_is_capped(self, http, sleeps):
        # A huge / proxy-injected Retry-After is clamped so it can't pin the
        # caller thread for minutes (_MAX_RETRY_DELAY_SECONDS = 30).
        respx.get(_URL).mock(side_effect=[
            httpx.Response(503, headers={"Retry-After": "3600"}),
            httpx.Response(200, json={"ok": 1}),
        ])
        http.get("agents")
        assert sleeps == [pytest.approx(30.0)]


class TestAsyncRetry:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_503_then_success(self, http, asleeps):
        route = respx.get(_URL).mock(side_effect=[
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ])
        assert await http.aget("agents") == {"ok": True}
        assert route.call_count == 2
        assert asleeps == [pytest.approx(0.01)]

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_exhausts_and_raises(self, http, asleeps):
        route = respx.get(_URL).mock(return_value=httpx.Response(503))
        with pytest.raises(ServiceUnavailableError):
            await http.aget("agents")
        assert route.call_count == 4
