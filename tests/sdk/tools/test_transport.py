"""Tests for ``lucidicai.sdk.tools.transport``.

Exercises the policy layer: PASS_THROUGH falls back to local impl,
drift falls back with a WARNING, other errors propagate, network
errors get wrapped. Sync + async both.

The transport's input is a ``LucidicAI``-shaped object; we use the
``stub_client`` fixture from conftest.py which wraps a real
``MockCallResource`` over a real ``HttpClient``. respx intercepts at
the network boundary.
"""
import asyncio
import logging

import httpx
import pytest
import respx

from lucidicai.core.errors import (
    LucidicMissingImplError,
    LucidicMockCallError,
    LucidicUnknownToolError,
    LucidicUnsupportedSQLError,
)
from lucidicai.sdk.tools.transport import (
    aemit_call_through_backend,
    emit_call_through_backend,
)


_ENDPOINT = "https://stub.lucidic.test/sdk/mock-call"


# ---------- 200 + was_mocked=True (the happy path) -----------------------


class TestHappyPathSync:
    @respx.mock
    def test_returns_return_value(self, stub_client):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"return_value": {"columns": ["x"], "rows": [[1]]},
                      "tier": "SQL_TEMPLATE", "was_mocked": True},
            )
        )
        result = emit_call_through_backend(
            client=stub_client, session_id="sess-1",
            tool_name="query_sql", args=(), kwargs={"sql": "SELECT 1"},
            real_fn=None,
        )
        assert result == {"columns": ["x"], "rows": [[1]]}

    @respx.mock
    def test_positional_args_fold_to_underscore_args(self, stub_client):
        route = respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": None, "tier": "PYTHON",
                           "was_mocked": True},
            )
        )
        emit_call_through_backend(
            client=stub_client, session_id="s",
            tool_name="t", args=(1, "two"), kwargs={"x": 3},
            real_fn=None,
        )
        import json as _json
        body = _json.loads(route.calls.last.request.read())
        # Backend ignores __args__ in v1 but the field is reserved.
        assert body["kwargs"]["__args__"] == [1, "two"]
        assert body["kwargs"]["x"] == 3

    @respx.mock
    def test_real_fn_not_called_when_mocked(self, stub_client):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": "mocked", "tier": "PYTHON",
                           "was_mocked": True},
            )
        )
        called = []
        def real_fn(*a, **k):
            called.append((a, k))
            return "real"
        result = emit_call_through_backend(
            client=stub_client, session_id="s",
            tool_name="t", args=(), kwargs={}, real_fn=real_fn,
        )
        assert result == "mocked"
        assert called == []


class TestHappyPathAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_returns_return_value(self, stub_client):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": 42, "tier": "PYTHON",
                           "was_mocked": True},
            )
        )
        result = await aemit_call_through_backend(
            client=stub_client, session_id="s",
            tool_name="t", args=(), kwargs={}, real_fn=None,
        )
        assert result == 42


# ---------- PASS_THROUGH (200 + was_mocked=False) ------------------------


class TestPassThroughSync:
    @respx.mock
    def test_runs_real_fn_with_args(self, stub_client):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": None, "tier": "PASS_THROUGH",
                           "was_mocked": False},
            )
        )
        def real_fn(a, b, *, c=10):
            return a + b + c
        result = emit_call_through_backend(
            client=stub_client, session_id="s", tool_name="t",
            args=(1, 2), kwargs={"c": 100}, real_fn=real_fn,
        )
        assert result == 103

    @respx.mock
    def test_missing_real_fn_raises_missing_impl(self, stub_client):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": None, "tier": "PASS_THROUGH",
                           "was_mocked": False},
            )
        )
        with pytest.raises(LucidicMissingImplError) as exc_info:
            emit_call_through_backend(
                client=stub_client, session_id="s", tool_name="missing_one",
                args=(), kwargs={}, real_fn=None,
            )
        assert exc_info.value.tool_name == "missing_one"
        assert "tier=PASS_THROUGH" in exc_info.value.reason


class TestPassThroughAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_runs_async_real_fn(self, stub_client):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": None, "tier": "PASS_THROUGH",
                           "was_mocked": False},
            )
        )
        async def real_fn(x):
            return x * 2
        result = await aemit_call_through_backend(
            client=stub_client, session_id="s", tool_name="t",
            args=(), kwargs={"x": 21}, real_fn=real_fn,
        )
        assert result == 42

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_missing_real_fn_raises(self, stub_client):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": None, "tier": "PASS_THROUGH",
                           "was_mocked": False},
            )
        )
        with pytest.raises(LucidicMissingImplError):
            await aemit_call_through_backend(
                client=stub_client, session_id="s", tool_name="t",
                args=(), kwargs={}, real_fn=None,
            )


# ---------- 409 tool_drift (the load-bearing UX contract) -----------------


class TestDriftFallback:
    """Drift never blocks the user — log WARNING, run local fn, continue.
    These tests pin the contract loudly."""

    @respx.mock
    def test_drift_runs_real_fn(self, stub_client):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                409,
                json={"error": {"code": "tool_drift",
                                "detail": "source_hash drifted",
                                "session_hash": "aaa1234567890abc",
                                "current_hash": "bbb1234567890abc"}},
            )
        )
        def real_fn(x):
            return x + 1
        result = emit_call_through_backend(
            client=stub_client, session_id="s", tool_name="my_tool",
            args=(), kwargs={"x": 5}, real_fn=real_fn,
        )
        assert result == 6  # ran the real fn, didn't raise

    @respx.mock
    def test_drift_logs_warning_with_hashes(self, stub_client, caplog):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                409,
                json={"error": {"code": "tool_drift",
                                "detail": "...",
                                "session_hash": "aaa1234567890",
                                "current_hash": "bbb1234567890"}},
            )
        )
        def real_fn():
            return "ok"
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            emit_call_through_backend(
                client=stub_client, session_id="sess-1234567890",
                tool_name="my_tool", args=(), kwargs={}, real_fn=real_fn,
            )
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) >= 1
        msg = warnings[0].message
        # Both hashes (truncated) + dashboard hint
        assert "aaa1234567" in msg
        assert "bbb1234567" in msg
        assert "tool_drift" in msg.lower() or "drift" in msg.lower()
        assert "dashboard" in msg.lower()
        assert "my_tool" in msg

    @respx.mock
    def test_drift_without_real_fn_raises_missing_impl(self, stub_client):
        # If we can't fall back, surface the bug — better than silently
        # returning None.
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                409,
                json={"error": {"code": "tool_drift",
                                "detail": "drift",
                                "session_hash": "a", "current_hash": "b"}},
            )
        )
        with pytest.raises(LucidicMissingImplError) as exc_info:
            emit_call_through_backend(
                client=stub_client, session_id="s", tool_name="foo",
                args=(), kwargs={}, real_fn=None,
            )
        assert exc_info.value.tool_name == "foo"
        assert "tool_drift" in exc_info.value.reason

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_drift_runs_async_real_fn(self, stub_client):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                409,
                json={"error": {"code": "tool_drift", "detail": "...",
                                "session_hash": "a", "current_hash": "b"}},
            )
        )
        async def real_fn():
            return "from_local"
        result = await aemit_call_through_backend(
            client=stub_client, session_id="s", tool_name="t",
            args=(), kwargs={}, real_fn=real_fn,
        )
        assert result == "from_local"


# ---------- Other errors propagate ---------------------------------------


class TestOtherErrorsPropagate:
    @respx.mock
    def test_unknown_tool_raises(self, stub_client):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                404, json={"error": {"code": "unknown_tool", "detail": "?"}},
            )
        )
        with pytest.raises(LucidicUnknownToolError):
            emit_call_through_backend(
                client=stub_client, session_id="s", tool_name="ghost",
                args=(), kwargs={}, real_fn=lambda: "would_not_run",
            )

    @respx.mock
    def test_unsupported_sql_raises_with_dialect(self, stub_client):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                422,
                json={"error": {"code": "unsupported_sql",
                                "detail": "DELETE not allowed",
                                "source_dialect": "POSTGRES"}},
            )
        )
        with pytest.raises(LucidicUnsupportedSQLError) as exc_info:
            emit_call_through_backend(
                client=stub_client, session_id="s", tool_name="q",
                args=(), kwargs={}, real_fn=None,
            )
        assert exc_info.value.source_dialect == "POSTGRES"


# ---------- Network failures wrap as LucidicMockCallError ----------------


class TestNetworkFailure:
    @respx.mock
    def test_connect_error_wraps(self, stub_client):
        respx.post(_ENDPOINT).mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with pytest.raises(LucidicMockCallError) as exc_info:
            emit_call_through_backend(
                client=stub_client, session_id="s", tool_name="t",
                args=(), kwargs={}, real_fn=None,
            )
        assert exc_info.value.code == "network_error"
        assert "ConnectError" in exc_info.value.detail

    @respx.mock
    def test_timeout_wraps(self, stub_client):
        respx.post(_ENDPOINT).mock(
            side_effect=httpx.ReadTimeout("too slow")
        )
        with pytest.raises(LucidicMockCallError) as exc_info:
            emit_call_through_backend(
                client=stub_client, session_id="s", tool_name="t",
                args=(), kwargs={}, real_fn=None,
            )
        assert exc_info.value.code == "network_error"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_network_failure_wraps(self, stub_client):
        respx.post(_ENDPOINT).mock(
            side_effect=httpx.ConnectError("nope")
        )
        with pytest.raises(LucidicMockCallError) as exc_info:
            await aemit_call_through_backend(
                client=stub_client, session_id="s", tool_name="t",
                args=(), kwargs={}, real_fn=None,
            )
        assert exc_info.value.code == "network_error"
