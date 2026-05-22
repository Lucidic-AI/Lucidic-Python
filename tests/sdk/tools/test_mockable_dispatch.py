"""Tests for the live-dispatch wiring of ``@mockable``.

LUC-577 covered surface capture + the fast path (no mock context).
This file pins the behavior with mock context bound: every @mockable
call should route through ``emit_call_through_backend`` and hit the
mocked backend endpoint.
"""
import asyncio

import httpx
import pytest
import respx

from lucidicai.sdk.tools.context import (
    MockContext,
    bind_mock_context,
    current_mock_context,
)
from lucidicai.sdk.tools.mockable import mockable
from lucidicai.sdk.tools.registry import _PENDING_BUFFER, _REGISTRY


_MOCK_CALL_ENDPOINT = "https://stub.lucidic.test/sdk/mock-call"


@pytest.fixture(autouse=True)
def _clear_state():
    _REGISTRY.clear()
    _PENDING_BUFFER.clear()
    current_mock_context.set(None)
    yield
    _REGISTRY.clear()
    _PENDING_BUFFER.clear()
    current_mock_context.set(None)


class TestSyncDispatch:
    @respx.mock
    def test_no_mock_context_runs_local(self, stub_client):
        @mockable
        def add(a: int, b: int) -> int:
            return a + b

        # No context → wrapper runs func, no network call
        assert add(2, 3) == 5
        assert len(respx.calls) == 0

    @respx.mock
    def test_with_mock_context_routes_to_backend(self, stub_client):
        route = respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": 999, "tier": "SQL_TEMPLATE",
                           "was_mocked": True},
            )
        )

        @mockable
        def add(a: int, b: int) -> int:
            return a + b  # Real impl returns 5; backend mocks 999

        bind_mock_context(MockContext(session_id="sess-1", client=stub_client))
        result = add(2, 3)
        assert result == 999  # backend's mocked return_value
        assert route.call_count == 1

    @respx.mock
    def test_pass_through_falls_back_to_local(self, stub_client):
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": None, "tier": "PASS_THROUGH",
                           "was_mocked": False},
            )
        )

        @mockable
        def add(a: int, b: int) -> int:
            return a + b

        bind_mock_context(MockContext(session_id="s", client=stub_client))
        # PASS_THROUGH → transport runs real_fn (the decorated original)
        assert add(2, 3) == 5

    @respx.mock
    def test_drift_logs_warning_and_falls_back(self, stub_client, caplog):
        import logging
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                409,
                json={"error": {"code": "tool_drift",
                                "detail": "hash drifted",
                                "session_hash": "aaa1234567",
                                "current_hash": "bbb1234567"}},
            )
        )

        @mockable
        def my_tool(x: int) -> int:
            return x * 2

        bind_mock_context(MockContext(session_id="sess-1", client=stub_client))
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            result = my_tool(7)
        assert result == 14  # ran the real fn
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("drift" in r.message.lower() for r in warnings)


class TestAsyncDispatch:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_no_context_runs_local(self, stub_client):
        @mockable
        async def aadd(a: int, b: int) -> int:
            return a + b

        assert await aadd(2, 3) == 5
        assert len(respx.calls) == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_with_context_routes(self, stub_client):
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": "from-backend", "tier": "PYTHON",
                           "was_mocked": True},
            )
        )

        @mockable
        async def aquery(x: str) -> str:
            return "from-local"

        bind_mock_context(MockContext(session_id="s", client=stub_client))
        result = await aquery("anything")
        assert result == "from-backend"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_pass_through_runs_async_local(self, stub_client):
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": None, "tier": "PASS_THROUGH",
                           "was_mocked": False},
            )
        )

        @mockable
        async def aquery(x: int) -> int:
            return x * 3

        bind_mock_context(MockContext(session_id="s", client=stub_client))
        result = await aquery(4)
        assert result == 12


class TestClientEventIdUnique:
    """Each @mockable invocation generates its own client_event_id —
    backend uses this as the FUNCTION_CALL event's idempotency key,
    so distinct calls must produce distinct ids."""

    @respx.mock
    def test_distinct_per_call(self, stub_client):
        route = respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": 1, "tier": "PYTHON", "was_mocked": True},
            )
        )

        @mockable
        def t():
            return 0

        bind_mock_context(MockContext(session_id="s", client=stub_client))
        t()
        t()
        t()

        import json as _json
        event_ids = [
            _json.loads(call.request.read()).get("client_event_id")
            for call in route.calls
        ]
        assert len(event_ids) == 3
        assert all(eid is not None for eid in event_ids)
        assert len(set(event_ids)) == 3  # all distinct
