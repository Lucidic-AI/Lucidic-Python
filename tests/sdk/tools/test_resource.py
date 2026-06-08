"""Tests for ``lucidicai.sdk.tools.resource.ToolsResource``.

Covers the user-facing surface (mockable, register, snapshot, sync),
the lifecycle hooks (_init_session, _maybe_sync_on_session_start) and
the debounce semantics that keep auto-sync from spamming the backend.

Network calls to ``/sdk/agent-tools/sync`` and
``/sdk/session-init-fixtures`` are intercepted via respx.
"""
import httpx
import pytest
import respx

from lucidicai.api.client import HttpClient
from lucidicai.core.config import NetworkConfig, SDKConfig
from lucidicai.sdk.tools.context import _current_mock_context, current_mock_context
from lucidicai.sdk.tools.registry import (
    ToolSurface,
    _PENDING_BUFFER,
    _REGISTRY,
)
from lucidicai.sdk.tools.resource import ToolsResource


_BASE_URL = "https://stub.lucidic.test"
_SYNC_ENDPOINT = f"{_BASE_URL}/sdk/agent-tools/sync"
_INIT_ENDPOINT = f"{_BASE_URL}/sdk/session-init-fixtures"


@pytest.fixture(autouse=True)
def _clear_module_state():
    _REGISTRY.clear()
    _PENDING_BUFFER.clear()
    current_mock_context.set(None)
    yield
    _REGISTRY.clear()
    _PENDING_BUFFER.clear()
    current_mock_context.set(None)


@pytest.fixture
def fake_client():
    """Minimal LucidicAI stand-in carrying just the fields ToolsResource reads."""
    class _FakeClient:
        def __init__(self):
            network = NetworkConfig(base_url=_BASE_URL)
            self._config = SDKConfig(
                api_key="test-key",
                agent_id="aaaa1111-0000-0000-0000-000000000000",
                network=network,
            )
            self._http = HttpClient(config=self._config)
            self._resources = {}
    return _FakeClient()


@pytest.fixture
def tools(fake_client) -> ToolsResource:
    return ToolsResource(client=fake_client)


# ---------- Public surface ----------------------------------------------


class TestPublicSurface:
    def test_register_direct(self, tools):
        surface = ToolSurface(
            name="t", signature={"params": [], "return_type": None},
            docstring="", return_shape=None, source_hash="a" * 64,
        )
        tools.register(surface)
        assert "t" in tools._registry

    def test_snapshot_sorted(self, tools):
        for name in ("zeta", "alpha", "middle"):
            tools.register(ToolSurface(
                name=name, signature={"params": [], "return_type": None},
                docstring="", return_shape=None, source_hash=name * 10 + "x" * 24,
            ))
        names = [s.name for s in tools.snapshot()]
        assert names == ["alpha", "middle", "zeta"]

    def test_mockable_decorates_into_client_registry(self, tools):
        @tools.mockable
        def my_tool(x: int) -> int:
            return x * 2
        assert "my_tool" in tools._registry
        assert my_tool(3) == 6  # wrapper transparent without mock context

    def test_mockable_async(self, tools):
        import asyncio

        @tools.mockable
        async def aquery(x: int) -> int:
            return x + 1

        assert "aquery" in tools._registry
        assert asyncio.run(aquery(4)) == 5


# ---------- sync() ------------------------------------------------------


class TestSync:
    @respx.mock
    def test_sync_posts_wire_payload(self, tools):
        route = respx.post(_SYNC_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"synced": True,
                                                    "stats": {"created": 1}}),
        )
        tools.register(ToolSurface(
            name="email_sender", signature={"params": [], "return_type": None},
            docstring="Send an email.", return_shape=None,
            source_hash="a" * 64,
        ))
        result = tools.sync()
        assert result["synced"] is True

        import json as _json
        body = _json.loads(route.calls.last.request.read())
        assert body["agent_id"] == "aaaa1111-0000-0000-0000-000000000000"
        assert len(body["tools"]) == 1
        assert body["tools"][0]["name"] == "email_sender"
        assert body["tools"][0]["docstring"] == "Send an email."
        assert body["tools"][0]["source_hash"] == "a" * 64

    @respx.mock
    def test_sync_empty_registry_noop(self, tools):
        # No network call should be made
        result = tools.sync()
        assert result["synced"] is True
        assert len(respx.calls) == 0

    @respx.mock
    def test_sync_updates_fingerprint(self, tools):
        respx.post(_SYNC_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"synced": True, "stats": {}}),
        )
        tools.register(ToolSurface(
            name="t", signature={"params": [], "return_type": None},
            docstring="", return_shape=None, source_hash="a" * 64,
        ))
        assert tools._last_synced_hash is None
        tools.sync()
        assert tools._last_synced_hash is not None
        fingerprint_after_sync = tools._last_synced_hash
        assert fingerprint_after_sync == tools._registry_fingerprint()


# ---------- _maybe_sync_on_session_start (debounce) ----------------------


class TestAutoSyncDebounce:
    @respx.mock
    def test_first_call_syncs(self, tools):
        route = respx.post(_SYNC_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"synced": True, "stats": {}}),
        )
        tools.register(ToolSurface(
            name="t", signature={"params": [], "return_type": None},
            docstring="", return_shape=None, source_hash="a" * 64,
        ))
        tools._maybe_sync_on_session_start()
        assert route.call_count == 1

    @respx.mock
    def test_unchanged_registry_skips(self, tools):
        route = respx.post(_SYNC_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"synced": True, "stats": {}}),
        )
        tools.register(ToolSurface(
            name="t", signature={"params": [], "return_type": None},
            docstring="", return_shape=None, source_hash="a" * 64,
        ))
        tools._maybe_sync_on_session_start()
        tools._maybe_sync_on_session_start()  # registry unchanged
        tools._maybe_sync_on_session_start()  # still unchanged
        assert route.call_count == 1  # only the first one hit the network

    @respx.mock
    def test_registry_change_triggers_resync(self, tools):
        route = respx.post(_SYNC_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"synced": True, "stats": {}}),
        )
        tools.register(ToolSurface(
            name="t1", signature={"params": [], "return_type": None},
            docstring="", return_shape=None, source_hash="a" * 64,
        ))
        tools._maybe_sync_on_session_start()
        # Add a new tool
        tools.register(ToolSurface(
            name="t2", signature={"params": [], "return_type": None},
            docstring="", return_shape=None, source_hash="b" * 64,
        ))
        tools._maybe_sync_on_session_start()
        assert route.call_count == 2

    @respx.mock
    def test_sync_error_swallowed_in_auto_path(self, tools, caplog):
        import logging
        respx.post(_SYNC_ENDPOINT).mock(
            return_value=httpx.Response(503, json={"error": "lock contention"}),
        )
        tools.register(ToolSurface(
            name="t", signature={"params": [], "return_type": None},
            docstring="", return_shape=None, source_hash="a" * 64,
        ))
        # Should not raise
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            tools._maybe_sync_on_session_start()
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("auto-sync failed" in r.message for r in warnings)


# ---------- _init_session ------------------------------------------------


class TestInitSession:
    @respx.mock
    def test_initialized_true_binds_mock_context(self, tools):
        respx.post(_INIT_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"initialized": True,
                      "fixture_ids": ["fx-1"], "tools_snapshotted": ["t1"],
                      "already_initialized": False},
            )
        )
        respx.post(_SYNC_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"synced": True, "stats": {}}),
        )
        tools._init_session("sess-xyz")
        ctx = _current_mock_context()
        assert ctx is not None
        assert ctx.session_id == "sess-xyz"
        assert ctx.client is tools._client

    @respx.mock
    def test_initialized_false_no_binding(self, tools):
        respx.post(_INIT_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"initialized": False, "reason": "no DatasetItem",
                      "fixture_ids": [], "tools_snapshotted": [],
                      "already_initialized": False},
            )
        )
        tools._init_session("sess-no-tools")
        assert _current_mock_context() is None

    @respx.mock
    def test_init_failure_logs_and_continues(self, tools, caplog):
        import logging
        respx.post(_INIT_ENDPOINT).mock(
            return_value=httpx.Response(500, json={"error": "internal"}),
        )
        # Must not raise
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            tools._init_session("sess-1")
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("session-init-fixtures failed" in r.message for r in warnings)
        assert _current_mock_context() is None

    @respx.mock
    def test_init_triggers_auto_sync(self, tools):
        respx.post(_INIT_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"initialized": True, "fixture_ids": [],
                      "tools_snapshotted": ["t"], "already_initialized": False},
            )
        )
        sync_route = respx.post(_SYNC_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"synced": True, "stats": {}}),
        )
        # Register a tool so sync has something to push
        tools.register(ToolSurface(
            name="t", signature={"params": [], "return_type": None},
            docstring="", return_shape=None, source_hash="a" * 64,
        ))
        tools._init_session("sess-1")
        assert sync_route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_init_binds(self, tools):
        respx.post(_INIT_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"initialized": True, "fixture_ids": [],
                      "tools_snapshotted": [], "already_initialized": False},
            )
        )
        await tools._ainit_session("sess-async")
        ctx = _current_mock_context()
        assert ctx is not None
        assert ctx.session_id == "sess-async"
