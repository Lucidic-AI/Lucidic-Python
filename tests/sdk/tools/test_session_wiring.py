"""Integration test for the SessionResource auto-init-fixtures wiring.

Bridges LUC-608's two halves: (1) when `client.sessions.create()` is
called with `datasetitem_id`, the SessionResource auto-fires
`/sdk/session-init-fixtures`; (2) on success, ToolsResource binds a
MockContext so subsequent `@mockable` calls in the same context route
through the backend.

Real LucidicAI construction is heavy (telemetry init, API key verify,
etc.) so we mock at the HTTP boundary via respx and use
``production=True`` to suppress the API-key validation that would fail
without a real backend.
"""
import os

import httpx
import pytest
import respx

import lucidicai
from lucidicai import LucidicAI
from lucidicai.sdk.tools.context import _current_mock_context, current_mock_context
from lucidicai.sdk.tools.registry import _PENDING_BUFFER, _REGISTRY


_INITSESSION_ENDPOINT = "https://stub.lucidic.test/initsession"
_INIT_FIXTURES_ENDPOINT = "https://stub.lucidic.test/sdk/session-init-fixtures"
_SYNC_ENDPOINT = "https://stub.lucidic.test/sdk/agent-tools/sync"
_MOCK_CALL_ENDPOINT = "https://stub.lucidic.test/sdk/mock-call"


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Force LUCIDIC_DEBUG off + neutralize .env interference."""
    monkeypatch.setenv("LUCIDIC_DEBUG", "false")
    monkeypatch.setenv("LUCIDIC_BASE_URL", "https://stub.lucidic.test")
    monkeypatch.delenv("LUCIDIC_REGION", raising=False)
    yield


@pytest.fixture(autouse=True)
def _clear_state():
    _REGISTRY.clear()
    _PENDING_BUFFER.clear()
    current_mock_context.set(None)
    yield
    _REGISTRY.clear()
    _PENDING_BUFFER.clear()
    current_mock_context.set(None)


@pytest.fixture
def client():
    """A real LucidicAI with production=True so API-key validation
    short-circuits. The HTTP client points at the stub URL via the
    env var fixture above."""
    return LucidicAI(
        api_key="test-key",
        agent_id="aaaa1111-0000-0000-0000-000000000000",
        production=True,
    )


class TestAutoInit:
    @respx.mock
    def test_no_datasetitem_skips_init(self, client):
        # Session created without datasetitem_id → no init-fixtures call
        respx.post(_INITSESSION_ENDPOINT).mock(
            return_value=httpx.Response(
                201, json={"session_id": "sess-x", "session_name": "test"},
            )
        )
        init_route = respx.post(_INIT_FIXTURES_ENDPOINT)

        client.sessions.create(session_name="test")

        assert init_route.call_count == 0
        assert _current_mock_context() is None

    @respx.mock
    def test_with_datasetitem_fires_init(self, client):
        respx.post(_INITSESSION_ENDPOINT).mock(
            return_value=httpx.Response(
                201, json={"session_id": "sess-x", "session_name": "test"},
            )
        )
        respx.post(_INIT_FIXTURES_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"initialized": True, "fixture_ids": [],
                      "tools_snapshotted": [], "already_initialized": False},
            )
        )

        client.sessions.create(
            session_name="test",
            datasetitem_id="ditm-1",
        )
        # MockContext is bound for the current context
        ctx = _current_mock_context()
        assert ctx is not None
        assert ctx.session_id == "sess-x"
        assert ctx.client is client

    @respx.mock
    def test_with_datasetitem_but_backend_not_tool_backed(self, client):
        # Backend says "this session has no Resources/Tools" — SDK skips
        # MockContext binding and the session proceeds as a normal one.
        respx.post(_INITSESSION_ENDPOINT).mock(
            return_value=httpx.Response(
                201, json={"session_id": "sess-x", "session_name": "test"},
            )
        )
        respx.post(_INIT_FIXTURES_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"initialized": False, "reason": "no Resources on dataset",
                      "fixture_ids": [], "tools_snapshotted": [],
                      "already_initialized": False},
            )
        )

        client.sessions.create(
            session_name="test",
            datasetitem_id="ditm-1",
        )
        assert _current_mock_context() is None

    @respx.mock
    def test_init_failure_does_not_block_session_creation(self, client):
        respx.post(_INITSESSION_ENDPOINT).mock(
            return_value=httpx.Response(
                201, json={"session_id": "sess-x", "session_name": "test"},
            )
        )
        respx.post(_INIT_FIXTURES_ENDPOINT).mock(
            return_value=httpx.Response(500, json={"error": "internal"}),
        )

        # Must not raise — session creation continues even if init fails
        session = client.sessions.create(
            session_name="test",
            datasetitem_id="ditm-1",
        )
        assert session is not None
        assert _current_mock_context() is None


class TestEndToEnd:
    """Decorate + create tool-backed session + invoke. The full chain."""

    @respx.mock
    def test_full_chain(self, client):
        # Decorate before session start
        @lucidicai.mockable
        def query_emails(sender: str) -> list:
            return ["local result"]

        # Wire the three backend endpoints
        respx.post(_INITSESSION_ENDPOINT).mock(
            return_value=httpx.Response(
                201, json={"session_id": "sess-1", "session_name": "test"},
            )
        )
        respx.post(_INIT_FIXTURES_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"initialized": True, "fixture_ids": [],
                      "tools_snapshotted": ["query_emails"],
                      "already_initialized": False},
            )
        )
        respx.post(_SYNC_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"synced": True, "stats": {}}),
        )
        mock_call_route = respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"return_value": ["mocked email"],
                      "tier": "PYTHON", "was_mocked": True},
            )
        )

        # Start session with datasetitem_id (the tool-backed signal)
        client.sessions.create(
            session_name="test", datasetitem_id="ditm-1",
        )

        # Invoke the decorated tool — should hit the backend
        result = query_emails("alice@example.com")
        assert result == ["mocked email"]
        assert mock_call_route.call_count == 1

        # Auto-sync should have fired once (registry had query_emails when
        # the tool-backed session started)
        assert len(respx.calls) >= 3  # initsession + init-fixtures + sync + mock-call
