"""Tests for ``lucidicai.api.resources.mock_call``.

Covers both the new thin ``call()`` / ``acall()`` transport (used by
``sdk/tools/transport.py``) and the legacy ``create()`` / ``acreate()``
API kept for LUC-483 era explicit-mock customers. Backend is mocked via
``respx`` at the HTTP boundary so we exercise the real request shape,
real ``_handle_response`` parsing, and the real exception-translation
path.
"""
import httpx
import pytest
import respx

from lucidicai.api.client import HttpClient
from lucidicai.api.resources.mock_call import MockCallResource
from lucidicai.core.config import NetworkConfig, SDKConfig
from lucidicai.core.errors import (
    LucidicMockCallError,
    LucidicToolBlockedError,
    LucidicToolDriftError,
    LucidicUnknownToolError,
    LucidicUnsupportedSQLError,
)


_BASE_URL = "https://stub.lucidic.test"
_ENDPOINT = f"{_BASE_URL}/sdk/mock-call"


@pytest.fixture
def resource() -> MockCallResource:
    # Construct configs directly rather than via from_env — the latter
    # honors LUCIDIC_DEBUG and would silently redirect to localhost.
    network = NetworkConfig(base_url=_BASE_URL)
    config = SDKConfig(
        api_key="test-key",
        agent_id="00000000-0000-0000-0000-000000000000",
        network=network,
    )
    http = HttpClient(config=config)
    return MockCallResource(http=http, production=False)


# ---------- call() / acall() — thin transport ----------------------------


class TestCallSuccess:
    @respx.mock
    def test_returns_full_v2_body(self, resource):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"return_value": [1, 2, 3], "tier": "SQL_TEMPLATE",
                      "was_mocked": True},
            )
        )
        body = resource.call(
            session_id="sess-1", tool_name="query", kwargs={"sql": "SELECT 1"},
        )
        assert body == {"return_value": [1, 2, 3], "tier": "SQL_TEMPLATE",
                        "was_mocked": True}

    @respx.mock
    def test_pass_through_returned_intact(self, resource):
        # The thin call() doesn't unwrap PASS_THROUGH; that's the
        # transport layer's policy. Here it just returns the body.
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"return_value": None, "tier": "PASS_THROUGH",
                      "was_mocked": False},
            )
        )
        body = resource.call(session_id="s", tool_name="t", kwargs={})
        assert body["was_mocked"] is False

    @respx.mock
    def test_request_shape(self, resource):
        route = respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"return_value": "ok", "tier": "PYTHON", "was_mocked": True},
            )
        )
        resource.call(
            session_id="sess-xyz",
            tool_name="my_tool",
            kwargs={"a": 1, "b": [2, 3]},
            client_event_id="evt-42",
        )
        # respx records the actual request body
        sent = route.calls.last.request.read()
        import json as _json
        body = _json.loads(sent)
        assert body["session_id"] == "sess-xyz"
        assert body["tool_name"] == "my_tool"
        assert body["kwargs"] == {"a": 1, "b": [2, 3]}
        assert body["client_event_id"] == "evt-42"
        # _add_timestamp injects current_time; just assert it's present
        assert "current_time" in body

    @respx.mock
    def test_client_event_id_omitted_when_none(self, resource):
        route = respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": 1, "tier": "PYTHON", "was_mocked": True},
            )
        )
        resource.call(session_id="s", tool_name="t", kwargs={})
        import json as _json
        body = _json.loads(route.calls.last.request.read())
        assert "client_event_id" not in body


class TestCallErrors:
    """All non-2xx envelopes translate to a typed exception subclass."""

    @respx.mock
    def test_404_unknown_tool(self, resource):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                404,
                json={"error": {"code": "unknown_tool",
                                "detail": "no tool 'foo' on agent X"}},
            )
        )
        with pytest.raises(LucidicUnknownToolError) as exc_info:
            resource.call(session_id="s", tool_name="foo", kwargs={})
        assert exc_info.value.code == "unknown_tool"
        assert "no tool 'foo'" in exc_info.value.detail

    @respx.mock
    def test_409_tool_drift_carries_hashes(self, resource):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                409,
                json={"error": {"code": "tool_drift",
                                "detail": "source_hash drifted",
                                "session_hash": "aaa", "current_hash": "bbb"}},
            )
        )
        with pytest.raises(LucidicToolDriftError) as exc_info:
            resource.call(session_id="s", tool_name="t", kwargs={})
        assert exc_info.value.session_hash == "aaa"
        assert exc_info.value.current_hash == "bbb"

    @respx.mock
    def test_422_unsupported_sql_carries_dialect(self, resource):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                422,
                json={"error": {"code": "unsupported_sql",
                                "detail": "parse failed",
                                "source_dialect": "POSTGRES"}},
            )
        )
        with pytest.raises(LucidicUnsupportedSQLError) as exc_info:
            resource.call(session_id="s", tool_name="t", kwargs={})
        assert exc_info.value.source_dialect == "POSTGRES"
        assert exc_info.value.detail == "parse failed"

    @respx.mock
    def test_422_tool_blocked(self, resource):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                422,
                json={"error": {"code": "tool_blocked", "detail": "BLOCKED"}},
            )
        )
        with pytest.raises(LucidicToolBlockedError):
            resource.call(session_id="s", tool_name="t", kwargs={})

    @respx.mock
    def test_unknown_code_falls_back_to_base(self, resource):
        # Forward-compat: a backend that adds a new error code should
        # still produce a typed mock_call error, just the base class.
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                422,
                json={"error": {"code": "future_error_code",
                                "detail": "something new"}},
            )
        )
        with pytest.raises(LucidicMockCallError) as exc_info:
            resource.call(session_id="s", tool_name="t", kwargs={})
        # Specifically the base, not a typed subclass
        assert type(exc_info.value) is LucidicMockCallError
        assert exc_info.value.code == "future_error_code"

    @respx.mock
    def test_malformed_envelope_becomes_typed_error(self, resource):
        # Backend / proxy returned non-JSON or wrong shape — still surface
        # as a mock_call error so callers can `except LucidicMockCallError`
        # uniformly.
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(500, text="<html>500</html>")
        )
        with pytest.raises(LucidicMockCallError) as exc_info:
            resource.call(session_id="s", tool_name="t", kwargs={})
        assert exc_info.value.code == "malformed_response"


# ---------- acall (async) ------------------------------------------------


class TestAsyncCall:
    @respx.mock
    @pytest.mark.asyncio
    async def test_acall_success(self, resource):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"return_value": {"ok": True}, "tier": "PYTHON",
                      "was_mocked": True},
            )
        )
        body = await resource.acall(
            session_id="s", tool_name="t", kwargs={},
        )
        assert body["return_value"] == {"ok": True}

    @respx.mock
    @pytest.mark.asyncio
    async def test_acall_translates_error(self, resource):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                404,
                json={"error": {"code": "unknown_tool", "detail": "..."}}
            )
        )
        with pytest.raises(LucidicUnknownToolError):
            await resource.acall(session_id="s", tool_name="t", kwargs={})


# ---------- Legacy create() / acreate() ----------------------------------


class TestLegacyCreate:
    @respx.mock
    def test_returns_unwrapped_return_value(self, resource):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"return_value": {"columns": ["a"], "rows": [[1]]},
                      "tier": "SQL_TEMPLATE", "was_mocked": True},
            )
        )
        result = resource.create("query", session_id="s", sql="SELECT 1")
        assert result == {"columns": ["a"], "rows": [[1]]}

    @respx.mock
    def test_pass_through_returns_none_with_warning(self, resource, caplog):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"return_value": None, "tier": "PASS_THROUGH",
                      "was_mocked": False},
            )
        )
        import logging
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            result = resource.create("query", session_id="s", sql="SELECT 1")
        assert result is None
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("was not mocked" in r.message for r in warnings)
        assert any("dashboard" in r.message.lower() for r in warnings)

    @respx.mock
    def test_no_active_session_returns_none(self, resource, caplog):
        # No session_id passed and no contextvar bound — short-circuit
        # without hitting the backend.
        import logging
        with caplog.at_level(logging.DEBUG, logger="Lucidic"):
            result = resource.create("query", sql="SELECT 1")
        assert result is None
        # Sanity: no network call was attempted
        assert len(respx.calls) == 0

    @respx.mock
    def test_legacy_propagates_typed_errors(self, resource):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                422,
                json={"error": {"code": "unsupported_sql", "detail": "bad",
                                "source_dialect": "MYSQL"}},
            )
        )
        with pytest.raises(LucidicUnsupportedSQLError) as exc_info:
            resource.create("query", session_id="s", sql="MALFORMED")
        assert exc_info.value.source_dialect == "MYSQL"

    @respx.mock
    @pytest.mark.asyncio
    async def test_acreate_unwraps_return_value(self, resource):
        respx.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"return_value": [42], "tier": "PYTHON", "was_mocked": True},
            )
        )
        result = await resource.acreate("query", session_id="s")
        assert result == [42]
