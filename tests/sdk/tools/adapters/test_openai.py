"""Tests for ``lucidicai.sdk.tools.adapters.openai``.

Covers surface extraction from OpenAI's nested ``{type: function,
function: {...}}`` shape, registration into client + module-level
buffer paths, and the user-side ``dispatch_*_tool_call`` helpers
(sync + async, with + without mock context, drift fallback,
missing-impl behavior, malformed argument strings).

Network is mocked at the HTTP boundary via respx — the adapter
exercises the real transport layer, not stubs, so the contract
between adapter and transport is also pinned here.
"""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
import respx

from lucidicai.sdk.tools.adapters.openai import (
    _compute_openai_source_hash,
    _extract_call,
    _surface_from_openai_function,
    adispatch_openai_tool_call,
    dispatch_openai_tool_call,
    register_openai_tools,
)
from lucidicai.sdk.tools.context import (
    MockContext,
    bind_mock_context,
)
from lucidicai.sdk.tools.registry import _PENDING_BUFFER, _REGISTRY


_MOCK_CALL_ENDPOINT = "https://stub.lucidic.test/sdk/mock-call"


# ---------- Surface extraction -------------------------------------------


def _spec(name="query_emails", **extra):
    s = {
        "name": name,
        "description": extra.pop("description", "Get emails"),
        "parameters": extra.pop("parameters", {
            "type": "object",
            "properties": {"sender": {"type": "string"}},
            "required": ["sender"],
        }),
    }
    s.update(extra)
    return s


class TestSurfaceExtraction:
    def test_basic_function_spec(self):
        surface = _surface_from_openai_function(_spec())
        assert surface.name == "query_emails"
        assert surface.docstring == "Get emails"
        assert surface.signature["return_type"] is None
        assert surface.signature["params"] == [
            {"name": "sender", "type": "string", "default": None, "required": True},
        ]
        assert len(surface.source_hash) == 64

    def test_missing_description_empty(self):
        spec = _spec()
        del spec["description"]
        surface = _surface_from_openai_function(spec)
        assert surface.docstring == ""

    def test_no_parameters_empty_params(self):
        surface = _surface_from_openai_function({
            "name": "no_args_tool", "description": "",
        })
        assert surface.signature["params"] == []

    def test_with_default_value(self):
        spec = _spec(parameters={
            "type": "object",
            "properties": {
                "sender": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["sender"],
        })
        params = _surface_from_openai_function(spec).signature["params"]
        assert params[1] == {
            "name": "limit", "type": "integer", "default": 50, "required": False,
        }

    def test_source_hash_stable_across_calls(self):
        spec = _spec()
        h1 = _compute_openai_source_hash(spec)
        h2 = _compute_openai_source_hash(spec)
        assert h1 == h2

    def test_source_hash_changes_on_spec_change(self):
        spec_a = _spec()
        spec_b = _spec(description="Different description")
        assert _compute_openai_source_hash(spec_a) != _compute_openai_source_hash(spec_b)

    def test_source_hash_independent_of_dict_order(self):
        # sort_keys=True guarantees order-insensitive hashing
        spec_a = {"name": "t", "description": "d", "parameters": {}}
        spec_b = {"description": "d", "parameters": {}, "name": "t"}
        assert _compute_openai_source_hash(spec_a) == _compute_openai_source_hash(spec_b)


# ---------- register_openai_tools ----------------------------------------


class TestRegisterOpenAITools:
    def test_registers_function_entries(self):
        TOOLS = [{"type": "function", "function": _spec()}]
        surfaces = register_openai_tools(TOOLS)
        assert len(surfaces) == 1
        assert surfaces[0].name == "query_emails"
        # Buffered into module-level registry (no client active in tests)
        assert "query_emails" in _REGISTRY

    def test_skips_non_function_entries(self):
        TOOLS = [
            {"type": "function", "function": _spec(name="real_tool")},
            {"type": "code_interpreter"},
            {"type": "file_search"},
            {"type": "web_search"},
        ]
        surfaces = register_openai_tools(TOOLS)
        assert [s.name for s in surfaces] == ["real_tool"]

    def test_idempotent_reregister_last_wins(self):
        spec_a = _spec(description="version 1")
        spec_b = _spec(description="version 2")
        register_openai_tools([{"type": "function", "function": spec_a}])
        register_openai_tools([{"type": "function", "function": spec_b}])
        # Same name → last-wins
        assert "query_emails" in _REGISTRY
        assert _REGISTRY["query_emails"].docstring == "version 2"

    def test_skips_malformed_entries(self):
        TOOLS = [
            "not a dict",
            {"type": "function"},  # no `function` key
            {"type": "function", "function": "not a dict"},
            {"type": "function", "function": _spec(name="good")},
        ]
        surfaces = register_openai_tools(TOOLS)
        assert [s.name for s in surfaces] == ["good"]

    def test_explicit_client_writes_directly(self):
        from types import SimpleNamespace
        fake_registry = {}
        fake_tools = SimpleNamespace(_registry=fake_registry)
        fake_client = SimpleNamespace(tools=fake_tools)
        TOOLS = [{"type": "function", "function": _spec()}]
        register_openai_tools(TOOLS, client=fake_client)
        # Goes to the explicit client's registry, NOT the module-level
        # buffer
        assert "query_emails" in fake_registry
        assert "query_emails" not in _REGISTRY


# ---------- _extract_call ------------------------------------------------


class TestExtractCall:
    def test_pydantic_style_attribute_access(self):
        call = SimpleNamespace(
            function=SimpleNamespace(
                name="t",
                arguments=json.dumps({"x": 1}),
            ),
        )
        name, kwargs = _extract_call(call)
        assert name == "t"
        assert kwargs == {"x": 1}

    def test_plain_dict_style(self):
        call = {"function": {"name": "t", "arguments": json.dumps({"x": 2})}}
        name, kwargs = _extract_call(call)
        assert name == "t"
        assert kwargs == {"x": 2}

    def test_empty_arguments_yields_empty_kwargs(self):
        call = SimpleNamespace(function=SimpleNamespace(name="t", arguments=""))
        name, kwargs = _extract_call(call)
        assert kwargs == {}

    def test_malformed_arguments_yields_empty(self, caplog):
        import logging
        call = SimpleNamespace(function=SimpleNamespace(
            name="t", arguments="not json",
        ))
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            name, kwargs = _extract_call(call)
        assert kwargs == {}
        assert any("un-parseable arguments" in r.message for r in caplog.records)

    def test_non_dict_parsed_arguments_yields_empty(self, caplog):
        import logging
        # arguments is a valid JSON value but not an object — e.g. a string
        call = SimpleNamespace(function=SimpleNamespace(
            name="t", arguments=json.dumps([1, 2, 3]),
        ))
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            name, kwargs = _extract_call(call)
        assert kwargs == {}

    def test_unsupported_call_type_raises(self):
        with pytest.raises(TypeError, match="expected ChatCompletionMessageToolCall"):
            _extract_call(42)


# ---------- dispatch_openai_tool_call ------------------------------------


def _make_call(name="query", **kwargs):
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=json.dumps(kwargs)),
    )


class TestDispatchNoContext:
    def test_runs_impl_directly(self):
        call = _make_call(name="hello", who="world")

        def hello(who):
            return f"hi {who}"

        result = dispatch_openai_tool_call(call, impls={"hello": hello})
        assert result == "hi world"

    def test_missing_impl_raises_keyerror(self):
        call = _make_call(name="ghost")
        with pytest.raises(KeyError, match="ghost"):
            dispatch_openai_tool_call(call, impls={})


class TestDispatchWithContext:
    @respx.mock
    def test_routes_through_backend(self, stub_client):
        bind_mock_context(MockContext(session_id="sess-1", client=stub_client))
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": "mocked-result", "tier": "PYTHON",
                           "was_mocked": True},
            )
        )
        call = _make_call(name="any_tool", x=5)

        def real_impl(x):
            return "would-have-been-real"

        result = dispatch_openai_tool_call(call, impls={"any_tool": real_impl})
        assert result == "mocked-result"

    @respx.mock
    def test_pass_through_falls_back_to_impl(self, stub_client):
        bind_mock_context(MockContext(session_id="s", client=stub_client))
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": None, "tier": "PASS_THROUGH",
                           "was_mocked": False},
            )
        )
        call = _make_call(name="t", x=10)

        def real_impl(x):
            return x * 100

        result = dispatch_openai_tool_call(call, impls={"t": real_impl})
        assert result == 1000  # local impl ran

    @respx.mock
    def test_drift_logs_warning_falls_back(self, stub_client, caplog):
        import logging
        bind_mock_context(MockContext(session_id="s", client=stub_client))
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                409, json={"error": {"code": "tool_drift",
                                     "detail": "...",
                                     "session_hash": "aaa12345",
                                     "current_hash": "bbb12345"}},
            )
        )
        call = _make_call(name="t", x=2)

        def real_impl(x):
            return x + 100

        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            result = dispatch_openai_tool_call(call, impls={"t": real_impl})
        assert result == 102
        assert any("drift" in r.message.lower() for r in caplog.records)

    @respx.mock
    def test_explicit_client_overrides_context(self, stub_client):
        # Context has a different (fake) client; explicit kwarg wins
        from types import SimpleNamespace as SN
        other_client = SN(_resources={"mock_calls": stub_client._resources["mock_calls"]})
        bind_mock_context(MockContext(session_id="s", client=other_client))
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": 42, "tier": "PYTHON",
                           "was_mocked": True},
            )
        )
        call = _make_call(name="t")
        result = dispatch_openai_tool_call(
            call, impls={"t": lambda: 0}, client=stub_client,
        )
        assert result == 42


# ---------- adispatch_openai_tool_call -----------------------------------


class TestAsyncDispatch:
    @pytest.mark.asyncio
    async def test_runs_async_impl_no_context(self):
        call = _make_call(name="t", x=4)

        async def aimpl(x):
            return x * 2

        result = await adispatch_openai_tool_call(call, impls={"t": aimpl})
        assert result == 8

    @respx.mock
    @pytest.mark.asyncio
    async def test_routes_with_context(self, stub_client):
        bind_mock_context(MockContext(session_id="s", client=stub_client))
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": "from-backend",
                           "tier": "PYTHON", "was_mocked": True},
            )
        )
        call = _make_call(name="t")

        async def aimpl():
            return "from-local"

        result = await adispatch_openai_tool_call(call, impls={"t": aimpl})
        assert result == "from-backend"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_pass_through(self, stub_client):
        bind_mock_context(MockContext(session_id="s", client=stub_client))
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": None, "tier": "PASS_THROUGH",
                           "was_mocked": False},
            )
        )
        call = _make_call(name="t", x=3)

        async def aimpl(x):
            return x * 5

        result = await adispatch_openai_tool_call(call, impls={"t": aimpl})
        assert result == 15


# ---------- client.tools.register_openai / dispatch_openai ---------------


class TestInstanceBoundSurface:
    """The canonical ``client.tools.register_openai`` / ``dispatch_openai``
    methods delegate to the module-level adapter functions. Use a real
    LucidicAI (production=True to skip API-key verify) so ``client.tools``
    resolves via the actual property."""

    def test_register_via_client(self, monkeypatch):
        monkeypatch.setenv("LUCIDIC_DEBUG", "false")
        monkeypatch.setenv("LUCIDIC_BASE_URL", "https://stub.lucidic.test")
        from lucidicai import LucidicAI

        client = LucidicAI(
            api_key="test-key",
            agent_id="00000000-0000-0000-0000-000000000000",
            production=True,
        )
        surfaces = client.tools.register_openai([
            {"type": "function", "function": _spec(name="instance_tool")},
        ])
        assert [s.name for s in surfaces] == ["instance_tool"]
        assert "instance_tool" in client.tools._registry
