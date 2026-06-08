"""Tests for ``lucidicai.sdk.tools.adapters.anthropic``.

Covers surface extraction from Anthropic's flat ``{name, description,
input_schema}`` shape, registration into client + module-level buffer
paths, and the user-side ``dispatch_*_tool_call`` helpers (sync + async,
with + without mock context, drift fallback, input-shape variants).

Symmetric to ``test_openai.py`` but pins the two structural differences:

- Flat tool spec (no nesting under ``function``)
- ``ToolUseBlock.input`` is already a dict (no JSON parsing needed)
"""
import json
from types import SimpleNamespace

import httpx
import pytest
import respx

from lucidicai.sdk.tools.adapters.anthropic import (
    _compute_anthropic_source_hash,
    _extract_block,
    _surface_from_anthropic_tool,
    adispatch_anthropic_tool_call,
    dispatch_anthropic_tool_call,
    register_anthropic_tools,
)
from lucidicai.sdk.tools.context import MockContext, bind_mock_context
from lucidicai.sdk.tools.registry import _REGISTRY


_MOCK_CALL_ENDPOINT = "https://stub.lucidic.test/sdk/mock-call"


def _spec(name="query_emails", **overrides):
    s = {
        "name": name,
        "description": overrides.pop("description", "Get emails"),
        "input_schema": overrides.pop("input_schema", {
            "type": "object",
            "properties": {"sender": {"type": "string"}},
            "required": ["sender"],
        }),
    }
    s.update(overrides)
    return s


# ---------- Surface extraction -------------------------------------------


class TestSurfaceExtraction:
    def test_basic_flat_spec(self):
        surface = _surface_from_anthropic_tool(_spec())
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
        assert _surface_from_anthropic_tool(spec).docstring == ""

    def test_no_input_schema_empty_params(self):
        spec = {"name": "no_args", "description": ""}
        assert _surface_from_anthropic_tool(spec).signature["params"] == []

    def test_with_default(self):
        spec = _spec(input_schema={
            "type": "object",
            "properties": {
                "sender": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["sender"],
        })
        params = _surface_from_anthropic_tool(spec).signature["params"]
        assert params[1] == {
            "name": "limit", "type": "integer", "default": 50, "required": False,
        }

    def test_source_hash_stable(self):
        spec = _spec()
        assert _compute_anthropic_source_hash(spec) == _compute_anthropic_source_hash(spec)

    def test_source_hash_changes_on_spec_change(self):
        a = _spec()
        b = _spec(description="different")
        assert _compute_anthropic_source_hash(a) != _compute_anthropic_source_hash(b)

    def test_source_hash_independent_of_key_order(self):
        a = {"name": "t", "description": "d", "input_schema": {}}
        b = {"input_schema": {}, "description": "d", "name": "t"}
        assert _compute_anthropic_source_hash(a) == _compute_anthropic_source_hash(b)


# ---------- register_anthropic_tools -------------------------------------


class TestRegister:
    def test_registers_flat_entries(self):
        surfaces = register_anthropic_tools([_spec()])
        assert [s.name for s in surfaces] == ["query_emails"]
        assert "query_emails" in _REGISTRY

    def test_skips_malformed_entries(self):
        TOOLS = [
            "not a dict",
            {"description": "no name"},
            _spec(name="good"),
        ]
        surfaces = register_anthropic_tools(TOOLS)
        assert [s.name for s in surfaces] == ["good"]

    def test_idempotent_last_wins(self):
        register_anthropic_tools([_spec(description="v1")])
        register_anthropic_tools([_spec(description="v2")])
        assert _REGISTRY["query_emails"].docstring == "v2"

    def test_explicit_client_writes_directly(self):
        fake_registry = {}
        fake_tools = SimpleNamespace(_registry=fake_registry)
        fake_client = SimpleNamespace(tools=fake_tools)
        register_anthropic_tools([_spec()], client=fake_client)
        assert "query_emails" in fake_registry
        assert "query_emails" not in _REGISTRY


# ---------- _extract_block -----------------------------------------------


class TestExtractBlock:
    def test_pydantic_style_attribute_access(self):
        block = SimpleNamespace(name="t", input={"x": 1})
        name, kwargs = _extract_block(block)
        assert name == "t"
        assert kwargs == {"x": 1}

    def test_plain_dict(self):
        block = {"name": "t", "input": {"x": 2}}
        name, kwargs = _extract_block(block)
        assert name == "t"
        assert kwargs == {"x": 2}

    def test_missing_input_yields_empty(self):
        block = SimpleNamespace(name="t", input=None)
        name, kwargs = _extract_block(block)
        assert kwargs == {}

    def test_no_input_attr_at_all(self):
        block = SimpleNamespace(name="t")
        name, kwargs = _extract_block(block)
        assert kwargs == {}

    def test_string_input_falls_back_to_json_parse(self, caplog):
        # Defensive code path: if Anthropic ever ships input as JSON string
        # (e.g. streaming partial), try to parse.
        block = SimpleNamespace(name="t", input=json.dumps({"x": 5}))
        name, kwargs = _extract_block(block)
        assert kwargs == {"x": 5}

    def test_unparseable_string_input_logs_and_empty(self, caplog):
        import logging
        block = SimpleNamespace(name="t", input="not json")
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            name, kwargs = _extract_block(block)
        assert kwargs == {}
        assert any("un-parseable input" in r.message for r in caplog.records)

    def test_non_dict_non_string_input_logs_and_empty(self, caplog):
        import logging
        block = SimpleNamespace(name="t", input=[1, 2, 3])
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            name, kwargs = _extract_block(block)
        assert kwargs == {}

    def test_missing_name_raises(self):
        with pytest.raises(TypeError, match="must have a 'name'"):
            _extract_block(SimpleNamespace(input={}))


# ---------- dispatch_anthropic_tool_call (sync) --------------------------


def _make_block(name="query", **input_kwargs):
    return SimpleNamespace(name=name, input=input_kwargs)


class TestDispatchNoContext:
    def test_runs_impl_directly(self):
        block = _make_block(name="hello", who="world")
        def hello(who):
            return f"hi {who}"
        assert dispatch_anthropic_tool_call(block, impls={"hello": hello}) == "hi world"

    def test_missing_impl_raises_keyerror(self):
        block = _make_block(name="ghost")
        with pytest.raises(KeyError, match="ghost"):
            dispatch_anthropic_tool_call(block, impls={})


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
        block = _make_block(name="t", x=5)
        result = dispatch_anthropic_tool_call(block, impls={"t": lambda x: x})
        assert result == "mocked-result"

    @respx.mock
    def test_pass_through_runs_local(self, stub_client):
        bind_mock_context(MockContext(session_id="s", client=stub_client))
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": None, "tier": "PASS_THROUGH",
                           "was_mocked": False},
            )
        )
        block = _make_block(name="t", x=10)
        def real(x):
            return x * 100
        assert dispatch_anthropic_tool_call(block, impls={"t": real}) == 1000

    @respx.mock
    def test_drift_logs_warning_falls_back(self, stub_client, caplog):
        import logging
        bind_mock_context(MockContext(session_id="s", client=stub_client))
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                409, json={"error": {"code": "tool_drift", "detail": "...",
                                     "session_hash": "aaa12345",
                                     "current_hash": "bbb12345"}},
            )
        )
        block = _make_block(name="t", x=2)
        def real(x):
            return x + 100
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            result = dispatch_anthropic_tool_call(block, impls={"t": real})
        assert result == 102
        assert any("drift" in r.message.lower() for r in caplog.records)


# ---------- adispatch_anthropic_tool_call (async) ------------------------


class TestAsyncDispatch:
    @pytest.mark.asyncio
    async def test_runs_async_impl_no_context(self):
        block = _make_block(name="t", x=4)
        async def aimpl(x):
            return x * 2
        assert await adispatch_anthropic_tool_call(block, impls={"t": aimpl}) == 8

    @respx.mock
    @pytest.mark.asyncio
    async def test_routes_with_context(self, stub_client):
        bind_mock_context(MockContext(session_id="s", client=stub_client))
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"return_value": "from-backend", "tier": "PYTHON",
                           "was_mocked": True},
            )
        )
        block = _make_block(name="t")
        async def aimpl():
            return "from-local"
        assert await adispatch_anthropic_tool_call(
            block, impls={"t": aimpl},
        ) == "from-backend"


# ---------- Instance-bound surface ---------------------------------------


class TestInstanceBoundSurface:
    def test_register_via_client(self, monkeypatch):
        monkeypatch.setenv("LUCIDIC_DEBUG", "false")
        monkeypatch.setenv("LUCIDIC_BASE_URL", "https://stub.lucidic.test")
        from lucidicai import LucidicAI

        client = LucidicAI(
            api_key="test-key",
            agent_id="00000000-0000-0000-0000-000000000000",
            production=True,
        )
        surfaces = client.tools.register_anthropic([_spec(name="instance_tool")])
        assert [s.name for s in surfaces] == ["instance_tool"]
        assert "instance_tool" in client.tools._registry
