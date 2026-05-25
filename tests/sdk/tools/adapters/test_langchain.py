"""Tests for ``lucidicai.sdk.tools.adapters.langchain``.

Covers surface extraction from both ``Tool`` (no ``args_schema``) and
``StructuredTool`` (Pydantic ``args_schema``) shapes, transparent
``tool.func`` replacement, idempotency of re-registration, and
dispatch behavior via LangChain's own ``tool.invoke()`` path.

LangChain is the most invasive adapter — it replaces the framework's
function pointer, so these tests pin the contract that
``tool.invoke(...)`` routes through our backend when a ``MockContext``
is bound and runs the original callable otherwise.
"""
import asyncio

import httpx
import pytest
import respx
from langchain_core.tools import StructuredTool, Tool

from lucidicai.sdk.tools.adapters.langchain import (
    _compute_langchain_source_hash,
    _looks_like_basetool,
    _surface_from_langchain_tool,
    register_langchain_tools,
)
from lucidicai.sdk.tools.context import MockContext, bind_mock_context
from lucidicai.sdk.tools.registry import _REGISTRY


_MOCK_CALL_ENDPOINT = "https://stub.lucidic.test/sdk/mock-call"


# ---------- _looks_like_basetool -----------------------------------------


class TestLooksLikeBasetool:
    def test_real_tool_passes(self):
        t = Tool(name="t", description="d", func=lambda x: x)
        assert _looks_like_basetool(t) is True

    def test_real_structured_tool_passes(self):
        def fn(x: int) -> int:
            """LangChain requires a docstring when no description given."""
            return x
        t = StructuredTool.from_function(func=fn)
        assert _looks_like_basetool(t) is True

    def test_plain_dict_rejected(self):
        # dicts don't have .name as an attribute (only as a key), so they
        # fail the duck-type check. Reject is the correct behavior — the
        # caller meant to pass a BaseTool, not a raw dict.
        assert _looks_like_basetool({"name": "t", "func": lambda: 0}) is False

    def test_no_callable_rejected(self):
        from types import SimpleNamespace
        assert _looks_like_basetool(SimpleNamespace(name="x")) is False

    def test_no_name_rejected(self):
        assert _looks_like_basetool(lambda: 0) is False


# ---------- Surface extraction -------------------------------------------


class TestSurfaceExtraction:
    def test_plain_tool_with_no_args_schema(self):
        def query(q: str) -> str:
            """Search."""
            return q
        t = Tool(name="query", description="Run a query", func=query)
        surface = _surface_from_langchain_tool(t)
        assert surface.name == "query"
        assert surface.docstring == "Run a query"
        # Plain Tool has no args_schema → fall back to callable introspection
        assert surface.signature["params"] == [
            {"name": "q", "type": "str", "default": None, "required": True},
        ]
        assert surface.signature["return_type"] == "str"
        assert len(surface.source_hash) == 64

    def test_structured_tool_uses_args_schema(self):
        def get_weather(city: str, unit: str = "celsius") -> dict:
            """Weather."""
            return {}
        t = StructuredTool.from_function(
            func=get_weather, name="get_weather",
            description="Get the weather",
        )
        surface = _surface_from_langchain_tool(t)
        assert surface.name == "get_weather"
        assert surface.docstring == "Get the weather"
        # StructuredTool synthesizes a Pydantic args_schema; JSON schema
        # type strings ("string") not Python types ("str") — the
        # _params_from_json_schema path was taken.
        names = [p["name"] for p in surface.signature["params"]]
        assert names == ["city", "unit"]
        types = {p["name"]: p["type"] for p in surface.signature["params"]}
        assert types["city"] == "string"
        assert types["unit"] == "string"
        # Defaults preserved
        defaults = {p["name"]: p["default"] for p in surface.signature["params"]}
        assert defaults["unit"] == "celsius"

    def test_source_hash_stable(self):
        sig = {"params": [], "return_type": None}
        h1 = _compute_langchain_source_hash(
            name="t", signature=sig, body_source="def t(): pass",
        )
        h2 = _compute_langchain_source_hash(
            name="t", signature=sig, body_source="def t(): pass",
        )
        assert h1 == h2

    def test_source_hash_picks_up_body_change(self):
        """LangChain adapter is the only one that catches impl-level drift
        because it has access to the user's actual function source."""
        sig = {"params": [], "return_type": None}
        h1 = _compute_langchain_source_hash(
            name="t", signature=sig, body_source="def t(): return 1",
        )
        h2 = _compute_langchain_source_hash(
            name="t", signature=sig, body_source="def t(): return 2",
        )
        assert h1 != h2


# ---------- register_langchain_tools --------------------------------------


class TestRegisterAndWrap:
    def test_registers_into_module_buffer(self):
        t = Tool(name="t1", description="", func=lambda x: x)
        surfaces = register_langchain_tools([t])
        assert [s.name for s in surfaces] == ["t1"]
        assert "t1" in _REGISTRY

    def test_skips_non_basetool_entries(self):
        t = Tool(name="good", description="", func=lambda x: x)
        surfaces = register_langchain_tools([t, "not a tool", 42, None])
        assert [s.name for s in surfaces] == ["good"]

    def test_wraps_tool_func_with_sentinel(self):
        original_called = []
        def original(x):
            original_called.append(x)
            return x * 2

        t = Tool(name="t", description="", func=original)
        # Pre-registration: not wrapped
        assert not getattr(t.func, "__lucidic_wrapped__", False)

        register_langchain_tools([t])
        # Post-registration: wrapped + carries surface
        assert getattr(t.func, "__lucidic_wrapped__", False) is True
        assert hasattr(t.func, "__lucidic_surface__")

        # Calling the wrapped func with no mock context runs the original
        assert t.func(5) == 10
        assert original_called == [5]

    def test_invoke_via_langchain_runs_original_no_context(self):
        def real(q):
            return f"real:{q}"

        t = Tool(name="t", description="", func=real)
        register_langchain_tools([t])
        # LangChain's invoke() goes through tool.func → our wrapper → no
        # mock context → real fn
        assert t.invoke({"q": "hello"}) == "real:hello"

    def test_idempotent_rewrap(self):
        t = Tool(name="t", description="", func=lambda x: x)
        register_langchain_tools([t])
        first_wrapper = t.func
        register_langchain_tools([t])  # same tool registered again
        # Sentinel check skips re-wrap; func unchanged
        assert t.func is first_wrapper

    def test_async_coroutine_wrapped_too(self):
        async def aimpl(x):
            return x + 1
        # StructuredTool can take a coroutine for ainvoke()
        st = StructuredTool.from_function(
            func=lambda x: x, coroutine=aimpl,
            name="dual", description="d",
        )
        register_langchain_tools([st])
        assert getattr(st.func, "__lucidic_wrapped__", False) is True
        assert getattr(st.coroutine, "__lucidic_wrapped__", False) is True


# ---------- Dispatch through LangChain with mock context ------------------


class TestDispatchWithContext:
    @respx.mock
    def test_invoke_routes_through_backend(self, stub_client):
        bind_mock_context(MockContext(session_id="s", client=stub_client))
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"return_value": "mocked-by-backend",
                      "tier": "PYTHON", "was_mocked": True},
            )
        )
        t = Tool(name="t", description="",
                 func=lambda q: f"local:{q}")
        register_langchain_tools([t])

        # LangChain's tool.invoke goes through our wrapper now
        result = t.invoke({"q": "hello"})
        assert result == "mocked-by-backend"

    @respx.mock
    def test_invoke_pass_through_falls_back_local(self, stub_client):
        bind_mock_context(MockContext(session_id="s", client=stub_client))
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"return_value": None, "tier": "PASS_THROUGH",
                      "was_mocked": False},
            )
        )
        called = []
        def real(q):
            called.append(q)
            return f"local:{q}"

        t = Tool(name="t", description="", func=real)
        register_langchain_tools([t])
        result = t.invoke({"q": "x"})
        assert result == "local:x"
        assert called == ["x"]

    @respx.mock
    def test_invoke_drift_falls_back(self, stub_client, caplog):
        import logging
        bind_mock_context(MockContext(session_id="s", client=stub_client))
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                409,
                json={"error": {"code": "tool_drift", "detail": "drift",
                                "session_hash": "aaa", "current_hash": "bbb"}},
            )
        )
        t = Tool(name="t", description="", func=lambda q: f"local:{q}")
        register_langchain_tools([t])
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            result = t.invoke({"q": "x"})
        assert result == "local:x"
        assert any("drift" in r.message.lower() for r in caplog.records)


# ---------- Async dispatch -----------------------------------------------


class TestAsyncDispatch:
    @respx.mock
    @pytest.mark.asyncio
    async def test_ainvoke_routes_through_backend(self, stub_client):
        bind_mock_context(MockContext(session_id="s", client=stub_client))
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"return_value": "from-backend",
                      "tier": "PYTHON", "was_mocked": True},
            )
        )

        async def acoroutine(q: str) -> str:
            return f"local-async:{q}"

        st = StructuredTool.from_function(
            func=lambda q: f"local-sync:{q}",
            coroutine=acoroutine,
            name="dual", description="d",
        )
        register_langchain_tools([st])
        result = await st.ainvoke({"q": "x"})
        assert result == "from-backend"

    @respx.mock
    @pytest.mark.asyncio
    async def test_ainvoke_pass_through_runs_async_local(self, stub_client):
        bind_mock_context(MockContext(session_id="s", client=stub_client))
        respx.post(_MOCK_CALL_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={"return_value": None, "tier": "PASS_THROUGH",
                      "was_mocked": False},
            )
        )

        async def acoroutine(q):
            return f"local-async:{q}"

        st = StructuredTool.from_function(
            func=lambda q: f"local-sync:{q}",
            coroutine=acoroutine,
            name="dual", description="d",
        )
        register_langchain_tools([st])
        result = await st.ainvoke({"q": "y"})
        assert result == "local-async:y"


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
        t = Tool(name="instance_tool", description="",
                 func=lambda x: x)
        surfaces = client.tools.register_langchain_tools([t])
        assert [s.name for s in surfaces] == ["instance_tool"]
        assert "instance_tool" in client.tools._registry
        # tool.func still wrapped after instance-bound registration
        assert getattr(t.func, "__lucidic_wrapped__", False) is True
