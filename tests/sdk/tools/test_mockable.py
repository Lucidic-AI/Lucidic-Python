"""Tests for ``@mockable`` decorator behavior.

Covers surface capture for plain/typed/optional params, async detection,
lambda rejection, and registry integration. The actual dispatch
interception (consult ``_current_mock_context()``, route through transport)
is wired in LUC-577c; the v1 of this ticket's wrapper is intentionally
transparent — these tests pin "decorating doesn't change call semantics".
"""
import asyncio

import pytest

from lucidicai.core.errors import LucidicError
from lucidicai.sdk.tools.mockable import mockable
from lucidicai.sdk.tools.registry import _PENDING_BUFFER, _REGISTRY


@pytest.fixture(autouse=True)
def _clear_module_state():
    _REGISTRY.clear()
    _PENDING_BUFFER.clear()
    yield
    _REGISTRY.clear()
    _PENDING_BUFFER.clear()


# ---------- Surface capture ----------------------------------------------


class TestSurfaceCapture:
    def test_basic_typed_function(self):
        @mockable
        def query_unread_emails(sender: str, limit: int = 50) -> list[dict]:
            """Return unread emails."""
            return []

        s = query_unread_emails.__lucidic_surface__
        assert s.name == "query_unread_emails"
        assert s.docstring == "Return unread emails."
        assert s.return_shape is None
        assert s.signature["return_type"] == "list"
        assert s.signature["params"] == [
            {"name": "sender", "type": "str", "default": None, "required": True},
            {"name": "limit", "type": "int", "default": 50, "required": False},
        ]
        assert len(s.source_hash) == 64

    def test_no_docstring(self):
        @mockable
        def f(x: int) -> int:
            return x
        assert f.__lucidic_surface__.docstring == ""

    def test_no_return_annotation(self):
        @mockable
        def f(x: int):
            return x
        assert f.__lucidic_surface__.signature["return_type"] is None

    def test_untyped_params(self):
        @mockable
        def f(a, b=1):
            return a, b
        params = f.__lucidic_surface__.signature["params"]
        assert params[0]["type"] == "any"
        assert params[1]["type"] == "any"

    def test_multiline_docstring_dedented(self):
        @mockable
        def f():
            """Summary line.

            More detail across
            multiple lines.
            """
            pass
        # inspect.getdoc dedents — pin that behavior since drift detection
        # would otherwise trip on indentation changes that don't change content
        assert f.__lucidic_surface__.docstring.startswith("Summary line.")
        assert "More detail across\nmultiple lines." in f.__lucidic_surface__.docstring


# ---------- Sync wrapper transparency ------------------------------------


class TestSyncWrapper:
    def test_call_returns_func_result(self):
        @mockable
        def add(a: int, b: int) -> int:
            return a + b
        assert add(2, 3) == 5

    def test_call_propagates_exceptions(self):
        @mockable
        def boom():
            raise ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            boom()

    def test_kwargs_work(self):
        @mockable
        def f(a, b=10):
            return a, b
        assert f(1) == (1, 10)
        assert f(1, b=20) == (1, 20)
        assert f(a=1) == (1, 10)


# ---------- Async wrapper -------------------------------------------------


class TestAsyncWrapper:
    def test_async_decoration_sniffed(self):
        @mockable
        async def afoo(x: int) -> int:
            return x

        assert asyncio.iscoroutinefunction(afoo)
        assert afoo.__lucidic_surface__.name == "afoo"

    def test_async_call_returns_result(self):
        @mockable
        async def afoo(x: int) -> int:
            return x * 2

        assert asyncio.run(afoo(5)) == 10

    def test_async_call_propagates_exceptions(self):
        @mockable
        async def aboom():
            raise RuntimeError("async-boom")

        with pytest.raises(RuntimeError, match="async-boom"):
            asyncio.run(aboom())


# ---------- Lambda rejection ----------------------------------------------


class TestLambdaRejection:
    def test_lambda_raises(self):
        with pytest.raises(LucidicError, match="valid Python identifier"):
            mockable(lambda x: x)


# ---------- Registry integration ------------------------------------------


class TestRegistryIntegration:
    def test_decoration_registers(self):
        @mockable
        def my_tool(x: int) -> int:
            return x
        assert "my_tool" in _REGISTRY
        assert _REGISTRY["my_tool"].name == "my_tool"

    def test_redecoration_replaces_in_registry(self):
        @mockable
        def my_tool(x: int) -> int:
            return x

        first_hash = _REGISTRY["my_tool"].source_hash

        @mockable
        def my_tool(x: int, y: int = 0) -> int:  # noqa: F811 — intentional override
            return x + y

        assert _REGISTRY["my_tool"].source_hash != first_hash
        assert len(_REGISTRY["my_tool"].signature["params"]) == 2

    def test_buffer_populated_outside_active_client(self):
        # The autouse fixture clears state; with no active client, decorating
        # falls through to the buffer.
        @mockable
        def t1():
            pass
        @mockable
        def t2():
            pass
        assert {s.name for s in _PENDING_BUFFER} == {"t1", "t2"}


# ---------- Hash equivalence smoke ----------------------------------------


class TestHashSemantics:
    """A few end-to-end checks via the decorator. The deep hash tests
    live in test_registry.py — these confirm the decorator integrates
    with _compute_source_hash correctly."""

    def test_same_function_same_hash(self):
        def make_tool():
            @mockable
            def emit(x: int) -> int:
                return x + 1
            return emit
        h1 = make_tool().__lucidic_surface__.source_hash
        _REGISTRY.clear()
        _PENDING_BUFFER.clear()
        h2 = make_tool().__lucidic_surface__.source_hash
        assert h1 == h2

    def test_signature_only_change_changes_hash(self):
        @mockable
        def emit(x: int) -> int:
            return x
        h_orig = emit.__lucidic_surface__.source_hash

        @mockable
        def emit(x: str) -> int:  # noqa: F811 — int → str
            return 0  # body also changes — both should affect hash
        assert emit.__lucidic_surface__.source_hash != h_orig
