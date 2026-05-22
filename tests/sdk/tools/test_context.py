"""Tests for ``lucidicai.sdk.tools.context``.

Pins MockContext semantics + contextvar binding. Coverage matters
because the @mockable wrapper consults _current_mock_context() on
every call — a regression here silently bypasses backend dispatch.
"""
import asyncio
from types import SimpleNamespace

import pytest

from lucidicai.sdk.tools.context import (
    MockContext,
    _current_mock_context,
    bind_mock_context,
    clear_mock_context,
    current_mock_context,
)


@pytest.fixture(autouse=True)
def _clear_context_var():
    """Reset the contextvar between tests. Module global persists
    across tests otherwise — bound state from one would leak."""
    current_mock_context.set(None)
    yield
    current_mock_context.set(None)


def _make_ctx(session_id="sess-1") -> MockContext:
    fake_client = SimpleNamespace(_resources={})
    return MockContext(session_id=session_id, client=fake_client)


class TestBind:
    def test_no_binding_returns_none(self):
        assert _current_mock_context() is None

    def test_bind_returns_token(self):
        ctx = _make_ctx()
        token = bind_mock_context(ctx)
        assert _current_mock_context() is ctx
        # Token can reset back to None
        current_mock_context.reset(token)
        assert _current_mock_context() is None

    def test_clear_unsets(self):
        bind_mock_context(_make_ctx())
        clear_mock_context()
        assert _current_mock_context() is None

    def test_rebind_replaces(self):
        bind_mock_context(_make_ctx(session_id="first"))
        bind_mock_context(_make_ctx(session_id="second"))
        assert _current_mock_context().session_id == "second"


class TestAsyncIsolation:
    @pytest.mark.asyncio
    async def test_concurrent_tasks_dont_leak(self):
        """ContextVar gives each asyncio task its own value.

        Critical for tool-backed eval runs that fire many sessions in
        parallel — task A's mock context must not leak into task B.
        """
        results = []

        async def task(name: str):
            bind_mock_context(_make_ctx(session_id=name))
            # Yield to scheduler so other tasks interleave
            await asyncio.sleep(0)
            # Each task should see ONLY its own binding
            ctx = _current_mock_context()
            results.append((name, ctx.session_id if ctx else None))

        await asyncio.gather(task("a"), task("b"), task("c"))
        # Order isn't guaranteed but each task sees its own name
        results_dict = dict(results)
        assert results_dict == {"a": "a", "b": "b", "c": "c"}
