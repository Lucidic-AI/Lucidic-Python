"""Isolation fixtures for adapter tests — clear contextvars between tests.

Mirrors ``tests/sdk/tools/conftest.py``; without this, mock context
or session bindings from one test leak into the next.
"""
import pytest

from lucidicai.sdk.context import current_session_id
from lucidicai.sdk.tools.context import current_mock_context
from lucidicai.sdk.tools.registry import _PENDING_BUFFER, _REGISTRY


@pytest.fixture(autouse=True)
def _isolate():
    _REGISTRY.clear()
    _PENDING_BUFFER.clear()
    current_session_id.set(None)
    current_mock_context.set(None)
    yield
    _REGISTRY.clear()
    _PENDING_BUFFER.clear()
    current_session_id.set(None)
    current_mock_context.set(None)
