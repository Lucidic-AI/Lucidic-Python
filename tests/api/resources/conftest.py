"""Isolation fixtures for tests/api/resources/.

Same rationale as ``tests/sdk/tools/conftest.py`` — clear contextvars
between tests so a session_id set by an earlier test doesn't leak
into the legacy ``test_no_active_session_returns_none`` path that
relies on the contextvar being None.
"""
import pytest

from lucidicai.sdk.context import current_session_id
from lucidicai.sdk.tools.context import current_mock_context


@pytest.fixture(autouse=True)
def _isolate_context_vars():
    current_session_id.set(None)
    current_mock_context.set(None)
    yield
    current_session_id.set(None)
    current_mock_context.set(None)
