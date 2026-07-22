"""LUC-911 — client.usage.get (org-aggregated usage counters)."""
import httpx
import pytest
import respx

from lucidicai.api.models.usage import Usage
from lucidicai.api.resources.usage import UsageResource
from lucidicai.core.errors import InsufficientScopeError

_BASE = "https://stub.lucidic.test"
_USAGE = f"{_BASE}/sdk/v2/usage"


@pytest.fixture
def usage(http):
    return UsageResource(http)


class TestGet:
    @respx.mock
    def test_returns_mapping(self, usage):
        respx.get(_USAGE).mock(return_value=httpx.Response(
            200, json={"num_sessions": 120, "num_events": 4500, "cost": 3.14}))
        u = usage.get()
        assert isinstance(u, Usage)
        assert u["num_sessions"] == 120
        assert u.get("cost") == 3.14
        assert u.get("missing", 0) == 0
        assert "num_events" in u
        assert set(u.keys()) == {"num_sessions", "num_events", "cost"}
        assert dict(u.items()) == u.to_dict()
        assert len(u) == 3

    @respx.mock
    def test_empty_map(self, usage):
        # org-less key -> {} (backend defensive path)
        respx.get(_USAGE).mock(return_value=httpx.Response(200, json={}))
        u = usage.get()
        assert u.to_dict() == {} and len(u) == 0

    @respx.mock
    def test_missing_scope_raises(self, usage):
        # Data-bearing read: surfaces the typed error, doesn't swallow.
        respx.get(_USAGE).mock(return_value=httpx.Response(
            403, json={"error": "missing scope", "required_scope": "usage:read"}))
        with pytest.raises(InsufficientScopeError):
            usage.get()

    @respx.mock
    @pytest.mark.asyncio
    async def test_aget(self, usage):
        respx.get(_USAGE).mock(return_value=httpx.Response(200, json={"num_sessions": 7}))
        u = await usage.aget()
        assert u["num_sessions"] == 7
