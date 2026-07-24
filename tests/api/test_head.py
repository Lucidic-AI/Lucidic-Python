"""LUC-903 — HEAD count/tags helpers."""
import httpx
import pytest
import respx

from lucidicai.core.errors import NotFoundError

_URL = "https://stub.lucidic.test/sdk/v2/sessions"


class TestHead:
    @respx.mock
    def test_returns_headers(self, http):
        respx.head(_URL).mock(return_value=httpx.Response(
            200, headers={"X-Total-Count": "42", "X-Tags": "prod,canary"}))
        headers = http.head("sdk/v2/sessions")
        assert headers["X-Total-Count"] == "42"
        assert headers["X-Tags"] == "prod,canary"

    @respx.mock
    def test_case_insensitive(self, http):
        respx.head(_URL).mock(return_value=httpx.Response(
            200, headers={"X-Total-Count": "7"}))
        headers = http.head("sdk/v2/sessions")
        assert headers["x-total-count"] == "7"

    @respx.mock
    def test_forwards_filter_params(self, http):
        route = respx.head(_URL).mock(return_value=httpx.Response(
            200, headers={"X-Total-Count": "3"}))
        http.head("sdk/v2/sessions", {"production": "true", "agent_id": "a1"})
        sent = route.calls.last.request
        assert sent.url.params["production"] == "true"
        assert sent.url.params["agent_id"] == "a1"

    @respx.mock
    def test_error_raises_typed(self, http):
        respx.head(_URL).mock(return_value=httpx.Response(404))
        with pytest.raises(NotFoundError):
            http.head("sdk/v2/sessions")

    @respx.mock
    @pytest.mark.asyncio
    async def test_ahead_returns_headers(self, http):
        respx.head(_URL).mock(return_value=httpx.Response(
            200, headers={"X-Total-Count": "9"}))
        headers = await http.ahead("sdk/v2/sessions")
        assert headers["X-Total-Count"] == "9"
