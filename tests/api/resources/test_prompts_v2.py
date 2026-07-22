"""LUC-909 — client.prompts reads (list / versions / labels)."""
import httpx
import pytest
import respx

from lucidicai.api.models.prompt import PromptInfo, PromptVersion
from lucidicai.api.resources.prompt import PromptResource
from lucidicai.core.config import NetworkConfig, SDKConfig

_BASE = "https://stub.lucidic.test"
_PROMPTS = f"{_BASE}/sdk/v2/prompts"


@pytest.fixture
def prompts(http):
    cfg = SDKConfig(api_key="test-key", agent_id="a1", network=NetworkConfig(base_url=_BASE))
    return PromptResource(http, cfg, production=False)


def _prompt(i):
    return {"prompt_id": f"p{i}", "name": f"prompt-{i}", "icon": "doc", "preview": "Hello..."}


def _version(n, **over):
    d = {
        "promptversion_id": f"v{n}", "prompt_version_number": n,
        "prompt_content": f"content {n}", "description": "", "metadata": {},
        "created_at": "2026-07-22T00:00:00Z", "labels": [],
    }
    d.update(over)
    return d


class TestList:
    @respx.mock
    def test_follows_pages_typed(self, prompts):
        respx.get(_PROMPTS).mock(side_effect=[
            httpx.Response(200, json={"results": [_prompt(1), _prompt(2)],
                                      "next": f"{_PROMPTS}?cursor=c2", "previous": None}),
            httpx.Response(200, json={"results": [_prompt(3)], "next": None}),
        ])
        got = list(prompts.list())
        assert [p.prompt_id for p in got] == ["p1", "p2", "p3"]
        assert all(isinstance(p, PromptInfo) for p in got)

    @respx.mock
    def test_defaults_configured_agent_id(self, prompts):
        route = respx.get(_PROMPTS).mock(
            return_value=httpx.Response(200, json={"results": [], "next": None}))
        list(prompts.list(ordering="name"))
        p = route.calls.last.request.url.params
        assert p["agent_id"] == "a1"
        assert p["ordering"] == "name"

    @respx.mock
    def test_list_page(self, prompts):
        respx.get(_PROMPTS).mock(return_value=httpx.Response(
            200, json={"results": [_prompt(1)], "next": f"{_PROMPTS}?cursor=c9"}))
        page = prompts.list_page()
        assert isinstance(page.results[0], PromptInfo) and page.next_cursor == "c9"


class TestVersions:
    @respx.mock
    def test_sends_prompt_name_and_typed(self, prompts):
        route = respx.get(f"{_PROMPTS}/versions").mock(side_effect=[
            httpx.Response(200, json={"results": [_version(2, labels=["latest", "production"])],
                                      "next": f"{_PROMPTS}/versions?cursor=c2"}),
            httpx.Response(200, json={"results": [_version(1)], "next": None}),
        ])
        got = list(prompts.versions("greeting"))
        assert route.calls[0].request.url.params["prompt_name"] == "greeting"
        assert route.calls[0].request.url.params["agent_id"] == "a1"
        assert [v.prompt_version_number for v in got] == [2, 1]
        assert all(isinstance(v, PromptVersion) for v in got)
        assert got[0].labels == ["latest", "production"]

    @respx.mock
    def test_versions_page(self, prompts):
        respx.get(f"{_PROMPTS}/versions").mock(return_value=httpx.Response(
            200, json={"results": [_version(1)], "next": None}))
        page = prompts.versions_page("greeting")
        assert isinstance(page.results[0], PromptVersion)


class TestLabels:
    @respx.mock
    def test_labels_unwrapped(self, prompts):
        route = respx.get(f"{_PROMPTS}/labels").mock(return_value=httpx.Response(
            200, json={"labels": ["latest", "production", "staging"]}))
        assert prompts.labels() == ["latest", "production", "staging"]
        assert route.calls.last.request.url.params["agent_id"] == "a1"

    @respx.mock
    def test_labels_empty(self, prompts):
        respx.get(f"{_PROMPTS}/labels").mock(return_value=httpx.Response(200, json={"labels": []}))
        assert prompts.labels() == []


class TestAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_alist(self, prompts):
        respx.get(_PROMPTS).mock(side_effect=[
            httpx.Response(200, json={"results": [_prompt(1)], "next": f"{_PROMPTS}?cursor=c2"}),
            httpx.Response(200, json={"results": [_prompt(2)], "next": None}),
        ])
        got = [p.prompt_id async for p in prompts.alist()]
        assert got == ["p1", "p2"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_aversions(self, prompts):
        respx.get(f"{_PROMPTS}/versions").mock(return_value=httpx.Response(
            200, json={"results": [_version(1)], "next": None}))
        got = [v.prompt_version_number async for v in prompts.aversions("greeting")]
        assert got == [1]

    @respx.mock
    @pytest.mark.asyncio
    async def test_alabels(self, prompts):
        respx.get(f"{_PROMPTS}/labels").mock(return_value=httpx.Response(
            200, json={"labels": ["latest"]}))
        assert await prompts.alabels() == ["latest"]
