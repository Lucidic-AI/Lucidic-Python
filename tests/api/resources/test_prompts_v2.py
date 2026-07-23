"""LUC-909 / LUC-914 — client.prompts reads + writes."""
import json

import httpx
import pytest
import respx

from lucidicai.api.models.prompt import PromptInfo, PromptVersion
from lucidicai.api.resources.prompt import PromptResource
from lucidicai.core.config import NetworkConfig, SDKConfig
from lucidicai.core.errors import (
    AgentIdRequiredError,
    ConflictError,
    InsufficientScopeError,
    ValidationError,
)

_BASE = "https://stub.lucidic.test"
_PROMPTS = f"{_BASE}/sdk/v2/prompts"
_DETAIL = f"{_PROMPTS}/detail"
_LABELS = f"{_PROMPTS}/labels"


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


def _ct(body):
    # PATCH/PUT bodies carry an auto-injected current_time; ignore in equality.
    return {"current_time": body["current_time"]} if "current_time" in body else {}


class TestCreate:
    """LUC-914 follow-up (unblocked by backend LUC-931): POST /sdk/v2/prompts —
    create a brand-new prompt + its first version."""

    @respx.mock
    def test_create_prompt_and_typed_first_version(self, prompts):
        route = respx.post(_PROMPTS).mock(return_value=httpx.Response(
            201, json={**_prompt(1), "version": _version(1, labels=["latest", "production"])}))
        info, v = prompts.create("prompt-1", "content 1")
        assert isinstance(info, PromptInfo) and info.prompt_id == "p1" and info.name == "prompt-1"
        assert isinstance(v, PromptVersion) and v.prompt_version_number == 1
        assert set(v.labels) == {"latest", "production"}
        body = json.loads(route.calls.last.request.read())
        assert body["agent_id"] == "a1"  # defaulted from config
        assert body["name"] == "prompt-1" and body["prompt_content"] == "content 1"
        # omit-None: unspecified optionals are NOT sent (backend applies its defaults)
        assert not any(k in body for k in ("icon", "description", "metadata", "labels"))

    @respx.mock
    def test_create_all_fields_sent(self, prompts):
        route = respx.post(_PROMPTS).mock(return_value=httpx.Response(201, json=_prompt(1)))
        prompts.create("prompt-1", "c", icon="star", description="d",
                       metadata={"k": "v"}, labels=["staging"])
        body = json.loads(route.calls.last.request.read())
        assert body["icon"] == "star" and body["description"] == "d"
        assert body["metadata"] == {"k": "v"} and body["labels"] == ["staging"]

    @respx.mock
    def test_create_explicit_agent_id_overrides_config(self, prompts):
        route = respx.post(_PROMPTS).mock(return_value=httpx.Response(201, json=_prompt(1)))
        prompts.create("prompt-1", "c", agent_id="a2")
        assert json.loads(route.calls.last.request.read())["agent_id"] == "a2"

    @respx.mock
    def test_create_without_version_subobject(self, prompts):
        # Forward-compat: a response missing the version subobject yields (PromptInfo, None).
        respx.post(_PROMPTS).mock(return_value=httpx.Response(201, json=_prompt(1)))
        info, v = prompts.create("prompt-1", "c")
        assert isinstance(info, PromptInfo) and v is None

    @respx.mock
    def test_create_duplicate_name_409(self, prompts):
        respx.post(_PROMPTS).mock(return_value=httpx.Response(
            409, json={"error": "A prompt named 'prompt-1' already exists for this agent."}))
        with pytest.raises(ConflictError):
            prompts.create("prompt-1", "c")

    @respx.mock
    def test_create_missing_content_400(self, prompts):
        respx.post(_PROMPTS).mock(return_value=httpx.Response(
            400, json={"error": "This field may not be blank."}))
        with pytest.raises(ValidationError):
            prompts.create("prompt-1", "")

    @respx.mock
    def test_create_without_write_scope_403(self, prompts):
        respx.post(_PROMPTS).mock(return_value=httpx.Response(
            403, json={"error": "requires prompt:write"}))
        with pytest.raises(InsufficientScopeError):
            prompts.create("prompt-1", "c")

    @respx.mock
    def test_create_invalidates_stale_same_name_cache(self, prompts):
        # A name deleted elsewhere then recreated must not keep serving old cached content.
        respx.post(_PROMPTS).mock(return_value=httpx.Response(201, json=_prompt(1)))
        prompts._cache[("prompt-1", "production", None, None)] = {"content": "old", "timestamp": 0}
        prompts.create("prompt-1", "c")
        assert not any(k[0] == "prompt-1" for k in prompts._cache)

    def test_create_without_agent_id_raises(self, http):
        # No HTTP call: the guard fires client-side before the request.
        res = PromptResource(http, SDKConfig(
            api_key="k", agent_id=None, network=NetworkConfig(base_url=_BASE)))
        with pytest.raises(AgentIdRequiredError):
            res.create("prompt-1", "c")

    @respx.mock
    @pytest.mark.asyncio
    async def test_acreate(self, prompts):
        respx.post(_PROMPTS).mock(return_value=httpx.Response(
            201, json={**_prompt(1), "version": _version(1)}))
        info, v = await prompts.acreate("prompt-1", "c")
        assert isinstance(info, PromptInfo) and isinstance(v, PromptVersion)


class TestRename:
    @respx.mock
    def test_rename_and_reicon(self, prompts):
        route = respx.patch(_DETAIL).mock(return_value=httpx.Response(200, json=_prompt(1)))
        p = prompts.rename("greeting", new_name="salutation", icon="wave")
        assert isinstance(p, PromptInfo)
        body = json.loads(route.calls.last.request.read())
        assert body["agent_id"] == "a1" and body["prompt_name"] == "greeting"
        assert body["name"] == "salutation" and body["icon"] == "wave"

    @respx.mock
    def test_rename_only_provided(self, prompts):
        route = respx.patch(_DETAIL).mock(return_value=httpx.Response(200, json=_prompt(1)))
        prompts.rename("greeting", icon="wave")
        body = json.loads(route.calls.last.request.read())
        assert "name" not in body and body["icon"] == "wave"

    @respx.mock
    def test_warning_on_extra(self, prompts):
        respx.patch(_DETAIL).mock(return_value=httpx.Response(
            200, json=dict(_prompt(1), warning="clients fetching the old name will 404")))
        p = prompts.rename("greeting", new_name="salutation")
        assert "404" in p.extra["warning"]

    @respx.mock
    def test_collision_409(self, prompts):
        respx.patch(_DETAIL).mock(return_value=httpx.Response(
            409, json={"error": "A prompt named 'x' already exists for this agent."}))
        with pytest.raises(ConflictError):
            prompts.rename("greeting", new_name="x")


class TestSetLabels:
    @respx.mock
    def test_promote(self, prompts):
        route = respx.put(_LABELS).mock(return_value=httpx.Response(
            200, json=_version(3, labels=["production"])))
        v = prompts.set_labels("greeting", 3, ["production"])
        assert isinstance(v, PromptVersion) and v.labels == ["production"]
        body = json.loads(route.calls.last.request.read())
        assert body == {"agent_id": "a1", "prompt_name": "greeting",
                        "version_number": 3, "labels": ["production"], **_ct(body)}

    @respx.mock
    def test_warning_on_extra(self, prompts):
        respx.put(_LABELS).mock(return_value=httpx.Response(
            200, json=dict(_version(3), warning="version 3 previously had no labels")))
        v = prompts.set_labels("greeting", 3, ["production"])
        assert "no labels" in v.extra["warning"]

    @respx.mock
    def test_move_latest_400(self, prompts):
        respx.put(_LABELS).mock(return_value=httpx.Response(
            400, json={"error": "'latest' is auto-managed"}))
        with pytest.raises(ValidationError):
            prompts.set_labels("greeting", 3, ["latest"])


class TestDelete:
    @respx.mock
    def test_delete_204_sends_query(self, prompts):
        route = respx.delete(_DETAIL).mock(return_value=httpx.Response(204))
        assert prompts.delete("greeting") is None
        p = route.calls.last.request.url.params
        assert p["agent_id"] == "a1" and p["prompt_name"] == "greeting"

    @respx.mock
    def test_checkpoint_bound_409(self, prompts):
        respx.delete(_DETAIL).mock(return_value=httpx.Response(
            409, json={"error": "This prompt has a version bound to a checkpoint and cannot be deleted."}))
        with pytest.raises(ConflictError):
            prompts.delete("greeting")


class TestAsyncWrite:
    @respx.mock
    @pytest.mark.asyncio
    async def test_arename(self, prompts):
        respx.patch(_DETAIL).mock(return_value=httpx.Response(200, json=_prompt(1)))
        p = await prompts.arename("greeting", new_name="salutation")
        assert isinstance(p, PromptInfo)

    @respx.mock
    @pytest.mark.asyncio
    async def test_aset_labels(self, prompts):
        respx.put(_LABELS).mock(return_value=httpx.Response(200, json=_version(3)))
        v = await prompts.aset_labels("greeting", 3, ["production"])
        assert v.prompt_version_number == 3

    @respx.mock
    @pytest.mark.asyncio
    async def test_adelete(self, prompts):
        route = respx.delete(_DETAIL).mock(return_value=httpx.Response(204))
        assert await prompts.adelete("greeting") is None
        assert route.called


class TestCacheInvalidation:
    """The writes must clear the local get() cache (keyed by prompt_name) —
    like update()/update_metadata() do — else a promoted/renamed/deleted prompt
    keeps serving stale cached content."""

    @staticmethod
    def _seed(prompts):
        prompts._cache[("greeting", "production", None, None)] = {"content": "old", "timestamp": 0}

    @respx.mock
    def test_rename_invalidates(self, prompts):
        respx.patch(_DETAIL).mock(return_value=httpx.Response(200, json=_prompt(1)))
        self._seed(prompts)
        prompts.rename("greeting", new_name="salutation")
        assert not any(k[0] == "greeting" for k in prompts._cache)

    @respx.mock
    def test_set_labels_invalidates(self, prompts):
        respx.put(_LABELS).mock(return_value=httpx.Response(200, json=_version(3)))
        self._seed(prompts)
        prompts.set_labels("greeting", 3, ["production"])
        assert not any(k[0] == "greeting" for k in prompts._cache)

    @respx.mock
    def test_delete_invalidates(self, prompts):
        respx.delete(_DETAIL).mock(return_value=httpx.Response(204))
        self._seed(prompts)
        prompts.delete("greeting")
        assert not any(k[0] == "greeting" for k in prompts._cache)
