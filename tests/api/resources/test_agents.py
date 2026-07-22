"""LUC-906 / LUC-912 — client.agents reads + writes.

Uses the hermetic ``http`` fixture from tests/api/conftest.py (stub URL,
retry sleeps irrelevant here). Backend mocked at the HTTP boundary via respx.
"""
import json

import httpx
import pytest
import respx

from lucidicai.api.models.agent import Agent, AgentToolCatalog, CatalogTool
from lucidicai.api.resources.agents import AgentsResource
from lucidicai.core.errors import InsufficientScopeError, NotFoundError, ValidationError

_BASE = "https://stub.lucidic.test"
_AGENTS = f"{_BASE}/sdk/v2/agents"


@pytest.fixture
def agents(http):
    return AgentsResource(http)


def _agent(i, **over):
    d = {
        "agent_id": f"a{i}",
        "name": f"agent-{i}",
        "icon": "compass",
        "project_id": None,
        "created_at": "2026-07-22T00:00:00Z",
    }
    d.update(over)
    return d


class TestGet:
    @respx.mock
    def test_returns_typed_agent(self, agents):
        respx.get(f"{_AGENTS}/a1").mock(return_value=httpx.Response(200, json=_agent(1)))
        a = agents.get("a1")
        assert isinstance(a, Agent)
        assert a.agent_id == "a1"
        assert a.name == "agent-1"
        assert a.icon == "compass"

    @respx.mock
    def test_404_raises_not_found(self, agents):
        respx.get(f"{_AGENTS}/missing").mock(
            return_value=httpx.Response(404, json={"error": "not found"}))
        with pytest.raises(NotFoundError):
            agents.get("missing")

    @respx.mock
    def test_tolerates_unknown_fields(self, agents):
        respx.get(f"{_AGENTS}/a1").mock(
            return_value=httpx.Response(200, json=_agent(1, future_field="x")))
        a = agents.get("a1")
        assert a.extra == {"future_field": "x"}


class TestList:
    @respx.mock
    def test_follows_pages(self, agents):
        respx.get(_AGENTS).mock(side_effect=[
            httpx.Response(200, json={"results": [_agent(1), _agent(2)],
                                      "next": f"{_AGENTS}?cursor=c2", "previous": None}),
            httpx.Response(200, json={"results": [_agent(3)], "next": None, "previous": None}),
        ])
        got = list(agents.list())
        assert [a.agent_id for a in got] == ["a1", "a2", "a3"]
        assert all(isinstance(a, Agent) for a in got)

    @respx.mock
    def test_forwards_ordering_and_page_size(self, agents):
        route = respx.get(_AGENTS).mock(
            return_value=httpx.Response(200, json={"results": [_agent(1)], "next": None}))
        list(agents.list(ordering="id", page_size=10))
        sent = route.calls.last.request
        assert sent.url.params["ordering"] == "id"
        assert sent.url.params["page_size"] == "10"

    @respx.mock
    def test_list_page_returns_cursor_page(self, agents):
        respx.get(_AGENTS).mock(return_value=httpx.Response(
            200, json={"results": [_agent(1)], "next": f"{_AGENTS}?cursor=c9", "previous": None}))
        page = agents.list_page()
        assert len(page.results) == 1 and isinstance(page.results[0], Agent)
        assert page.next_cursor == "c9" and page.has_next


class TestToolCatalog:
    @respx.mock
    def test_typed_catalog_with_typed_tools(self, agents):
        catalog = {
            "agent_id": "a1",
            "agent_name": "agent-1",
            "tools": [{
                "tool_id": "t1", "name": "query_sql", "tier": "SQL_TEMPLATE",
                "call_count": 5, "last_called_at": "2026-07-22T00:00:00Z",
                "has_drift": False,
                "resources": [{"resource_id": "r1", "name": "db", "type": "SQL"}],
            }],
            "resources": [{"resource_id": "r1", "name": "db", "type": "SQL",
                           "tool_names": ["query_sql"]}],
        }
        respx.get(f"{_AGENTS}/a1/tool-catalog").mock(
            return_value=httpx.Response(200, json=catalog))
        result = agents.tool_catalog("a1")
        assert isinstance(result, AgentToolCatalog)
        assert result.agent_name == "agent-1"
        assert len(result.tools) == 1
        assert isinstance(result.tools[0], CatalogTool)
        assert result.tools[0].call_count == 5
        assert result.tools[0].name == "query_sql"
        # The aux resource union stays as raw dicts.
        assert result.resources[0]["resource_id"] == "r1"

    @respx.mock
    def test_empty_catalog(self, agents):
        respx.get(f"{_AGENTS}/a1/tool-catalog").mock(return_value=httpx.Response(
            200, json={"agent_id": "a1", "agent_name": "n", "tools": [], "resources": []}))
        result = agents.tool_catalog("a1")
        assert result.tools == [] and result.resources == []

    @respx.mock
    def test_null_tools_and_resources_become_empty(self, agents):
        # Backend `null` (not []) for the nested lists must not crash or leave
        # None — the reference nested-conversion + central null-normalization
        # both turn it into [].
        respx.get(f"{_AGENTS}/a1/tool-catalog").mock(return_value=httpx.Response(
            200, json={"agent_id": "a1", "agent_name": "n", "tools": None, "resources": None}))
        result = agents.tool_catalog("a1")
        assert result.tools == [] and result.resources == []


class TestAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_aget(self, agents):
        respx.get(f"{_AGENTS}/a1").mock(return_value=httpx.Response(200, json=_agent(1)))
        a = await agents.aget("a1")
        assert a.agent_id == "a1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_alist_follows_pages(self, agents):
        respx.get(_AGENTS).mock(side_effect=[
            httpx.Response(200, json={"results": [_agent(1)], "next": f"{_AGENTS}?cursor=c2"}),
            httpx.Response(200, json={"results": [_agent(2)], "next": None}),
        ])
        got = [a.agent_id async for a in agents.alist()]
        assert got == ["a1", "a2"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_atool_catalog(self, agents):
        respx.get(f"{_AGENTS}/a1/tool-catalog").mock(return_value=httpx.Response(
            200, json={"agent_id": "a1", "agent_name": "n", "tools": [], "resources": []}))
        c = await agents.atool_catalog("a1")
        assert c.agent_id == "a1" and c.tools == []


class TestCreate:
    @respx.mock
    def test_create_omits_unset_optionals(self, agents):
        route = respx.post(_AGENTS).mock(return_value=httpx.Response(201, json=_agent(1)))
        a = agents.create("agent-1")
        assert isinstance(a, Agent) and a.agent_id == "a1"
        body = json.loads(route.calls.last.request.read())
        assert body["name"] == "agent-1"
        assert "icon" not in body            # omitted -> backend defaults ("compass")
        assert "project_id" not in body

    @respx.mock
    def test_with_icon_and_project(self, agents):
        route = respx.post(_AGENTS).mock(return_value=httpx.Response(201, json=_agent(1)))
        agents.create("a", icon="rocket", project_id="p1")
        body = json.loads(route.calls.last.request.read())
        assert body["icon"] == "rocket" and body["project_id"] == "p1"

    @respx.mock
    def test_bound_key_403_is_insufficient_scope(self, agents):
        respx.post(_AGENTS).mock(return_value=httpx.Response(
            403, json={"error": "An agent-bound API key cannot create new agents."}))
        with pytest.raises(InsufficientScopeError):
            agents.create("x")


class TestUpdate:
    @respx.mock
    def test_sends_only_provided_fields(self, agents):
        route = respx.put(f"{_AGENTS}/a1").mock(return_value=httpx.Response(200, json=_agent(1)))
        a = agents.update("a1", name="renamed")
        assert isinstance(a, Agent)
        body = json.loads(route.calls.last.request.read())
        assert body["name"] == "renamed"
        assert "icon" not in body and "project_id" not in body

    @respx.mock
    def test_all_fields(self, agents):
        route = respx.put(f"{_AGENTS}/a1").mock(return_value=httpx.Response(200, json=_agent(1)))
        agents.update("a1", name="n", icon="i", project_id="p1")
        body = json.loads(route.calls.last.request.read())
        assert body["name"] == "n" and body["icon"] == "i" and body["project_id"] == "p1"

    @respx.mock
    def test_over_length_400_is_validation_error(self, agents):
        respx.put(f"{_AGENTS}/a1").mock(return_value=httpx.Response(
            400, json={"error": "A field exceeds its maximum length."}))
        with pytest.raises(ValidationError):
            agents.update("a1", name="x" * 999)


class TestAsyncWrite:
    @respx.mock
    @pytest.mark.asyncio
    async def test_acreate(self, agents):
        respx.post(_AGENTS).mock(return_value=httpx.Response(201, json=_agent(1)))
        a = await agents.acreate("agent-1")
        assert a.agent_id == "a1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_aupdate(self, agents):
        respx.put(f"{_AGENTS}/a1").mock(return_value=httpx.Response(200, json=_agent(1)))
        a = await agents.aupdate("a1", icon="star")
        assert a.agent_id == "a1"
