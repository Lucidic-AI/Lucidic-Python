"""LUC-917 — client.resources CRUD (SQL-substrate resources; new namespace)."""
import json

import httpx
import pytest
import respx

from lucidicai.api.models.resource import Resource
from lucidicai.api.resources.resources import ResourcesResource
from lucidicai.core.errors import ConflictError, NotFoundError, ValidationError

_BASE = "https://stub.lucidic.test"
_RESOURCES = f"{_BASE}/sdk/v2/resources"


@pytest.fixture
def resources(http):
    return ResourcesResource(http)


def _resource(i, **over):
    d = {
        "resource_id": f"res{i}", "type": "SQL", "dialect": "POSTGRES",
        "tool_name": f"tool_{i}", "name": f"resource-{i}", "description": "",
        "spec": {"tables": []}, "handler_type": "SQL",
        "hydration_config": {"default_rows_per_table": 100},
        "created_at": "2026-07-22T00:00:00Z", "updated_at": "2026-07-22T00:00:00Z",
    }
    d.update(over)
    return d


class TestList:
    @respx.mock
    def test_follows_pages_typed(self, resources):
        respx.get(_RESOURCES).mock(side_effect=[
            httpx.Response(200, json={"results": [_resource(1), _resource(2)],
                                      "next": f"{_RESOURCES}?cursor=c2", "previous": None}),
            httpx.Response(200, json={"results": [_resource(3)], "next": None}),
        ])
        got = list(resources.list())
        assert [r.resource_id for r in got] == ["res1", "res2", "res3"]
        assert all(isinstance(r, Resource) for r in got)

    @respx.mock
    def test_org_scoped_no_agent_id_param(self, resources):
        # Resources are org-scoped: no agent_id is sent (scoped by the key's org).
        route = respx.get(_RESOURCES).mock(
            return_value=httpx.Response(200, json={"results": [], "next": None}))
        list(resources.list(ordering="id", page_size=10))
        p = route.calls.last.request.url.params
        assert "agent_id" not in p and p["ordering"] == "id" and p["page_size"] == "10"

    @respx.mock
    def test_list_page(self, resources):
        respx.get(_RESOURCES).mock(return_value=httpx.Response(
            200, json={"results": [_resource(1)], "next": f"{_RESOURCES}?cursor=c9"}))
        page = resources.list_page()
        assert isinstance(page.results[0], Resource) and page.next_cursor == "c9"


class TestGet:
    @respx.mock
    def test_get_typed(self, resources):
        respx.get(f"{_RESOURCES}/res1").mock(return_value=httpx.Response(200, json=_resource(1)))
        r = resources.get("res1")
        assert isinstance(r, Resource) and r.resource_id == "res1"
        assert r.type == "SQL" and r.spec == {"tables": []}
        assert r.handler_type == "SQL" and r.hydration_config == {"default_rows_per_table": 100}

    @respx.mock
    def test_get_404(self, resources):
        respx.get(f"{_RESOURCES}/missing").mock(
            return_value=httpx.Response(404, json={"error": "Specified Resource not found"}))
        with pytest.raises(NotFoundError):
            resources.get("missing")


class TestCreate:
    @respx.mock
    def test_create_with_spec(self, resources):
        route = respx.post(_RESOURCES).mock(return_value=httpx.Response(201, json=_resource(1)))
        spec = {"tables": [{"name": "users", "columns": []}]}
        r = resources.create(type="SQL", tool_name="query_db", name="prod-db",
                             dialect="POSTGRES", spec=spec)
        assert isinstance(r, Resource) and r.resource_id == "res1"
        body = json.loads(route.calls.last.request.read())
        assert body["type"] == "SQL" and body["tool_name"] == "query_db"
        assert body["name"] == "prod-db" and body["dialect"] == "POSTGRES"
        assert body["spec"] == spec and "ddl" not in body
        # the org is taken from the key server-side — never sent from the client
        assert "org" not in body and "org_id" not in body

    @respx.mock
    def test_create_with_ddl_and_optionals(self, resources):
        route = respx.post(_RESOURCES).mock(return_value=httpx.Response(201, json=_resource(1)))
        resources.create(type="SQL", tool_name="query_db", name="db2", dialect="POSTGRES",
                        ddl="CREATE TABLE users (id INT);", description="d",
                        hydration_config={"default_rows_per_table": 50})
        body = json.loads(route.calls.last.request.read())
        assert body["ddl"] == "CREATE TABLE users (id INT);" and "spec" not in body
        assert body["description"] == "d"
        assert body["hydration_config"] == {"default_rows_per_table": 50}

    @respx.mock
    def test_create_duplicate_409(self, resources):
        respx.post(_RESOURCES).mock(return_value=httpx.Response(
            409, json={"error": "A resource named 'db2' already exists for this org."}))
        with pytest.raises(ConflictError):
            resources.create(type="SQL", tool_name="t", name="db2", dialect="POSTGRES", spec={})

    @respx.mock
    def test_create_xor_violation_400(self, resources):
        # Neither spec nor ddl — the SDK forwards and lets the backend enforce the XOR.
        respx.post(_RESOURCES).mock(return_value=httpx.Response(
            400, json={"error": "exactly one of ddl or spec is required on create"}))
        with pytest.raises(ValidationError):
            resources.create(type="SQL", tool_name="t", name="db3", dialect="POSTGRES")


class TestUpdate:
    @respx.mock
    def test_update_partial(self, resources):
        route = respx.put(f"{_RESOURCES}/res1").mock(return_value=httpx.Response(200, json=_resource(1)))
        r = resources.update("res1", name="renamed", hydration_config={"max_result_rows": 5000})
        assert isinstance(r, Resource)
        body = json.loads(route.calls.last.request.read())
        assert body["name"] == "renamed" and body["hydration_config"] == {"max_result_rows": 5000}
        # type/dialect are immutable and intentionally absent from the update surface
        assert "type" not in body and "dialect" not in body
        assert all(k not in body for k in ("tool_name", "description", "spec", "ddl"))

    @respx.mock
    def test_update_ddl_reparse(self, resources):
        route = respx.put(f"{_RESOURCES}/res1").mock(return_value=httpx.Response(200, json=_resource(1)))
        resources.update("res1", ddl="CREATE TABLE t (id INT);")
        assert json.loads(route.calls.last.request.read())["ddl"] == "CREATE TABLE t (id INT);"

    @respx.mock
    def test_update_rename_collision_409(self, resources):
        respx.put(f"{_RESOURCES}/res1").mock(return_value=httpx.Response(
            409, json={"error": "A resource named 'taken' already exists for this org."}))
        with pytest.raises(ConflictError):
            resources.update("res1", name="taken")


class TestDelete:
    @respx.mock
    def test_delete_204(self, resources):
        route = respx.delete(f"{_RESOURCES}/res1").mock(return_value=httpx.Response(204))
        assert resources.delete("res1") is None
        assert route.called

    @respx.mock
    def test_delete_in_use_409(self, resources):
        respx.delete(f"{_RESOURCES}/res1").mock(return_value=httpx.Response(
            409, json={"error": "detach it from those Datasets first", "dataset_ids": ["d1"]}))
        with pytest.raises(ConflictError):
            resources.delete("res1")

    @respx.mock
    def test_delete_404(self, resources):
        respx.delete(f"{_RESOURCES}/missing").mock(return_value=httpx.Response(
            404, json={"error": "Specified Resource not found"}))
        with pytest.raises(NotFoundError):
            resources.delete("missing")


class TestAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_alist(self, resources):
        respx.get(_RESOURCES).mock(side_effect=[
            httpx.Response(200, json={"results": [_resource(1)], "next": f"{_RESOURCES}?cursor=c2"}),
            httpx.Response(200, json={"results": [_resource(2)], "next": None}),
        ])
        got = [r.resource_id async for r in resources.alist()]
        assert got == ["res1", "res2"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_acreate(self, resources):
        respx.post(_RESOURCES).mock(return_value=httpx.Response(201, json=_resource(1)))
        r = await resources.acreate(type="SQL", tool_name="t", name="n", dialect="POSTGRES", spec={})
        assert r.resource_id == "res1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_aupdate(self, resources):
        respx.put(f"{_RESOURCES}/res1").mock(return_value=httpx.Response(200, json=_resource(1)))
        r = await resources.aupdate("res1", description="d")
        assert r.resource_id == "res1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_adelete(self, resources):
        route = respx.delete(f"{_RESOURCES}/res1").mock(return_value=httpx.Response(204))
        assert await resources.adelete("res1") is None
        assert route.called
