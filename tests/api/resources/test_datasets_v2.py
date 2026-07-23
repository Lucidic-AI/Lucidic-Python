"""LUC-918 — client.datasets v2: items + schemas (CRUD) + fixtures.create."""
import json

import httpx
import pytest
import respx

from lucidicai.api.models.dataset import DatasetItem, DatasetSchema, Fixture
from lucidicai.api.resources.dataset import (
    DatasetResource,
    DatasetFixturesResource,
    DatasetSchemasResource,
)
from lucidicai.core.errors import ConflictError, NotFoundError, ValidationError

_BASE = "https://stub.lucidic.test"
_SCHEMAS = f"{_BASE}/sdk/v2/datasets/schemas"
_FIXTURES = f"{_BASE}/sdk/v2/fixtures"


def _items_url(dataset_id):
    return f"{_BASE}/sdk/v2/datasets/{dataset_id}/items"


@pytest.fixture
def datasets(http):
    return DatasetResource(http, agent_id="a1", production=False)


def _schema(i, **over):
    d = {
        "schema_id": f"s{i}", "name": f"schema-{i}", "description": "",
        "fields": [{"key": "q", "type": "string"}],
        "created_at": "2026-07-22T00:00:00Z", "updated_at": "2026-07-22T00:00:00Z",
    }
    d.update(over)
    return d


def _item(i, **over):
    d = {
        "datasetitem_id": f"it{i}", "name": f"item-{i}", "description": "",
        "tags": [], "input": {"q": "hi"}, "expected_output": "ok",
        "metadata": {}, "flag_overrides": None, "created_at": "2026-07-22T00:00:00Z",
    }
    d.update(over)
    return d


def _fixture(**over):
    d = {"fixture_id": "f1", "blob_key": "tenant=o/dataset=d/fixture=f1/data.duckdb",
         "byte_size": 2048, "status": "COMPLETED"}
    d.update(over)
    return d


class TestWiring:
    def test_subnamespaces_resolve(self, datasets):
        assert isinstance(datasets.schemas, DatasetSchemasResource)
        assert isinstance(datasets.fixtures, DatasetFixturesResource)


class TestItems:
    @respx.mock
    def test_follows_pages_typed(self, datasets):
        url = _items_url("d1")
        respx.get(url).mock(side_effect=[
            httpx.Response(200, json={"results": [_item(1), _item(2)], "next": f"{url}?cursor=c2"}),
            httpx.Response(200, json={"results": [_item(3)], "next": None}),
        ])
        got = list(datasets.items("d1"))
        assert [i.datasetitem_id for i in got] == ["it1", "it2", "it3"]
        assert all(isinstance(i, DatasetItem) for i in got)
        assert got[0].input == {"q": "hi"} and got[0].expected_output == "ok"

    @respx.mock
    def test_items_page_size(self, datasets):
        route = respx.get(_items_url("d1")).mock(
            return_value=httpx.Response(200, json={"results": [_item(1)], "next": None}))
        page = datasets.items_page("d1", page_size=25)
        assert isinstance(page.results[0], DatasetItem)
        assert route.calls.last.request.url.params["page_size"] == "25"

    @respx.mock
    def test_items_404(self, datasets):
        respx.get(_items_url("missing")).mock(
            return_value=httpx.Response(404, json={"error": "Specified Dataset not found"}))
        with pytest.raises(NotFoundError):
            list(datasets.items("missing"))


class TestSchemas:
    @respx.mock
    def test_list_typed_org_scoped(self, datasets):
        route = respx.get(_SCHEMAS).mock(side_effect=[
            httpx.Response(200, json={"results": [_schema(1), _schema(2)], "next": f"{_SCHEMAS}?cursor=c2"}),
            httpx.Response(200, json={"results": [_schema(3)], "next": None}),
        ])
        got = list(datasets.schemas.list(ordering="id"))
        assert [s.schema_id for s in got] == ["s1", "s2", "s3"]
        assert all(isinstance(s, DatasetSchema) for s in got)
        p = route.calls[0].request.url.params
        assert "agent_id" not in p and p["ordering"] == "id"

    @respx.mock
    def test_get(self, datasets):
        respx.get(f"{_SCHEMAS}/s1").mock(return_value=httpx.Response(200, json=_schema(1)))
        s = datasets.schemas.get("s1")
        assert isinstance(s, DatasetSchema) and s.fields == [{"key": "q", "type": "string"}]

    @respx.mock
    def test_create(self, datasets):
        route = respx.post(_SCHEMAS).mock(return_value=httpx.Response(201, json=_schema(1)))
        fields = [{"key": "question", "type": "string"}]
        s = datasets.schemas.create("qa-schema", fields=fields)
        assert isinstance(s, DatasetSchema) and s.schema_id == "s1"
        body = json.loads(route.calls.last.request.read())
        assert body["name"] == "qa-schema" and body["fields"] == fields
        assert "description" not in body  # omit-None

    @respx.mock
    def test_create_with_description(self, datasets):
        route = respx.post(_SCHEMAS).mock(return_value=httpx.Response(201, json=_schema(1)))
        datasets.schemas.create("s", fields=[{"key": "q", "type": "string"}], description="d")
        assert json.loads(route.calls.last.request.read())["description"] == "d"

    @respx.mock
    def test_create_duplicate_409(self, datasets):
        respx.post(_SCHEMAS).mock(return_value=httpx.Response(
            409, json={"error": "A schema named 's' already exists for this org."}))
        with pytest.raises(ConflictError):
            datasets.schemas.create("s", fields=[{"key": "q", "type": "string"}])

    @respx.mock
    def test_create_bad_field_400(self, datasets):
        respx.post(_SCHEMAS).mock(return_value=httpx.Response(
            400, json={"error": "Validation failed", "details": {"fields": ["categorical needs options"]}}))
        with pytest.raises(ValidationError):
            datasets.schemas.create("s", fields=[{"key": "c", "type": "categorical"}])

    @respx.mock
    def test_update_partial(self, datasets):
        route = respx.put(f"{_SCHEMAS}/s1").mock(return_value=httpx.Response(200, json=_schema(1)))
        s = datasets.schemas.update("s1", name="renamed")
        assert isinstance(s, DatasetSchema)
        body = json.loads(route.calls.last.request.read())
        assert body["name"] == "renamed" and "fields" not in body and "description" not in body

    @respx.mock
    def test_update_rename_collision_409(self, datasets):
        respx.put(f"{_SCHEMAS}/s1").mock(return_value=httpx.Response(
            409, json={"error": "A schema with this name already exists for this org."}))
        with pytest.raises(ConflictError):
            datasets.schemas.update("s1", name="taken")

    @respx.mock
    def test_delete_204(self, datasets):
        route = respx.delete(f"{_SCHEMAS}/s1").mock(return_value=httpx.Response(204))
        assert datasets.schemas.delete("s1") is None
        assert route.called

    @respx.mock
    def test_delete_404(self, datasets):
        respx.delete(f"{_SCHEMAS}/missing").mock(return_value=httpx.Response(
            404, json={"error": "Specified DatasetSchema not found"}))
        with pytest.raises(NotFoundError):
            datasets.schemas.delete("missing")


class TestFixtures:
    @respx.mock
    def test_create(self, datasets):
        route = respx.post(_FIXTURES).mock(return_value=httpx.Response(201, json=_fixture()))
        tables = [{"name": "users", "rows": [{"id": 1, "name": "a"}]}]
        f = datasets.fixtures.create(resource_id="r1", dataset_id="d1", tables=tables)
        assert isinstance(f, Fixture) and f.fixture_id == "f1"
        assert f.status == "COMPLETED" and f.byte_size == 2048
        body = json.loads(route.calls.last.request.read())
        assert body["resource_id"] == "r1" and body["dataset_id"] == "d1"
        assert body["tables"] == tables

    @respx.mock
    def test_create_duplicate_409(self, datasets):
        respx.post(_FIXTURES).mock(return_value=httpx.Response(
            409, json={"error": "A fixture already exists for this dataset and resource."}))
        with pytest.raises(ConflictError):
            datasets.fixtures.create(resource_id="r1", dataset_id="d1", tables=[])

    @respx.mock
    def test_create_validation_400(self, datasets):
        respx.post(_FIXTURES).mock(return_value=httpx.Response(
            400, json={"error": "Fixture validation failed", "details": {"users": ["unknown column"]}}))
        with pytest.raises(ValidationError):
            datasets.fixtures.create(resource_id="r1", dataset_id="d1",
                                     tables=[{"name": "users", "rows": [{"bad": 1}]}])

    @respx.mock
    def test_create_unknown_resource_404(self, datasets):
        respx.post(_FIXTURES).mock(return_value=httpx.Response(
            404, json={"error": "Specified Resource not found"}))
        with pytest.raises(NotFoundError):
            datasets.fixtures.create(resource_id="missing", dataset_id="d1", tables=[])


class TestAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_aitems(self, datasets):
        url = _items_url("d1")
        respx.get(url).mock(side_effect=[
            httpx.Response(200, json={"results": [_item(1)], "next": f"{url}?cursor=c2"}),
            httpx.Response(200, json={"results": [_item(2)], "next": None}),
        ])
        got = [i.datasetitem_id async for i in datasets.aitems("d1")]
        assert got == ["it1", "it2"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_aschema_create(self, datasets):
        respx.post(_SCHEMAS).mock(return_value=httpx.Response(201, json=_schema(1)))
        s = await datasets.schemas.acreate("s", fields=[{"key": "q", "type": "string"}])
        assert s.schema_id == "s1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_aschema_delete(self, datasets):
        route = respx.delete(f"{_SCHEMAS}/s1").mock(return_value=httpx.Response(204))
        assert await datasets.schemas.adelete("s1") is None
        assert route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_afixture_create(self, datasets):
        respx.post(_FIXTURES).mock(return_value=httpx.Response(201, json=_fixture()))
        f = await datasets.fixtures.acreate(resource_id="r1", dataset_id="d1", tables=[])
        assert f.fixture_id == "f1"
