"""LUC-918 — client.datasets v2: items + schemas (CRUD) + fixtures.create."""
import json

import httpx
import pytest
import respx

from lucidicai.api.models.dataset import (
    DatasetGenerationRun,
    DatasetItem,
    DatasetSchema,
    Fixture,
)
from lucidicai.api.resources.dataset import (
    DatasetResource,
    DatasetFixturesResource,
    DatasetSchemasResource,
)
from lucidicai.core.errors import (
    ConflictError,
    NotFoundError,
    ServiceUnavailableError,
    ValidationError,
    WaitTimeout,
)

_BASE = "https://stub.lucidic.test"
_SCHEMAS = f"{_BASE}/sdk/v2/datasets/schemas"
_FIXTURES = f"{_BASE}/sdk/v2/fixtures"
_GENERATE = f"{_BASE}/sdk/v2/datasets/generate"


def _status_url(run_id):
    return f"{_GENERATE}/{run_id}/status"


def _gen_run(**over):
    d = {"run_id": "run1", "dataset_id": "ds1", "status": "queued",
         "created_at": "2026-07-22T00:00:00Z"}
    d.update(over)
    return d


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


# ==================== LUC-921 generation ====================


class TestGenerationModel:
    def test_terminal_and_succeeded_flags(self):
        assert DatasetGenerationRun.from_dict(_gen_run(status="queued")).is_terminal is False
        assert DatasetGenerationRun.from_dict(_gen_run(status="generating")).is_terminal is False
        done = DatasetGenerationRun.from_dict(_gen_run(status="completed"))
        assert done.is_terminal is True and done.succeeded is True
        failed = DatasetGenerationRun.from_dict(_gen_run(status="failed", error_message="boom"))
        assert failed.is_terminal is True and failed.succeeded is False
        assert failed.error_message == "boom"


class TestGenerate:
    @respx.mock
    def test_generate_required_fields_and_omit_none(self, datasets):
        route = respx.post(_GENERATE).mock(return_value=httpx.Response(201, json=_gen_run()))
        run = datasets.generate(agent_id="a1", experiment_id="e1", schema_id="s1",
                                dataset_name="gen-1")
        assert isinstance(run, DatasetGenerationRun)
        assert run.run_id == "run1" and run.dataset_id == "ds1" and run.status == "queued"
        body = json.loads(route.calls.last.request.read())
        assert body["agent_id"] == "a1" and body["experiment_id"] == "e1"
        assert body["schema_id"] == "s1" and body["dataset_name"] == "gen-1"
        # omit-None: unspecified optionals defer to the backend defaults
        assert not any(k in body for k in
                       ("dataset_description", "target_count", "combo_cap",
                        "selected_dimension_ids", "resource_ids"))

    @respx.mock
    def test_generate_all_options(self, datasets):
        route = respx.post(_GENERATE).mock(return_value=httpx.Response(201, json=_gen_run()))
        datasets.generate(agent_id="a1", experiment_id="e1", schema_id="s1", dataset_name="g",
                          dataset_description="d", target_count=100, combo_cap=20,
                          selected_dimension_ids=["dim1"], resource_ids=["r1"])
        body = json.loads(route.calls.last.request.read())
        assert body["target_count"] == 100 and body["combo_cap"] == 20
        assert body["selected_dimension_ids"] == ["dim1"] and body["resource_ids"] == ["r1"]
        assert body["dataset_description"] == "d"

    @respx.mock
    def test_generate_503_service_unavailable(self, datasets):
        respx.post(_GENERATE).mock(return_value=httpx.Response(
            503, json={"error": "Workflow service is temporarily unavailable; please retry."}))
        with pytest.raises(ServiceUnavailableError):
            datasets.generate(agent_id="a1", experiment_id="e1", schema_id="s1", dataset_name="g")

    @respx.mock
    def test_generate_no_taxonomy_400(self, datasets):
        respx.post(_GENERATE).mock(return_value=httpx.Response(
            400, json={"error": "Experiment has no completed taxonomy run"}))
        with pytest.raises(ValidationError):
            datasets.generate(agent_id="a1", experiment_id="e1", schema_id="s1", dataset_name="g")

    @respx.mock
    def test_generate_unknown_agent_404(self, datasets):
        respx.post(_GENERATE).mock(return_value=httpx.Response(
            404, json={"error": "Specified Agent not found"}))
        with pytest.raises(NotFoundError):
            datasets.generate(agent_id="missing", experiment_id="e1", schema_id="s1", dataset_name="g")


class TestGenerationStatus:
    @respx.mock
    def test_status_full_shape(self, datasets):
        respx.get(_status_url("run1")).mock(return_value=httpx.Response(200, json=_gen_run(
            status="generating", items_generated=12, total_target=50, dataset_name="g",
            experiment_id="e1", schema_id="s1", config={"target_count": 50})))
        run = datasets.generation_status("run1")
        assert run.status == "generating" and run.items_generated == 12 and run.total_target == 50
        assert run.is_terminal is False and run.config == {"target_count": 50}

    @respx.mock
    def test_status_404(self, datasets):
        respx.get(_status_url("missing")).mock(return_value=httpx.Response(
            404, json={"error": "Specified DatasetGenerationRun not found"}))
        with pytest.raises(NotFoundError):
            datasets.generation_status("missing")


class TestRetryGeneration:
    @respx.mock
    def test_retry_returns_new_run(self, datasets):
        route = respx.post(f"{_GENERATE}/run1/retry").mock(
            return_value=httpx.Response(201, json=_gen_run(run_id="run2", dataset_id="ds2")))
        run = datasets.retry_generation("run1")
        assert run.run_id == "run2" and run.dataset_id == "ds2"
        assert route.called

    @respx.mock
    def test_retry_non_failed_400(self, datasets):
        respx.post(f"{_GENERATE}/run1/retry").mock(return_value=httpx.Response(
            400, json={"error": "Only failed runs can be retried"}))
        with pytest.raises(ValidationError):
            datasets.retry_generation("run1")


class TestWaitForGeneration:
    @respx.mock
    def test_polls_until_completed(self, datasets):
        respx.get(_status_url("run1")).mock(side_effect=[
            httpx.Response(200, json=_gen_run(status="queued")),
            httpx.Response(200, json=_gen_run(status="generating", items_generated=5)),
            httpx.Response(200, json=_gen_run(status="completed", items_generated=50)),
        ])
        run = datasets.wait_for_generation("run1", timeout=30, interval=0)
        assert run.succeeded is True and run.status == "completed" and run.items_generated == 50

    @respx.mock
    def test_failed_run_is_returned_not_raised(self, datasets):
        respx.get(_status_url("run1")).mock(side_effect=[
            httpx.Response(200, json=_gen_run(status="generating")),
            httpx.Response(200, json=_gen_run(status="failed", error_message="llm error")),
        ])
        run = datasets.wait_for_generation("run1", timeout=30, interval=0)
        assert run.is_terminal is True and run.succeeded is False
        assert run.error_message == "llm error"

    @respx.mock
    def test_timeout_raises_wait_timeout(self, datasets):
        respx.get(_status_url("run1")).mock(
            return_value=httpx.Response(200, json=_gen_run(status="generating")))
        with pytest.raises(WaitTimeout) as exc:
            datasets.wait_for_generation("run1", timeout=0, interval=0)
        assert exc.value.last_state.status == "generating"


class TestGenerationAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_agenerate(self, datasets):
        respx.post(_GENERATE).mock(return_value=httpx.Response(201, json=_gen_run()))
        run = await datasets.agenerate(agent_id="a1", experiment_id="e1", schema_id="s1",
                                       dataset_name="g")
        assert run.run_id == "run1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_ageneration_status(self, datasets):
        respx.get(_status_url("run1")).mock(return_value=httpx.Response(
            200, json=_gen_run(status="completed")))
        run = await datasets.ageneration_status("run1")
        assert run.succeeded is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_await_for_generation(self, datasets):
        respx.get(_status_url("run1")).mock(side_effect=[
            httpx.Response(200, json=_gen_run(status="generating")),
            httpx.Response(200, json=_gen_run(status="completed")),
        ])
        run = await datasets.await_for_generation("run1", timeout=30, interval=0)
        assert run.succeeded is True
