"""LUC-918 — client.datasets v2: items + schemas (CRUD) + fixtures.create.
LUC-921 generation; LUC-1195 eval runs (start_run / runs), case evaluators, the
fallback evaluator and the cases list."""
import json

import httpx
import pytest
import respx

from lucidicai.api.models.dataset import (
    CaseEvaluatorUpdate,
    CaseGrader,
    DatasetCase,
    DatasetGenerationRun,
    DatasetItem,
    DatasetSchema,
    EvalRun,
    Fixture,
)
from lucidicai.api.models.evaluator import Evaluator
from lucidicai.api.resources.dataset import (
    DatasetResource,
    DatasetFixturesResource,
    DatasetSchemasResource,
)
from lucidicai.core.errors import (
    ConflictError,
    InsufficientScopeError,
    InvalidOperationError,
    NotFoundError,
    RateLimitError,
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


# ==================== LUC-1195 eval runs ====================

_DS = "11111111-1111-1111-1111-111111111111"
_RUNNER = "https://runner.example.test/hook"


def _runs_url(dataset_id=_DS):
    return f"{_BASE}/sdk/v2/datasets/{dataset_id}/runs"


def _eval_run(i=1, **over):
    # EvalRunStart.as_dict (eval_run_services.py) — the 202 body
    d = {
        "eval_run_id": f"er{i}", "experiment_id": f"ex{i}", "evosim_id": f"ev{i}",
        "tmi_id": f"tmi{i}", "dataset_id": _DS, "world_id": None, "role": "baseline",
        "repeats": 3, "checkpoint_id": None, "window": 10, "repeat_deadline_seconds": 1800,
        "case_count": 4, "session_count": 12, "status": "running",
    }
    d.update(over)
    return d


def _run_row(i=1, **over):
    # run_rows: the start shape plus created_at and the compact summary
    d = _eval_run(i, created_at="2026-10-25T00:00:00Z", summary=None)
    d.update(over)
    return d


_SUMMARY = {
    "metric": "primary_pass_rate", "k": 3,
    "per_repeat": [{"repeat": 1, "pass_rate": 0.5}],
    "pass_rate_spread": 0.0,
    "buckets": {"always": {"count": 2, "share": 0.5}},
}


class TestEvalRunModel:
    @pytest.mark.parametrize("status,terminal,complete", [
        ("running", False, False),
        ("grading", False, False),
        ("complete", True, True),
        ("failed", True, False),
        ("canceled", True, False),
        (None, False, False),
    ])
    def test_status_flags(self, status, terminal, complete):
        run = EvalRun.from_dict(_eval_run(status=status))
        assert run.is_terminal is terminal and run.is_complete is complete

    def test_start_body_parses(self):
        run = EvalRun.from_dict(_eval_run())
        assert run.eval_run_id == "er1" and run.evosim_id == "ev1" and run.experiment_id == "ex1"
        assert run.dataset_id == _DS and run.world_id is None
        assert run.window == 10 and run.repeat_deadline_seconds == 1800
        assert run.case_count == 4 and run.session_count == 12
        assert run.created_at is None and run.summary is None

    def test_list_row_parses_and_keeps_unknown_keys(self):
        run = EvalRun.from_dict(_run_row(status="complete", summary=_SUMMARY, graded_by_new=1))
        assert run.created_at == "2026-10-25T00:00:00Z" and run.summary == _SUMMARY
        assert run.extra == {"graded_by_new": 1}


class TestStartRun:
    @respx.mock
    def test_minimal_body_is_omit_none(self, datasets):
        route = respx.post(_runs_url()).mock(return_value=httpx.Response(202, json=_eval_run()))
        run = datasets.start_run(_DS, webhook_url=_RUNNER)
        assert isinstance(run, EvalRun) and run.eval_run_id == "er1" and run.status == "running"
        body = json.loads(route.calls.last.request.read())
        body.pop("current_time")  # the transport stamps every POST
        assert body == {"webhook_url": _RUNNER}

    @respx.mock
    def test_every_option_sent_by_name(self, datasets):
        route = respx.post(_runs_url()).mock(return_value=httpx.Response(202, json=_eval_run()))
        datasets.start_run(
            _DS, webhook_url=_RUNNER, repeats=2, role="candidate", checkpoint_id="cp1",
            item_ids=["i3", "i1"], window=2, repeat_deadline_seconds=600,
        )
        body = json.loads(route.calls.last.request.read())
        body.pop("current_time")
        assert body == {
            "webhook_url": _RUNNER, "repeats": 2, "role": "candidate", "checkpoint_id": "cp1",
            "item_ids": ["i3", "i1"], "window": 2, "repeat_deadline_seconds": 600,
        }

    @respx.mock
    @pytest.mark.parametrize("given,sent", [
        ("i1", ["i1"]),  # one id, not its characters
        (("i1", "i2"), ["i1", "i2"]),
        ((i for i in ["i1", "i2"]), ["i1", "i2"]),
        ([], []),  # sent, so the backend refuses it, never "every case"
    ])
    def test_item_ids_one_or_many(self, datasets, given, sent):
        route = respx.post(_runs_url()).mock(return_value=httpx.Response(202, json=_eval_run()))
        datasets.start_run(_DS, webhook_url=_RUNNER, item_ids=given)
        assert json.loads(route.calls.last.request.read())["item_ids"] == sent

    @respx.mock
    def test_out_of_range_400(self, datasets):
        respx.post(_runs_url()).mock(return_value=httpx.Response(400, json={
            "error": "Validation failed",
            "details": {"window": ["Ensure this value is less than or equal to 50."]}}))
        with pytest.raises(ValidationError) as exc:
            datasets.start_run(_DS, webhook_url=_RUNNER, window=51)
        assert exc.value.details == {"window": ["Ensure this value is less than or equal to 50."]}

    @respx.mock
    def test_service_refusal_400(self, datasets):
        respx.post(_runs_url()).mock(return_value=httpx.Response(
            400, json={"error": "200 cases x 10 repeats is over 1000 sessions per run"}))
        with pytest.raises(ValidationError, match="over 1000 sessions"):
            datasets.start_run(_DS, webhook_url=_RUNNER, repeats=10)

    @respx.mock
    def test_missing_scope_403(self, datasets):
        respx.post(_runs_url()).mock(return_value=httpx.Response(403, json={
            "error": "API key lacks the required scope", "required_scope": "evosim:run"}))
        with pytest.raises(InsufficientScopeError) as exc:
            datasets.start_run(_DS, webhook_url=_RUNNER)
        assert exc.value.required_scope == "evosim:run"

    @respx.mock
    def test_unknown_dataset_404(self, datasets):
        respx.post(_runs_url("missing")).mock(return_value=httpx.Response(
            404, json={"error": "Specified Dataset not found"}))
        with pytest.raises(NotFoundError):
            datasets.start_run("missing", webhook_url=_RUNNER)

    @respx.mock
    def test_throttled_429_is_not_retried(self, datasets):
        route = respx.post(_runs_url()).mock(return_value=httpx.Response(
            429, json={"error": "Request was throttled. Expected available in 7 seconds."},
            headers={"Retry-After": "7"}))
        with pytest.raises(RateLimitError) as exc:
            datasets.start_run(_DS, webhook_url=_RUNNER)
        assert route.call_count == 1 and exc.value.retry_after == 7

    @respx.mock
    def test_start_failed_503_is_not_retried_and_keeps_the_ids(self, datasets):
        message = ("Workflow service is temporarily unavailable and the run may have started; "
                   "cancel it by evosim_id before retrying.")
        route = respx.post(_runs_url()).mock(return_value=httpx.Response(
            503, json={**_eval_run(), "status": "start_failed", "error": message}))
        with pytest.raises(ServiceUnavailableError, match="cancel it by evosim_id") as exc:
            datasets.start_run(_DS, webhook_url=_RUNNER)
        assert route.call_count == 1  # a retry could start a second run
        assert exc.value.response_body["evosim_id"] == "ev1"
        assert exc.value.response_body["status"] == "start_failed"


class TestRuns:
    @respx.mock
    def test_follows_pages_typed_with_filters(self, datasets):
        url = _runs_url()
        route = respx.get(url).mock(side_effect=[
            httpx.Response(200, json={"results": [_run_row(1, status="complete", summary=_SUMMARY),
                                                  _run_row(2)],
                                      "next": f"{url}?cursor=c2&role=baseline"}),
            httpx.Response(200, json={"results": [_run_row(3)], "next": None}),
        ])
        got = list(datasets.runs(_DS, role="baseline", status="complete", page_size=2))
        assert [r.eval_run_id for r in got] == ["er1", "er2", "er3"]
        assert all(isinstance(r, EvalRun) for r in got)
        assert got[0].summary == _SUMMARY and got[1].summary is None
        first, second = (c.request.url.params for c in route.calls)
        assert dict(first) == {"role": "baseline", "status": "complete", "page_size": "2"}
        assert dict(second) == {"role": "baseline", "status": "complete", "page_size": "2", "cursor": "c2"}

    @respx.mock
    def test_no_filters_sends_no_params(self, datasets):
        route = respx.get(_runs_url()).mock(
            return_value=httpx.Response(200, json={"results": [], "next": None}))
        assert list(datasets.runs(_DS)) == []
        assert dict(route.calls.last.request.url.params) == {}

    @respx.mock
    def test_runs_page(self, datasets):
        url = _runs_url()
        route = respx.get(url).mock(return_value=httpx.Response(
            200, json={"results": [_run_row(1)], "next": f"{url}?cursor=c2", "previous": None}))
        page = datasets.runs_page(_DS, cursor="c1", role="candidate")
        assert isinstance(page.results[0], EvalRun) and page.next_cursor == "c2"
        assert dict(route.calls.last.request.url.params) == {"role": "candidate", "cursor": "c1"}

    @respx.mock
    def test_bad_role_400(self, datasets):
        respx.get(_runs_url()).mock(return_value=httpx.Response(
            400, json={"error": "role must be one of baseline, candidate"}))
        with pytest.raises(ValidationError, match="role must be one of"):
            list(datasets.runs(_DS, role="champion"))

    @respx.mock
    def test_unknown_dataset_404(self, datasets):
        respx.get(_runs_url("missing")).mock(return_value=httpx.Response(
            404, json={"error": "Specified Dataset not found"}))
        with pytest.raises(NotFoundError):
            list(datasets.runs("missing"))


class TestEvalRunsAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_astart_run(self, datasets):
        route = respx.post(_runs_url()).mock(return_value=httpx.Response(202, json=_eval_run()))
        run = await datasets.astart_run(_DS, webhook_url=_RUNNER, item_ids="i1", repeats=1)
        assert isinstance(run, EvalRun) and run.eval_run_id == "er1"
        body = json.loads(route.calls.last.request.read())
        assert body["item_ids"] == ["i1"] and body["repeats"] == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_astart_run_start_failed_503_is_not_retried(self, datasets):
        route = respx.post(_runs_url()).mock(return_value=httpx.Response(
            503, json={**_eval_run(), "status": "start_failed",
                       "error": "Workflow service is temporarily unavailable; the run did not start. Please retry."}))
        with pytest.raises(ServiceUnavailableError) as exc:
            await datasets.astart_run(_DS, webhook_url=_RUNNER)
        assert route.call_count == 1  # a retry could start a second run
        assert exc.value.response_body["evosim_id"] == "ev1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_aruns_follows_pages(self, datasets):
        url = _runs_url()
        respx.get(url).mock(side_effect=[
            httpx.Response(200, json={"results": [_run_row(1)], "next": f"{url}?cursor=c2"}),
            httpx.Response(200, json={"results": [_run_row(2)], "next": None}),
        ])
        got = [r.eval_run_id async for r in datasets.aruns(_DS)]
        assert got == ["er1", "er2"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_aruns_page(self, datasets):
        route = respx.get(_runs_url()).mock(return_value=httpx.Response(
            200, json={"results": [_run_row(1)], "next": None}))
        page = await datasets.aruns_page(_DS, status="complete")
        assert isinstance(page.results[0], EvalRun) and page.has_next is False
        assert dict(route.calls.last.request.url.params) == {"status": "complete"}


# ==================== LUC-1195 case evaluators (over LUC-1190) ====================

_CASES_URL = f"{_BASE}/sdk/v2/datasets/{_DS}/case-evaluator"
_EV = "22222222-2222-2222-2222-222222222222"
_BOOL_CRIT = [{"name": "cites the clause", "pass_definition": "names the clause number",
               "fail_definition": "does not name it"}]
_NUM_CRIT = [{"name": "accuracy", "score_definitions": [{"score": 0, "definition": "wrong"},
                                                         {"score": 1, "definition": "right"}]}]


def _case_url(item_id):
    return f"{_BASE}/sdk/v2/datasets/{_DS}/items/{item_id}/case-evaluator"


def _update(**over):
    # DatasetCaseEvaluator.put (dataset_views.py, LUC-1190)
    d = {"evaluator_id": _EV, "created": False, "item_count": 1, "changed": 1}
    d.update(over)
    return d


def _case_ev(**over):
    # _case_evaluator_body: EvaluatorDetailSerializer + datasetitem_id
    d = {"evaluator_id": _EV, "name": "cites-clause", "description": "", "icon": "compass",
         "type": "LLM", "result_type": "boolean", "is_default": False, "is_primary": False,
         "created_at": "2026-10-25T00:00:00Z", "updated_at": "2026-10-25T00:00:00Z",
         "criteria": _BOOL_CRIT, "rubric_json": {"criteria": _BOOL_CRIT}, "datasetitem_id": "i1"}
    d.update(over)
    return d


# CaseEvaluatorBulkSerializer's item_ids (allow_empty=False), in the backend's envelope
_EMPTY_IDS_400 = {"error": "Validation failed", "details": {"item_ids": ["This list may not be empty."]}}


def _put_body(route):
    body = json.loads(route.calls.last.request.read())
    body.pop("current_time")  # the transport stamps every PUT
    return body


class TestCaseEvaluatorModels:
    def test_item_carries_case_evaluator_id(self):
        assert DatasetItem.from_dict(_item(1, case_evaluator_id=_EV)).case_evaluator_id == _EV
        assert DatasetItem.from_dict(_item(1)).case_evaluator_id is None

    def test_evaluator_datasetitem_id_only_on_case_reads(self):
        assert Evaluator.from_dict(_case_ev()).datasetitem_id == "i1"
        plain = _case_ev()
        plain.pop("datasetitem_id")
        assert Evaluator.from_dict(plain).datasetitem_id is None

    def test_update_parses_a_clear(self):
        u = CaseEvaluatorUpdate.from_dict(_update(evaluator_id=None, item_count=3, changed=2))
        assert u.evaluator_id is None and u.created is False and (u.item_count, u.changed) == (3, 2)


class TestSetCaseEvaluator:
    @respx.mock
    def test_by_id_for_one_case(self, datasets):
        route = respx.put(_CASES_URL).mock(return_value=httpx.Response(200, json=_update()))
        u = datasets.set_case_evaluator(_DS, "i1", evaluator_id=_EV)
        assert isinstance(u, CaseEvaluatorUpdate) and u.evaluator_id == _EV and u.changed == 1
        assert _put_body(route) == {"item_ids": ["i1"], "evaluator_id": _EV}

    @respx.mock
    def test_by_id_for_many_cases(self, datasets):
        route = respx.put(_CASES_URL).mock(return_value=httpx.Response(
            200, json=_update(item_count=3, changed=3)))
        u = datasets.set_case_evaluator(_DS, ("i1", "i2", "i3"), evaluator_id=_EV)
        assert u.item_count == 3
        assert _put_body(route)["item_ids"] == ["i1", "i2", "i3"]

    @respx.mock
    def test_boolean_spec_is_omit_none(self, datasets):
        route = respx.put(_CASES_URL).mock(return_value=httpx.Response(201, json=_update(created=True)))
        u = datasets.set_case_evaluator(_DS, ["i1"], name="cites-clause", criteria=_BOOL_CRIT,
                                        result_type="boolean")
        assert u.created is True
        assert _put_body(route) == {"item_ids": ["i1"], "evaluator": {
            "name": "cites-clause", "criteria": _BOOL_CRIT, "result_type": "boolean"}}

    @respx.mock
    def test_number_spec_with_pass_mark(self, datasets):
        route = respx.put(_CASES_URL).mock(return_value=httpx.Response(201, json=_update(created=True)))
        datasets.set_case_evaluator(_DS, ["i1", "i2"], name="accuracy", criteria=_NUM_CRIT,
                                    result_type="number", description="d", pass_threshold=0.7)
        assert _put_body(route)["evaluator"] == {
            "name": "accuracy", "criteria": _NUM_CRIT, "result_type": "number",
            "description": "d", "pass_threshold": 0.7}

    @respx.mock
    @pytest.mark.parametrize("kwargs,match", [
        ({"evaluator_id": _EV, "name": "x"}, "not both"),
        ({"evaluator_id": _EV, "pass_threshold": 0.5}, "not both"),
        ({}, "pass evaluator_id, or name"),
        ({"name": "x", "criteria": _BOOL_CRIT}, "missing result_type"),
        ({"pass_threshold": 0.5}, "missing name, criteria, result_type"),
        ({"evaluator_id": _EV, "pass_threshold": 0}, "not both"),  # 0 is a mark, not "none"
        ({"description": ""}, "missing name"),
    ])
    def test_bad_mode_raises_before_any_request(self, datasets, kwargs, match):
        route = respx.put(_CASES_URL).mock(return_value=httpx.Response(200, json=_update()))
        ids = (i for i in ["i1", "i2"])
        with pytest.raises(InvalidOperationError, match=match):
            datasets.set_case_evaluator(_DS, ids, **kwargs)
        assert not route.called
        assert list(ids) == ["i1", "i2"]  # the caller's generator is untouched

    @respx.mock
    def test_falsy_spec_values_are_sent(self, datasets):
        route = respx.put(_CASES_URL).mock(return_value=httpx.Response(201, json=_update(created=True)))
        datasets.set_case_evaluator(_DS, "i1", name="accuracy", criteria=_NUM_CRIT, result_type="number",
                                    description="", pass_threshold=0)
        spec = _put_body(route)["evaluator"]
        assert spec["pass_threshold"] == 0 and spec["description"] == ""

    @respx.mock
    def test_empty_item_ids_are_sent(self, datasets):
        # sent as [], so the backend refuses it (allow_empty=False) rather than the SDK guessing
        route = respx.put(_CASES_URL).mock(return_value=httpx.Response(400, json=_EMPTY_IDS_400))
        with pytest.raises(ValidationError):
            datasets.set_case_evaluator(_DS, [], evaluator_id=_EV)
        assert _put_body(route)["item_ids"] == []

    @respx.mock
    @pytest.mark.parametrize("call", [
        lambda ds: ds.set_case_evaluator(_DS, None, evaluator_id=_EV),
        lambda ds: ds.remove_case_evaluator(_DS, None),
    ])
    def test_none_item_ids_raise_before_any_request(self, datasets, call):
        route = respx.put(_CASES_URL).mock(return_value=httpx.Response(200, json=_update()))
        with pytest.raises(InvalidOperationError, match="item_ids: pass one id or a list"):
            call(datasets)
        assert not route.called

    @respx.mock
    def test_ids_not_in_the_dataset_400(self, datasets):
        respx.put(_CASES_URL).mock(return_value=httpx.Response(
            400, json={"error": "Not cases of this Dataset: i9"}))
        with pytest.raises(ValidationError, match="Not cases of this Dataset"):
            datasets.set_case_evaluator(_DS, "i9", evaluator_id=_EV)

    @respx.mock
    def test_malformed_spec_400_with_details(self, datasets):
        details = {"evaluator": {"pass_threshold": ["a finite number is required for a number evaluator."]}}
        respx.put(_CASES_URL).mock(return_value=httpx.Response(
            400, json={"error": "Validation failed", "details": details}))
        with pytest.raises(ValidationError) as exc:
            datasets.set_case_evaluator(_DS, "i1", name="accuracy", criteria=_NUM_CRIT,
                                        result_type="number")
        assert exc.value.details == details

    @respx.mock
    def test_unknown_evaluator_404(self, datasets):
        respx.put(_CASES_URL).mock(return_value=httpx.Response(
            404, json={"error": "Specified Evaluator not found"}))
        with pytest.raises(NotFoundError):
            datasets.set_case_evaluator(_DS, "i1", evaluator_id="missing")

    @respx.mock
    def test_name_clash_409(self, datasets):
        respx.put(_CASES_URL).mock(return_value=httpx.Response(409, json={
            "error": "An evaluator named 'cites-clause' already exists for this agent with a different rubric"}))
        with pytest.raises(ConflictError, match="different rubric"):
            datasets.set_case_evaluator(_DS, "i1", name="cites-clause", criteria=_BOOL_CRIT,
                                        result_type="boolean")

    @respx.mock
    def test_put_is_retried_and_a_retry_reads_reused(self, datasets):
        # the PUT is idempotent: the retry reuses the evaluator the first try made
        route = respx.put(_CASES_URL).mock(side_effect=[
            httpx.Response(503, json={"error": "unavailable"}),
            httpx.Response(200, json=_update(created=False, changed=0)),
        ])
        u = datasets.set_case_evaluator(_DS, "i1", name="cites-clause", criteria=_BOOL_CRIT,
                                        result_type="boolean")
        assert route.call_count == 2 and u.created is False


class TestCaseEvaluatorRead:
    @respx.mock
    def test_get_one_case(self, datasets):
        respx.get(_case_url("i1")).mock(return_value=httpx.Response(200, json=_case_ev()))
        ev = datasets.case_evaluator(_DS, "i1")
        assert isinstance(ev, Evaluator) and ev.evaluator_id == _EV and ev.datasetitem_id == "i1"
        assert ev.criteria == _BOOL_CRIT and ev.rubric_json == {"criteria": _BOOL_CRIT}

    @respx.mock
    def test_none_set_404(self, datasets):
        respx.get(_case_url("i2")).mock(return_value=httpx.Response(
            404, json={"error": "Specified case evaluator not found"}))
        with pytest.raises(NotFoundError):
            datasets.case_evaluator(_DS, "i2")


class TestRemoveCaseEvaluator:
    @respx.mock
    @pytest.mark.parametrize("given,sent", [("i1", ["i1"]), (["i1", "i2"], ["i1", "i2"])])
    def test_sends_a_null_evaluator_id(self, datasets, given, sent):
        route = respx.put(_CASES_URL).mock(return_value=httpx.Response(
            200, json=_update(evaluator_id=None, item_count=len(sent), changed=1)))
        u = datasets.remove_case_evaluator(_DS, given)
        assert u.evaluator_id is None and u.changed == 1
        assert _put_body(route) == {"item_ids": sent, "evaluator_id": None}

    @respx.mock
    def test_empty_item_ids_are_sent(self, datasets):
        route = respx.put(_CASES_URL).mock(return_value=httpx.Response(400, json=_EMPTY_IDS_400))
        with pytest.raises(ValidationError):
            datasets.remove_case_evaluator(_DS, [])
        assert _put_body(route) == {"item_ids": [], "evaluator_id": None}

    @respx.mock
    def test_unknown_dataset_404(self, datasets):
        respx.put(f"{_BASE}/sdk/v2/datasets/missing/case-evaluator").mock(
            return_value=httpx.Response(404, json={"error": "Specified Dataset not found"}))
        with pytest.raises(NotFoundError):
            datasets.remove_case_evaluator("missing", "i1")


class TestCaseEvaluatorsAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_aset_case_evaluator(self, datasets):
        route = respx.put(_CASES_URL).mock(return_value=httpx.Response(201, json=_update(created=True)))
        u = await datasets.aset_case_evaluator(_DS, "i1", name="cites-clause", criteria=_BOOL_CRIT,
                                               result_type="boolean")
        assert u.created is True
        assert _put_body(route)["evaluator"]["name"] == "cites-clause"

    @respx.mock
    @pytest.mark.asyncio
    async def test_aset_case_evaluator_checks_the_mode(self, datasets):
        route = respx.put(_CASES_URL).mock(return_value=httpx.Response(200, json=_update()))
        with pytest.raises(InvalidOperationError):
            await datasets.aset_case_evaluator(_DS, "i1")
        assert not route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_acase_evaluator(self, datasets):
        respx.get(_case_url("i1")).mock(return_value=httpx.Response(200, json=_case_ev()))
        ev = await datasets.acase_evaluator(_DS, "i1")
        assert ev.datasetitem_id == "i1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_aremove_case_evaluator(self, datasets):
        route = respx.put(_CASES_URL).mock(return_value=httpx.Response(
            200, json=_update(evaluator_id=None, changed=0)))
        u = await datasets.aremove_case_evaluator(_DS, "i1")
        assert u.changed == 0 and _put_body(route) == {"item_ids": ["i1"], "evaluator_id": None}


# ==================== LUC-1195 fallback evaluator (over LUC-1191) ====================

_FALLBACK_URL = f"{_BASE}/sdk/v2/datasets/{_DS}/fallback-evaluator"
_RUN_IN_PROGRESS = {"error": f"Dataset {_DS} has a run in progress; change its fallback evaluator after it finishes."}


def _fallback_ev(**over):
    # _fallback_body: EvaluatorDetailSerializer + dataset_id (the PUT adds created)
    d = _case_ev()
    d.pop("datasetitem_id")
    d.update(dataset_id=_DS)
    d.update(over)
    return d


def _evaluator_plain():
    d = _fallback_ev()
    d.pop("dataset_id")
    return d


class TestFallbackEvaluatorModel:
    def test_fallback_fields_only_on_fallback_bodies(self):
        ev = Evaluator.from_dict(_fallback_ev(created=True))
        assert ev.dataset_id == _DS and ev.created is True and ev.datasetitem_id is None
        plain = Evaluator.from_dict(_evaluator_plain())
        assert plain.dataset_id is None and plain.created is None


class TestSetFallbackEvaluator:
    @respx.mock
    def test_by_id(self, datasets):
        route = respx.put(_FALLBACK_URL).mock(return_value=httpx.Response(
            200, json=_fallback_ev(created=False)))
        ev = datasets.set_fallback_evaluator(_DS, _EV)
        assert isinstance(ev, Evaluator) and ev.evaluator_id == _EV
        assert ev.dataset_id == _DS and ev.created is False
        assert _put_body(route) == {"evaluator_id": _EV}

    @respx.mock
    def test_by_number_spec(self, datasets):
        route = respx.put(_FALLBACK_URL).mock(return_value=httpx.Response(
            201, json=_fallback_ev(result_type="number", created=True)))
        ev = datasets.set_fallback_evaluator(_DS, name="answers-the-question", criteria=_NUM_CRIT,
                                             result_type="number", pass_threshold=0.7)
        assert ev.created is True
        assert _put_body(route) == {"evaluator": {
            "name": "answers-the-question", "criteria": _NUM_CRIT, "result_type": "number",
            "pass_threshold": 0.7}}

    @respx.mock
    @pytest.mark.parametrize("args,kwargs,match", [
        ((), {}, "pass evaluator_id, or name"),  # a bare call never clears
        ((_EV,), {"name": "x"}, "not both"),
        ((), {"criteria": _BOOL_CRIT, "result_type": "boolean"}, "missing name"),
        ((_EV,), {"pass_threshold": 0}, "not both"),
        ((), {"description": ""}, "missing name"),
    ])
    def test_bad_mode_raises_before_any_request(self, datasets, args, kwargs, match):
        put = respx.put(_FALLBACK_URL).mock(return_value=httpx.Response(200, json=_fallback_ev()))
        delete = respx.delete(_FALLBACK_URL).mock(return_value=httpx.Response(204))
        with pytest.raises(InvalidOperationError, match=match):
            datasets.set_fallback_evaluator(_DS, *args, **kwargs)
        assert not put.called and not delete.called

    @respx.mock
    def test_not_gradable_400(self, datasets):
        respx.put(_FALLBACK_URL).mock(return_value=httpx.Response(
            400, json={"error": "The agent's pass/fail metric can't also be a fallback evaluator."}))
        with pytest.raises(ValidationError, match="fallback evaluator"):
            datasets.set_fallback_evaluator(_DS, _EV)

    @respx.mock
    def test_unknown_evaluator_404(self, datasets):
        respx.put(_FALLBACK_URL).mock(return_value=httpx.Response(
            404, json={"error": "Specified Evaluator not found"}))
        with pytest.raises(NotFoundError):
            datasets.set_fallback_evaluator(_DS, "missing")

    @respx.mock
    def test_run_in_progress_409(self, datasets):
        respx.put(_FALLBACK_URL).mock(return_value=httpx.Response(409, json=_RUN_IN_PROGRESS))
        with pytest.raises(ConflictError, match="run in progress"):
            datasets.set_fallback_evaluator(_DS, _EV)


class TestFallbackEvaluatorRead:
    @respx.mock
    def test_get(self, datasets):
        respx.get(_FALLBACK_URL).mock(return_value=httpx.Response(200, json=_fallback_ev()))
        ev = datasets.fallback_evaluator(_DS)
        assert isinstance(ev, Evaluator) and ev.dataset_id == _DS and ev.created is None

    @respx.mock
    def test_unset_404(self, datasets):
        respx.get(_FALLBACK_URL).mock(return_value=httpx.Response(
            404, json={"error": "Specified fallback evaluator not found"}))
        with pytest.raises(NotFoundError):
            datasets.fallback_evaluator(_DS)

    @respx.mock
    def test_dataset_dict_carries_fallback_evaluator_id(self, datasets):
        # no typed Dataset model: the gen-3 get() dict (DatasetSerializer) carries the id
        route = respx.get(f"{_BASE}/getdataset").mock(return_value=httpx.Response(
            200, json={"dataset_id": _DS, "name": "harvey", "fallback_evaluator_id": _EV}))
        assert datasets.get(_DS)["fallback_evaluator_id"] == _EV
        assert route.calls.last.request.url.params["dataset_id"] == _DS


class TestRemoveFallbackEvaluator:
    @respx.mock
    def test_delete_204(self, datasets):
        route = respx.delete(_FALLBACK_URL).mock(return_value=httpx.Response(204))
        assert datasets.remove_fallback_evaluator(_DS) is None
        assert route.called

    @respx.mock
    def test_run_in_progress_409(self, datasets):
        respx.delete(_FALLBACK_URL).mock(return_value=httpx.Response(409, json=_RUN_IN_PROGRESS))
        with pytest.raises(ConflictError, match="run in progress"):
            datasets.remove_fallback_evaluator(_DS)

    @respx.mock
    def test_unknown_dataset_404(self, datasets):
        respx.delete(f"{_BASE}/sdk/v2/datasets/missing/fallback-evaluator").mock(
            return_value=httpx.Response(404, json={"error": "Specified Dataset not found"}))
        with pytest.raises(NotFoundError):
            datasets.remove_fallback_evaluator("missing")


class TestCaseEvaluatorRunInProgress:
    # LUC-1190's review fix: case-evaluator writes answer 409 while a run is in progress
    @respx.mock
    def test_set_409(self, datasets):
        respx.put(_CASES_URL).mock(return_value=httpx.Response(409, json={
            "error": f"Dataset {_DS} has a run in progress; change case evaluators after it finishes."}))
        with pytest.raises(ConflictError, match="run in progress"):
            datasets.set_case_evaluator(_DS, "i1", evaluator_id=_EV)

    @respx.mock
    def test_remove_409(self, datasets):
        respx.put(_CASES_URL).mock(return_value=httpx.Response(409, json={
            "error": f"Dataset {_DS} has a run in progress; change case evaluators after it finishes."}))
        with pytest.raises(ConflictError, match="run in progress"):
            datasets.remove_case_evaluator(_DS, "i1")


class TestFallbackEvaluatorAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_aset_fallback_evaluator(self, datasets):
        route = respx.put(_FALLBACK_URL).mock(return_value=httpx.Response(
            201, json=_fallback_ev(created=True)))
        ev = await datasets.aset_fallback_evaluator(_DS, name="cites-clause", criteria=_BOOL_CRIT,
                                                    result_type="boolean")
        assert ev.created is True and _put_body(route)["evaluator"]["name"] == "cites-clause"

    @respx.mock
    @pytest.mark.asyncio
    async def test_aset_fallback_evaluator_checks_the_mode(self, datasets):
        route = respx.put(_FALLBACK_URL).mock(return_value=httpx.Response(200, json=_fallback_ev()))
        with pytest.raises(InvalidOperationError):
            await datasets.aset_fallback_evaluator(_DS)
        assert not route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_afallback_evaluator(self, datasets):
        respx.get(_FALLBACK_URL).mock(return_value=httpx.Response(200, json=_fallback_ev()))
        assert (await datasets.afallback_evaluator(_DS)).dataset_id == _DS

    @respx.mock
    @pytest.mark.asyncio
    async def test_aremove_fallback_evaluator(self, datasets):
        route = respx.delete(_FALLBACK_URL).mock(return_value=httpx.Response(204))
        assert await datasets.aremove_fallback_evaluator(_DS) is None
        assert route.called


# ==================== LUC-1195 cases list (over LUC-1194) ====================

_FB = "33333333-3333-3333-3333-333333333333"
_METRIC = "44444444-4444-4444-4444-444444444444"


def _cases_url(dataset_id=_DS):
    return f"{_BASE}/sdk/v2/datasets/{dataset_id}/cases"


def _grader(evaluator_id=_EV, **over):
    # case_evaluator_services._grader
    d = {"evaluator_id": evaluator_id, "name": "cites-clause", "type": "LLM", "result_type": "boolean",
         "scope": "case", "graded_by": "case_evaluator", "pass_threshold": None, "criteria": _BOOL_CRIT}
    d.update(over)
    return d


def _case_row(i, **over):
    # DatasetCaseSerializer: the item fields + dataset_cases' four keys
    d = {**_item(i, case_evaluator_id=None), "scored": True, "graded_by": None,
         "grader": None, "if_not_reported": None}
    d.update(over)
    return d


_BY_CASE = dict(case_evaluator_id=_EV, graded_by="case_evaluator", grader=_grader())
_BY_AGENT = dict(
    graded_by="agent_reported",
    grader=_grader(_METRIC, name="task_passed", type="RUNTIME", scope="agent",
                   graded_by="agent_reported", criteria=None),
    if_not_reported=_grader(_FB, name="answers-the-question", result_type="number", scope="dataset",
                            graded_by="fallback", pass_threshold=0.7, criteria=_NUM_CRIT),
)


class TestDatasetCaseModel:
    def test_case_evaluator_row(self):
        case = DatasetCase.from_dict(_case_row(1, **_BY_CASE))
        assert isinstance(case, DatasetItem)  # a case is an item plus how it's graded
        assert case.datasetitem_id == "it1" and case.input == {"q": "hi"}
        assert case.case_evaluator_id == _EV and case.scored is True
        assert case.graded_by == "case_evaluator"
        assert isinstance(case.grader, CaseGrader) and case.grader.scope == "case"
        assert case.grader.criteria == _BOOL_CRIT and case.if_not_reported is None

    def test_agent_reported_row_with_a_fallback(self):
        case = DatasetCase.from_dict(_case_row(2, **_BY_AGENT))
        assert case.graded_by == "agent_reported" and case.grader.evaluator_id == _METRIC
        assert case.grader.criteria is None
        fb = case.if_not_reported
        assert isinstance(fb, CaseGrader) and fb.evaluator_id == _FB
        assert (fb.scope, fb.graded_by, fb.pass_threshold) == ("dataset", "fallback", 0.7)

    def test_ungraded_row(self):
        case = DatasetCase.from_dict(_case_row(3))
        assert case.scored is True and case.graded_by is None
        assert case.grader is None and case.if_not_reported is None

    def test_world_entry_item_is_not_scored(self):
        assert DatasetCase.from_dict(_case_row(4, scored=False)).scored is False

    def test_criteria_hidden_without_evaluator_read(self):
        case = DatasetCase.from_dict(_case_row(5, **{**_BY_CASE, "grader": _grader(criteria=None)}))
        assert case.grader.criteria is None and case.grader.name == "cites-clause"


class TestCases:
    @respx.mock
    def test_follows_pages_typed(self, datasets):
        url = _cases_url()
        route = respx.get(url).mock(side_effect=[
            httpx.Response(200, json={"results": [_case_row(1, **_BY_CASE), _case_row(2, **_BY_AGENT)],
                                      "next": f"{url}?cursor=c2"}),
            httpx.Response(200, json={"results": [_case_row(3)], "next": None}),
        ])
        got = list(datasets.cases(_DS, page_size=2))
        assert [c.datasetitem_id for c in got] == ["it1", "it2", "it3"]
        assert all(isinstance(c, DatasetCase) for c in got)
        assert [c.graded_by for c in got] == ["case_evaluator", "agent_reported", None]
        first, second = (dict(c.request.url.params) for c in route.calls)
        assert first == {"page_size": "2"} and second == {"page_size": "2", "cursor": "c2"}

    @respx.mock
    def test_no_page_size_sends_no_params(self, datasets):
        route = respx.get(_cases_url()).mock(
            return_value=httpx.Response(200, json={"results": [], "next": None}))
        assert list(datasets.cases(_DS)) == []
        assert dict(route.calls.last.request.url.params) == {}

    @respx.mock
    def test_cases_page(self, datasets):
        url = _cases_url()
        route = respx.get(url).mock(return_value=httpx.Response(
            200, json={"results": [_case_row(1, **_BY_CASE)], "next": f"{url}?cursor=c2", "previous": None}))
        page = datasets.cases_page(_DS, cursor="c1", page_size=50)
        assert isinstance(page.results[0], DatasetCase) and page.next_cursor == "c2"
        assert isinstance(page.results[0].grader, CaseGrader)
        assert dict(route.calls.last.request.url.params) == {"page_size": "50", "cursor": "c1"}

    @respx.mock
    def test_unknown_dataset_404(self, datasets):
        respx.get(_cases_url("missing")).mock(return_value=httpx.Response(
            404, json={"error": "Specified Dataset not found"}))
        with pytest.raises(NotFoundError):
            list(datasets.cases("missing"))


class TestCasesAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_acases_follows_pages(self, datasets):
        url = _cases_url()
        respx.get(url).mock(side_effect=[
            httpx.Response(200, json={"results": [_case_row(1, **_BY_CASE)], "next": f"{url}?cursor=c2"}),
            httpx.Response(200, json={"results": [_case_row(2, **_BY_AGENT)], "next": None}),
        ])
        got = [c async for c in datasets.acases(_DS)]
        assert [c.datasetitem_id for c in got] == ["it1", "it2"]
        assert isinstance(got[1].if_not_reported, CaseGrader)

    @respx.mock
    @pytest.mark.asyncio
    async def test_acases_page(self, datasets):
        route = respx.get(_cases_url()).mock(return_value=httpx.Response(
            200, json={"results": [_case_row(1)], "next": None}))
        page = await datasets.acases_page(_DS, page_size=10)
        assert isinstance(page.results[0], DatasetCase) and page.has_next is False
        assert dict(route.calls.last.request.url.params) == {"page_size": "10"}
