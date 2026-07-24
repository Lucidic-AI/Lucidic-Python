"""LUC-908 — client.experiments reads (list / count / get).
LUC-915 — client.experiments writes (delete + evaluators attach/detach/list).
LUC-922 — client.experiments taxonomy + failure-modes (async, trigger-then-poll)."""
import json

import httpx
import pytest
import respx

from lucidicai.api.models.evaluator import Evaluator
from lucidicai.api.models.experiment import Experiment, FailureGroup
from lucidicai.api.models.taxonomy import TaxonomyRun, TaxonomyStatus
from lucidicai.api.resources.experiment import ExperimentResource, ExperimentTaxonomyResource
from lucidicai.core.errors import (
    ConflictError,
    InsufficientScopeError,
    NotFoundError,
    ServiceUnavailableError,
    ValidationError,
    WaitTimeout,
)

_BASE = "https://stub.lucidic.test"
_EXPERIMENTS = f"{_BASE}/sdk/v2/experiments"


def _evals_url(xid):
    return f"{_EXPERIMENTS}/{xid}/evaluators"


def _tax_url(xid):
    return f"{_EXPERIMENTS}/{xid}/taxonomy"


def _tax_run(**over):
    d = {"run_id": "tr1", "version": 1, "status": "queued", "created_at": "2026-07-22T00:00:00Z"}
    d.update(over)
    return d


def _tax_status(**over):
    d = {"ongoing": None, "latest_completed": None, "evaluating_session_count": 0}
    d.update(over)
    return d


def _tax_completed(**over):
    # The status endpoint's latest_completed carries NO status field (unlike a run) —
    # {run_id, version, created_at, completed_at}. Keep the fixture faithful.
    d = {"run_id": "tr1", "version": 1, "created_at": "2026-07-22T00:00:00Z",
         "completed_at": "2026-07-22T01:00:00Z"}
    d.update(over)
    return d


def _fgroup(i, **over):
    d = {"id": f"fg{i}", "group_name": f"group-{i}", "group_description": "",
         "icon": "warn", "events": [{"event_id": "e1", "session_id": "s1"}]}
    d.update(over)
    return d


@pytest.fixture
def experiments(http):
    # Configured agent_id "a1" — list()/count() default to it.
    return ExperimentResource(http, agent_id="a1", production=False)


def _exp(i, **over):
    d = {
        "experiment_id": f"x{i}", "name": f"exp-{i}", "description": "",
        "created_at": "2026-07-22T00:00:00Z", "updated_at": "2026-07-22T00:00:00Z",
        "num_sessions": 3, "tags": ["prod"], "eval_metrics": [],
    }
    d.update(over)
    return d


def _ev(i, **over):
    d = {"evaluator_id": f"e{i}", "name": f"eval-{i}", "type": "LLM",
         "result_type": "boolean", "is_default": False}
    d.update(over)
    return d


class TestList:
    @respx.mock
    def test_follows_pages_typed(self, experiments):
        respx.get(_EXPERIMENTS).mock(side_effect=[
            httpx.Response(200, json={"results": [_exp(1), _exp(2)],
                                      "next": f"{_EXPERIMENTS}?cursor=c2", "previous": None}),
            httpx.Response(200, json={"results": [_exp(3)], "next": None}),
        ])
        got = list(experiments.list())
        assert [e.experiment_id for e in got] == ["x1", "x2", "x3"]
        assert all(isinstance(e, Experiment) for e in got)

    @respx.mock
    def test_defaults_configured_agent_id(self, experiments):
        route = respx.get(_EXPERIMENTS).mock(
            return_value=httpx.Response(200, json={"results": [], "next": None}))
        list(experiments.list())
        assert route.calls.last.request.url.params["agent_id"] == "a1"

    @respx.mock
    def test_explicit_agent_id_and_ordering(self, experiments):
        route = respx.get(_EXPERIMENTS).mock(
            return_value=httpx.Response(200, json={"results": [], "next": None}))
        list(experiments.list("a2", ordering="created_at", page_size=10))
        p = route.calls.last.request.url.params
        assert p["agent_id"] == "a2"
        assert p["ordering"] == "created_at"
        assert p["page_size"] == "10"

    @respx.mock
    def test_list_page(self, experiments):
        respx.get(_EXPERIMENTS).mock(return_value=httpx.Response(
            200, json={"results": [_exp(1)], "next": f"{_EXPERIMENTS}?cursor=c9"}))
        page = experiments.list_page()
        assert isinstance(page.results[0], Experiment) and page.next_cursor == "c9"


class TestCount:
    @respx.mock
    def test_count(self, experiments):
        respx.head(_EXPERIMENTS).mock(return_value=httpx.Response(
            200, headers={"X-Total-Count": "12"}))
        assert experiments.count() == 12

    @respx.mock
    def test_count_missing_header_zero(self, experiments):
        respx.head(_EXPERIMENTS).mock(return_value=httpx.Response(200))
        assert experiments.count() == 0


class TestGet:
    @respx.mock
    def test_detail_with_metrics(self, experiments):
        detail = _exp(1, agent_id="a1", eval_metrics_by_tag={"prod": {}},
                      time_data={"p50": 1.2}, event_failure_groups=[{"group": "timeout"}],
                      analytics_session_count=3)
        respx.get(f"{_EXPERIMENTS}/x1").mock(return_value=httpx.Response(200, json=detail))
        e = experiments.get("x1")
        assert isinstance(e, Experiment)
        assert e.agent_id == "a1"
        assert e.time_data == {"p50": 1.2}
        assert e.event_failure_groups == [{"group": "timeout"}]

    @respx.mock
    def test_404_raises(self, experiments):
        respx.get(f"{_EXPERIMENTS}/missing").mock(
            return_value=httpx.Response(404, json={"error": "not found"}))
        with pytest.raises(NotFoundError):
            experiments.get("missing")


class TestAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_alist(self, experiments):
        respx.get(_EXPERIMENTS).mock(side_effect=[
            httpx.Response(200, json={"results": [_exp(1)], "next": f"{_EXPERIMENTS}?cursor=c2"}),
            httpx.Response(200, json={"results": [_exp(2)], "next": None}),
        ])
        got = [e.experiment_id async for e in experiments.alist()]
        assert got == ["x1", "x2"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_acount(self, experiments):
        respx.head(_EXPERIMENTS).mock(return_value=httpx.Response(200, headers={"X-Total-Count": "5"}))
        assert await experiments.acount() == 5

    @respx.mock
    @pytest.mark.asyncio
    async def test_aget(self, experiments):
        respx.get(f"{_EXPERIMENTS}/x1").mock(return_value=httpx.Response(200, json=_exp(1)))
        e = await experiments.aget("x1")
        assert e.experiment_id == "x1"


# ==================== LUC-915 writes ====================


class TestDelete:
    @respx.mock
    def test_delete_default_no_cascade(self, experiments):
        # delete_sessions is a JSON BODY flag, default False (not a query param).
        route = respx.delete(f"{_EXPERIMENTS}/x1").mock(return_value=httpx.Response(204))
        assert experiments.delete("x1") is None
        assert json.loads(route.calls.last.request.read()) == {"delete_sessions": False}

    @respx.mock
    def test_delete_sessions_true_sends_body_flag(self, experiments):
        route = respx.delete(f"{_EXPERIMENTS}/x1").mock(return_value=httpx.Response(204))
        experiments.delete("x1", delete_sessions=True)
        assert json.loads(route.calls.last.request.read()) == {"delete_sessions": True}

    @respx.mock
    def test_delete_forbidden_403(self, experiments):
        # experiment:delete is a full-key-only scope; an ingest key → 403.
        respx.delete(f"{_EXPERIMENTS}/x1").mock(return_value=httpx.Response(
            403, json={"error": "This API key is not authorized for this action."}))
        with pytest.raises(InsufficientScopeError):
            experiments.delete("x1")

    @respx.mock
    def test_delete_missing_404(self, experiments):
        respx.delete(f"{_EXPERIMENTS}/missing").mock(return_value=httpx.Response(
            404, json={"error": "Specified Experiment not found"}))
        with pytest.raises(NotFoundError):
            experiments.delete("missing")


class TestEvaluatorsList:
    @respx.mock
    def test_follows_pages_typed(self, experiments):
        url = _evals_url("x1")
        respx.get(url).mock(side_effect=[
            httpx.Response(200, json={"results": [_ev(1), _ev(2)], "next": f"{url}?cursor=c2"}),
            httpx.Response(200, json={"results": [_ev(3)], "next": None}),
        ])
        got = list(experiments.evaluators.list("x1"))
        assert [e.evaluator_id for e in got] == ["e1", "e2", "e3"]
        assert all(isinstance(e, Evaluator) for e in got)

    @respx.mock
    def test_list_page_and_page_size(self, experiments):
        route = respx.get(_evals_url("x1")).mock(return_value=httpx.Response(
            200, json={"results": [_ev(1)], "next": None}))
        page = experiments.evaluators.list_page("x1", page_size=50)
        assert isinstance(page.results[0], Evaluator)
        assert route.calls.last.request.url.params["page_size"] == "50"


class TestEvaluatorsAttach:
    @respx.mock
    def test_attach_posts_names_returns_full_set(self, experiments):
        route = respx.post(_evals_url("x1")).mock(return_value=httpx.Response(
            200, json={"experiment_id": "x1", "evaluators": [_ev(1), _ev(2)]}))
        got = experiments.evaluators.attach("x1", ["eval-1", "eval-2"])
        assert [e.evaluator_id for e in got] == ["e1", "e2"]
        assert all(isinstance(e, Evaluator) for e in got)
        # backend field is evaluator_names (names, not ids); current_time is auto-injected.
        assert json.loads(route.calls.last.request.read())["evaluator_names"] == ["eval-1", "eval-2"]

    @respx.mock
    def test_attach_unknown_name_400(self, experiments):
        respx.post(_evals_url("x1")).mock(return_value=httpx.Response(
            400, json={"error": "Validation failed",
                       "details": {"evaluator_names": ["Unknown evaluators for this experiment's agent: ['nope']"]}}))
        with pytest.raises(ValidationError):
            experiments.evaluators.attach("x1", ["nope"])


class TestEvaluatorsDetach:
    @respx.mock
    def test_detach_returns_none_204(self, experiments):
        route = respx.delete(f"{_evals_url('x1')}/e1").mock(return_value=httpx.Response(204))
        assert experiments.evaluators.detach("x1", "e1") is None
        assert route.called

    @respx.mock
    def test_detach_not_attached_404(self, experiments):
        respx.delete(f"{_evals_url('x1')}/e9").mock(return_value=httpx.Response(
            404, json={"error": "Specified evaluator not found"}))
        with pytest.raises(NotFoundError):
            experiments.evaluators.detach("x1", "e9")


class TestWriteAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_adelete_sends_body(self, experiments):
        route = respx.delete(f"{_EXPERIMENTS}/x1").mock(return_value=httpx.Response(204))
        assert await experiments.adelete("x1", delete_sessions=True) is None
        assert json.loads(route.calls.last.request.read()) == {"delete_sessions": True}

    @respx.mock
    @pytest.mark.asyncio
    async def test_aattach(self, experiments):
        respx.post(_evals_url("x1")).mock(return_value=httpx.Response(
            200, json={"experiment_id": "x1", "evaluators": [_ev(1)]}))
        got = await experiments.evaluators.aattach("x1", ["eval-1"])
        assert [e.evaluator_id for e in got] == ["e1"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_adetach(self, experiments):
        route = respx.delete(f"{_evals_url('x1')}/e1").mock(return_value=httpx.Response(204))
        assert await experiments.evaluators.adetach("x1", "e1") is None
        assert route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_alist(self, experiments):
        url = _evals_url("x1")
        respx.get(url).mock(side_effect=[
            httpx.Response(200, json={"results": [_ev(1)], "next": f"{url}?cursor=c2"}),
            httpx.Response(200, json={"results": [_ev(2)], "next": None}),
        ])
        got = [e.evaluator_id async for e in experiments.evaluators.alist("x1")]
        assert got == ["e1", "e2"]


# ==================== LUC-922 taxonomy + failure-modes ====================


class TestWiring:
    def test_taxonomy_subnamespace_resolves(self, experiments):
        assert isinstance(experiments.taxonomy, ExperimentTaxonomyResource)


class TestTaxonomyModel:
    def test_status_types_nested_runs_and_is_terminal(self):
        # ongoing present -> not terminal, nested runs typed
        st = TaxonomyStatus.from_dict(_tax_status(
            ongoing=_tax_run(status="sampling"), latest_completed=_tax_completed(run_id="tr0", version=0)))
        assert st.is_terminal is False
        assert isinstance(st.ongoing, TaxonomyRun) and st.ongoing.status == "sampling"
        assert isinstance(st.latest_completed, TaxonomyRun) and st.latest_completed.run_id == "tr0"

    def test_status_no_ongoing_is_terminal(self):
        st = TaxonomyStatus.from_dict(_tax_status(latest_completed=_tax_completed()))
        assert st.is_terminal is True and st.ongoing is None
        assert isinstance(st.latest_completed, TaxonomyRun)


class TestTaxonomyGenerate:
    @respx.mock
    def test_generate_empty_body_and_typed(self, experiments):
        route = respx.post(_tax_url("x1")).mock(return_value=httpx.Response(201, json=_tax_run()))
        run = experiments.taxonomy.generate("x1")
        assert isinstance(run, TaxonomyRun) and run.run_id == "tr1" and run.status == "queued"
        body = json.loads(route.calls.last.request.read())
        assert "seed_dimensions" not in body and "use_previous" not in body  # omit-None

    @respx.mock
    def test_generate_with_seeds_and_use_previous(self, experiments):
        route = respx.post(_tax_url("x1")).mock(return_value=httpx.Response(201, json=_tax_run()))
        seeds = [{"name": "tone", "description": "d"}]
        experiments.taxonomy.generate("x1", seed_dimensions=seeds, use_previous=True)
        body = json.loads(route.calls.last.request.read())
        assert body["seed_dimensions"] == seeds and body["use_previous"] is True

    @respx.mock
    def test_generate_too_few_sessions_400(self, experiments):
        respx.post(_tax_url("x1")).mock(return_value=httpx.Response(
            400, json={"error": "Need at least 20 finished sessions, found 3"}))
        with pytest.raises(ValidationError):
            experiments.taxonomy.generate("x1")

    @respx.mock
    def test_generate_already_running_409(self, experiments):
        respx.post(_tax_url("x1")).mock(return_value=httpx.Response(
            409, json={"error": "A taxonomy run is already in progress", "ongoing_run_id": "tr9"}))
        with pytest.raises(ConflictError):
            experiments.taxonomy.generate("x1")

    @respx.mock
    def test_generate_503(self, experiments):
        respx.post(_tax_url("x1")).mock(return_value=httpx.Response(
            503, json={"error": "Workflow service is temporarily unavailable; please retry."}))
        with pytest.raises(ServiceUnavailableError):
            experiments.taxonomy.generate("x1")


class TestTaxonomyGetStatus:
    @respx.mock
    def test_get_completed_taxonomy(self, experiments):
        respx.get(_tax_url("x1")).mock(return_value=httpx.Response(200, json=_tax_run(
            status=None, completed_at="2026-07-22T01:00:00Z", taxonomy={"dimensions": []})))
        run = experiments.taxonomy.get("x1")
        assert run.taxonomy == {"dimensions": []} and run.completed_at is not None

    @respx.mock
    def test_get_404_until_completed(self, experiments):
        respx.get(_tax_url("x1")).mock(return_value=httpx.Response(
            404, json={"error": "Specified CompletedTaxonomyRun not found"}))
        with pytest.raises(NotFoundError):
            experiments.taxonomy.get("x1")

    @respx.mock
    def test_status_typed(self, experiments):
        respx.get(f"{_tax_url('x1')}/status").mock(return_value=httpx.Response(
            200, json=_tax_status(ongoing=_tax_run(status="validating"), evaluating_session_count=4)))
        st = experiments.taxonomy.status("x1")
        assert isinstance(st, TaxonomyStatus) and st.evaluating_session_count == 4
        assert st.ongoing.status == "validating" and st.is_terminal is False


class TestTaxonomyWaitFor:
    @respx.mock
    def test_polls_until_no_ongoing(self, experiments):
        respx.get(f"{_tax_url('x1')}/status").mock(side_effect=[
            httpx.Response(200, json=_tax_status(ongoing=_tax_run(status="queued"))),
            httpx.Response(200, json=_tax_status(ongoing=_tax_run(status="validating"))),
            httpx.Response(200, json=_tax_status(latest_completed=_tax_completed(run_id="tr1"))),
        ])
        st = experiments.taxonomy.wait_for("x1", timeout=30, interval=0)
        assert st.is_terminal is True and st.ongoing is None
        # latest_completed carries no status field from the backend — match on run_id
        assert st.latest_completed.run_id == "tr1" and st.latest_completed.status is None

    @respx.mock
    def test_timeout_raises(self, experiments):
        respx.get(f"{_tax_url('x1')}/status").mock(
            return_value=httpx.Response(200, json=_tax_status(ongoing=_tax_run(status="sampling"))))
        with pytest.raises(WaitTimeout) as exc:
            experiments.taxonomy.wait_for("x1", timeout=0, interval=0)
        assert exc.value.last_state.ongoing.status == "sampling"


class TestFailureModes:
    @respx.mock
    def test_generate_failure_modes_returns_none(self, experiments):
        route = respx.post(f"{_EXPERIMENTS}/x1/failure-modes").mock(
            return_value=httpx.Response(202, json={"experiment_id": "x1", "status": "started"}))
        assert experiments.generate_failure_modes("x1") is None
        assert route.called

    @respx.mock
    def test_generate_failure_modes_no_sessions_400(self, experiments):
        respx.post(f"{_EXPERIMENTS}/x1/failure-modes").mock(return_value=httpx.Response(
            400, json={"error": "No sessions found in experiment"}))
        with pytest.raises(ValidationError):
            experiments.generate_failure_modes("x1")

    @respx.mock
    def test_generate_failure_modes_503(self, experiments):
        respx.post(f"{_EXPERIMENTS}/x1/failure-modes").mock(return_value=httpx.Response(
            503, json={"error": "Workflow service is temporarily unavailable; please retry."}))
        with pytest.raises(ServiceUnavailableError):
            experiments.generate_failure_modes("x1")

    @respx.mock
    def test_failure_groups_unwraps_typed_list(self, experiments):
        respx.get(f"{_EXPERIMENTS}/x1/failure-groups").mock(return_value=httpx.Response(
            200, json={"event_failure_groups": [_fgroup(1), _fgroup(2)]}))
        groups = experiments.failure_groups("x1")
        assert [g.id for g in groups] == ["fg1", "fg2"]
        assert all(isinstance(g, FailureGroup) for g in groups)
        assert groups[0].events == [{"event_id": "e1", "session_id": "s1"}]

    @respx.mock
    def test_failure_groups_empty(self, experiments):
        respx.get(f"{_EXPERIMENTS}/x1/failure-groups").mock(
            return_value=httpx.Response(200, json={"event_failure_groups": []}))
        assert experiments.failure_groups("x1") == []


class TestTaxonomyFailureAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_agenerate(self, experiments):
        respx.post(_tax_url("x1")).mock(return_value=httpx.Response(201, json=_tax_run()))
        run = await experiments.taxonomy.agenerate("x1")
        assert run.run_id == "tr1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_await_for(self, experiments):
        respx.get(f"{_tax_url('x1')}/status").mock(side_effect=[
            httpx.Response(200, json=_tax_status(ongoing=_tax_run(status="sampling"))),
            httpx.Response(200, json=_tax_status(latest_completed=_tax_completed())),
        ])
        st = await experiments.taxonomy.await_for("x1", timeout=30, interval=0)
        assert st.is_terminal is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_afailure_groups(self, experiments):
        respx.get(f"{_EXPERIMENTS}/x1/failure-groups").mock(return_value=httpx.Response(
            200, json={"event_failure_groups": [_fgroup(1)]}))
        groups = await experiments.afailure_groups("x1")
        assert groups[0].id == "fg1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_agenerate_failure_modes(self, experiments):
        route = respx.post(f"{_EXPERIMENTS}/x1/failure-modes").mock(
            return_value=httpx.Response(202, json={"experiment_id": "x1", "status": "started"}))
        assert await experiments.agenerate_failure_modes("x1") is None
        assert route.called
