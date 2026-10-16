"""LUC-923 — client.evosims run management (list / get / cancel / iteration_instances
/ wait_for). The gen-3 train() kickoff is covered elsewhere (agent-id guard test)."""
import httpx
import pytest
import respx

from lucidicai.api.models.evosim import EvoSim, EvoSimIteration, TrainingModuleInstance
from lucidicai.api.resources.evosim import EvoSimsResource
from lucidicai.core.errors import (
    AgentIdRequiredError,
    NotFoundError,
    ServiceUnavailableError,
    WaitTimeout,
)

_BASE = "https://stub.lucidic.test"
_EVOSIMS = f"{_BASE}/sdk/evosims"


@pytest.fixture
def evosims(http):
    return EvoSimsResource(http, agent_id="a1", production=False)


def _evosim(i, **over):
    d = {"evosim_id": f"es{i}", "agent_id": "a1", "experiment_id": "e1",
         "name": f"run-{i}", "description": "", "max_iterations": 10,
         "hard_stop_seconds_per_iteration": 1800, "max_session_concurrency": 10,
         "webhook_url": None, "temporal_workflow_id": "wf1", "temporal_run_id": "trun1",
         "created_at": "2026-07-22T00:00:00Z", "updated_at": "2026-07-22T00:00:00Z"}
    d.update(over)
    return d


def _iteration(**over):
    d = {"evosimiteration_id": "it1", "evosim_id": "es1", "iteration": 1,
         "status": "RUNNING", "failure_reason": None, "is_finished": False,
         "hard_stop": None, "created_at": "2026-07-22T00:00:00Z"}
    d.update(over)
    return d


def _detail(status=None, **over):
    d = {"evosim": _evosim(1), "iterations": [], "status": status,
         "failure_reason": None, "checkpoint_id": None}
    d.update(over)
    return d


def _tmi(i, **over):
    d = {"tmi_id": f"tmi{i}", "module_key": "filesystem", "module_artifact_key": None,
         "created_by": "u1", "status": "READY", "session_id": "s1",
         "created_at": "2026-07-22T00:00:00Z"}
    d.update(over)
    return d


class TestModel:
    def test_is_terminal_across_statuses(self):
        for s in (None, "RUNNING"):
            assert EvoSim.from_dict(_evosim(1, status=s)).is_terminal is False
        for s in ("SUCCEEDED", "PARTIAL_SUCCESS", "FAILED", "CANCELED"):
            assert EvoSim.from_dict(_evosim(1, status=s)).is_terminal is True

    def test_iterations_typed(self):
        run = EvoSim.from_dict(_evosim(1, iterations=[_iteration(), _iteration(evosimiteration_id="it2")]))
        assert all(isinstance(it, EvoSimIteration) for it in run.iterations)
        assert run.iterations[0].evosimiteration_id == "it1"


class TestList:
    @respx.mock
    def test_follows_pages_typed_and_defaults_agent(self, evosims):
        route = respx.get(_EVOSIMS).mock(side_effect=[
            httpx.Response(200, json={"results": [_evosim(1), _evosim(2)],
                                      "next": f"{_EVOSIMS}?cursor=c2"}),
            httpx.Response(200, json={"results": [_evosim(3)], "next": None}),
        ])
        got = list(evosims.list())
        assert [e.evosim_id for e in got] == ["es1", "es2", "es3"]
        assert all(isinstance(e, EvoSim) for e in got)
        assert route.calls[0].request.url.params["agent_id"] == "a1"

    @respx.mock
    def test_explicit_agent_and_page_size(self, evosims):
        route = respx.get(_EVOSIMS).mock(return_value=httpx.Response(200, json={"results": [], "next": None}))
        list(evosims.list("a2", page_size=25))
        p = route.calls.last.request.url.params
        assert p["agent_id"] == "a2" and p["page_size"] == "25"

    def test_list_without_agent_id_raises(self, http):
        res = EvoSimsResource(http, agent_id=None)
        with pytest.raises(AgentIdRequiredError):
            list(res.list())


class TestGet:
    @respx.mock
    def test_get_flattens_detail_envelope(self, evosims):
        respx.get(f"{_EVOSIMS}/es1").mock(return_value=httpx.Response(200, json=_detail(
            status="SUCCEEDED", evosim=_evosim(1), iterations=[_iteration(status="SUCCEEDED")],
            checkpoint_id="ck1", failure_reason=None)))
        run = evosims.get("es1")
        assert isinstance(run, EvoSim) and run.evosim_id == "es1"
        assert run.status == "SUCCEEDED" and run.checkpoint_id == "ck1" and run.is_terminal is True
        assert run.iterations[0].evosimiteration_id == "it1"

    @respx.mock
    def test_get_running_status_none_checkpoint(self, evosims):
        respx.get(f"{_EVOSIMS}/es1").mock(return_value=httpx.Response(200, json=_detail(status="RUNNING")))
        run = evosims.get("es1")
        assert run.status == "RUNNING" and run.is_terminal is False and run.checkpoint_id is None

    @respx.mock
    def test_get_404(self, evosims):
        respx.get(f"{_EVOSIMS}/missing").mock(return_value=httpx.Response(
            404, json={"error": "Specified EvoSim not found"}))
        with pytest.raises(NotFoundError):
            evosims.get("missing")


class TestCancel:
    @respx.mock
    def test_cancel_returns_effective_status(self, evosims):
        route = respx.post(f"{_EVOSIMS}/es1/cancel").mock(
            return_value=httpx.Response(200, json={"evosim_id": "es1", "status": "CANCELED"}))
        assert evosims.cancel("es1") == "CANCELED"
        assert route.called

    @respx.mock
    def test_cancel_already_terminal_returns_its_status(self, evosims):
        # idempotent — cancelling a finished run reports its real terminal status
        respx.post(f"{_EVOSIMS}/es1/cancel").mock(
            return_value=httpx.Response(200, json={"evosim_id": "es1", "status": "SUCCEEDED"}))
        assert evosims.cancel("es1") == "SUCCEEDED"

    @respx.mock
    def test_cancel_503(self, evosims):
        respx.post(f"{_EVOSIMS}/es1/cancel").mock(return_value=httpx.Response(
            503, json={"error": "Workflow service is temporarily unavailable; please retry."}))
        with pytest.raises(ServiceUnavailableError):
            evosims.cancel("es1")


class TestIterationInstances:
    @respx.mock
    def test_unwraps_typed_list(self, evosims):
        respx.get(f"{_BASE}/sdk/evosim-iterations/it1/instances").mock(return_value=httpx.Response(
            200, json={"evosim_iteration_id": "it1", "training_module_instances": [_tmi(1), _tmi(2)]}))
        tmis = evosims.iteration_instances("it1")
        assert [t.tmi_id for t in tmis] == ["tmi1", "tmi2"]
        assert all(isinstance(t, TrainingModuleInstance) for t in tmis)
        assert tmis[0].module_key == "filesystem" and tmis[0].status == "READY"

    @respx.mock
    def test_empty(self, evosims):
        respx.get(f"{_BASE}/sdk/evosim-iterations/it1/instances").mock(
            return_value=httpx.Response(200, json={"evosim_iteration_id": "it1", "training_module_instances": []}))
        assert evosims.iteration_instances("it1") == []


class TestWaitFor:
    @respx.mock
    def test_polls_past_none_and_running_to_terminal(self, evosims):
        respx.get(f"{_EVOSIMS}/es1").mock(side_effect=[
            httpx.Response(200, json=_detail(status=None)),      # iteration not yet materialized
            httpx.Response(200, json=_detail(status="RUNNING")),
            httpx.Response(200, json=_detail(status="SUCCEEDED", checkpoint_id="ck1")),
        ])
        run = evosims.wait_for("es1", timeout=60, interval=0)
        assert run.is_terminal is True and run.status == "SUCCEEDED" and run.checkpoint_id == "ck1"

    @respx.mock
    def test_canceled_is_terminal(self, evosims):
        respx.get(f"{_EVOSIMS}/es1").mock(side_effect=[
            httpx.Response(200, json=_detail(status="RUNNING")),
            httpx.Response(200, json=_detail(status="CANCELED")),
        ])
        run = evosims.wait_for("es1", timeout=60, interval=0)
        assert run.status == "CANCELED" and run.is_terminal is True

    @respx.mock
    def test_timeout_raises(self, evosims):
        respx.get(f"{_EVOSIMS}/es1").mock(return_value=httpx.Response(200, json=_detail(status="RUNNING")))
        with pytest.raises(WaitTimeout) as exc:
            evosims.wait_for("es1", timeout=0, interval=0)
        assert exc.value.last_state.status == "RUNNING"


class TestAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_alist(self, evosims):
        respx.get(_EVOSIMS).mock(side_effect=[
            httpx.Response(200, json={"results": [_evosim(1)], "next": f"{_EVOSIMS}?cursor=c2"}),
            httpx.Response(200, json={"results": [_evosim(2)], "next": None}),
        ])
        got = [e.evosim_id async for e in evosims.alist()]
        assert got == ["es1", "es2"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_aget(self, evosims):
        respx.get(f"{_EVOSIMS}/es1").mock(return_value=httpx.Response(200, json=_detail(status="FAILED")))
        run = await evosims.aget("es1")
        assert run.status == "FAILED" and run.is_terminal is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_acancel(self, evosims):
        respx.post(f"{_EVOSIMS}/es1/cancel").mock(
            return_value=httpx.Response(200, json={"evosim_id": "es1", "status": "CANCELED"}))
        assert await evosims.acancel("es1") == "CANCELED"

    @respx.mock
    @pytest.mark.asyncio
    async def test_await_for(self, evosims):
        respx.get(f"{_EVOSIMS}/es1").mock(side_effect=[
            httpx.Response(200, json=_detail(status="RUNNING")),
            httpx.Response(200, json=_detail(status="SUCCEEDED")),
        ])
        run = await evosims.await_for("es1", timeout=60, interval=0)
        assert run.is_terminal is True
