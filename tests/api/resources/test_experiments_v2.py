"""LUC-908 — client.experiments reads (list / count / get)."""
import httpx
import pytest
import respx

from lucidicai.api.models.experiment import Experiment
from lucidicai.api.resources.experiment import ExperimentResource
from lucidicai.core.errors import NotFoundError

_BASE = "https://stub.lucidic.test"
_EXPERIMENTS = f"{_BASE}/sdk/v2/experiments"


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
