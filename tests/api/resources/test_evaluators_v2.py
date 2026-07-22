"""LUC-910 — client.evaluators reads (list / get / evals / result)."""
import httpx
import pytest
import respx

from lucidicai.api.models.evaluator import Evaluator
from lucidicai.api.models.session import EvalResult
from lucidicai.api.resources.evaluators import EvaluatorsResource
from lucidicai.core.errors import NotFoundError

_BASE = "https://stub.lucidic.test"
_EVALUATORS = f"{_BASE}/sdk/v2/evaluators"


@pytest.fixture
def evaluators(http):
    return EvaluatorsResource(http, agent_id="a1")


def _evaluator(i, **over):
    d = {
        "evaluator_id": f"e{i}", "name": f"eval-{i}", "description": "", "icon": "star",
        "type": "LLM", "result_type": "boolean", "is_default": False,
        "created_at": "2026-07-22T00:00:00Z", "updated_at": "2026-07-22T00:00:00Z",
        "criteria": [],
    }
    d.update(over)
    return d


def _result(i, **over):
    d = {"eval_id": f"r{i}", "evaluator_name": "eval-1", "result": True,
         "result_type": "boolean", "is_pending": False}
    d.update(over)
    return d


class TestListGet:
    @respx.mock
    def test_list_typed_and_defaults_agent(self, evaluators):
        route = respx.get(_EVALUATORS).mock(side_effect=[
            httpx.Response(200, json={"results": [_evaluator(1), _evaluator(2)],
                                      "next": f"{_EVALUATORS}?cursor=c2"}),
            httpx.Response(200, json={"results": [_evaluator(3)], "next": None}),
        ])
        got = list(evaluators.list())
        assert [e.evaluator_id for e in got] == ["e1", "e2", "e3"]
        assert all(isinstance(e, Evaluator) for e in got)
        assert route.calls[0].request.url.params["agent_id"] == "a1"

    @respx.mock
    def test_get_detail(self, evaluators):
        respx.get(f"{_EVALUATORS}/e1").mock(return_value=httpx.Response(
            200, json=_evaluator(1, rubric_json={"q": "is it good?"}, config={"model": "gpt"})))
        e = evaluators.get("e1")
        assert isinstance(e, Evaluator)
        assert e.rubric_json == {"q": "is it good?"}
        assert e.config == {"model": "gpt"}

    @respx.mock
    def test_get_404(self, evaluators):
        respx.get(f"{_EVALUATORS}/missing").mock(
            return_value=httpx.Response(404, json={"error": "not found"}))
        with pytest.raises(NotFoundError):
            evaluators.get("missing")


class TestEvalsDistribution:
    @respx.mock
    def test_evals_typed_and_no_agent_param(self, evaluators):
        route = respx.get(f"{_EVALUATORS}/e1/evals").mock(side_effect=[
            httpx.Response(200, json={"results": [_result(1), _result(2)],
                                      "next": f"{_EVALUATORS}/e1/evals?cursor=c2"}),
            httpx.Response(200, json={"results": [_result(3)], "next": None}),
        ])
        got = list(evaluators.evals("e1"))
        assert [r.eval_id for r in got] == ["r1", "r2", "r3"]
        assert all(isinstance(r, EvalResult) for r in got)
        # path-keyed: no agent_id forwarded on the distribution read
        assert "agent_id" not in route.calls[0].request.url.params

    @respx.mock
    def test_evals_page(self, evaluators):
        respx.get(f"{_EVALUATORS}/e1/evals").mock(return_value=httpx.Response(
            200, json={"results": [_result(1)], "next": None}))
        page = evaluators.evals_page("e1")
        assert isinstance(page.results[0], EvalResult)


class TestAgentIdGuard:
    def test_list_without_agent_id_raises(self, http):
        # No configured agent + none passed -> a clear AgentIdRequiredError
        # (raised eagerly at the .list() call, before any HTTP), not a backend
        # 400 or an empty-string param (LUC-926).
        from lucidicai.core.errors import AgentIdRequiredError
        with pytest.raises(AgentIdRequiredError):
            EvaluatorsResource(http, agent_id=None).list()


class TestAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_alist(self, evaluators):
        respx.get(_EVALUATORS).mock(side_effect=[
            httpx.Response(200, json={"results": [_evaluator(1)], "next": f"{_EVALUATORS}?cursor=c2"}),
            httpx.Response(200, json={"results": [_evaluator(2)], "next": None}),
        ])
        got = [e.evaluator_id async for e in evaluators.alist()]
        assert got == ["e1", "e2"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_aevals(self, evaluators):
        respx.get(f"{_EVALUATORS}/e1/evals").mock(return_value=httpx.Response(
            200, json={"results": [_result(1)], "next": None}))
        got = [r.eval_id async for r in evaluators.aevals("e1")]
        assert got == ["r1"]

