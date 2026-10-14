"""LUC-910 — client.evaluators reads (list / get / evals / result).
LUC-916 — client.evaluators writes (create LLM / update / delete)."""
import json

import httpx
import pytest
import respx

from lucidicai.api.models.evaluator import Evaluator
from lucidicai.api.models.session import EvalResult
from lucidicai.api.resources.evaluators import EvaluatorsResource
from lucidicai.core.errors import (
    ConflictError,
    InvalidOperationError,
    NotFoundError,
    ValidationError,
)

# A valid pass/fail criterion (boolean result_type).
_CRIT = [{"name": "accurate", "pass_definition": "correct", "fail_definition": "wrong"}]

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


# ==================== LUC-916 writes ====================


class TestCreate:
    @respx.mock
    def test_create_llm_boolean(self, evaluators):
        route = respx.post(_EVALUATORS).mock(return_value=httpx.Response(201, json=_evaluator(1)))
        e = evaluators.create("eval-1", criteria=_CRIT, result_type="boolean")
        assert isinstance(e, Evaluator) and e.evaluator_id == "e1"
        body = json.loads(route.calls.last.request.read())
        assert body["agent_id"] == "a1"  # defaulted from configured agent
        assert body["name"] == "eval-1" and body["result_type"] == "boolean"
        assert body["criteria"] == _CRIT and body["is_default"] is False
        # omit-None: unspecified optionals aren't sent (backend defaults icon=compass)
        assert "description" not in body and "icon" not in body
        # the SDK never sends a type / rubric_json — the backend hardcodes LLM + builds the rubric
        assert "type" not in body and "rubric_json" not in body

    @respx.mock
    def test_create_with_description_icon_default(self, evaluators):
        route = respx.post(_EVALUATORS).mock(return_value=httpx.Response(201, json=_evaluator(1)))
        evaluators.create("eval-1", criteria=_CRIT, result_type="boolean",
                          description="d", icon="wave", is_default=True)
        body = json.loads(route.calls.last.request.read())
        assert body["description"] == "d" and body["icon"] == "wave" and body["is_default"] is True

    @respx.mock
    def test_create_rejects_code_type_before_http(self, evaluators):
        route = respx.post(_EVALUATORS).mock(return_value=httpx.Response(201, json=_evaluator(1)))
        # A LucidicError subclass (not a bare ValueError) so `except LucidicError` catches it.
        with pytest.raises(InvalidOperationError):
            evaluators.create("eval-1", criteria=_CRIT, result_type="boolean", type="code")
        assert not route.called  # rejected client-side, no request made

    @respx.mock
    def test_create_rejects_non_string_type_cleanly(self, evaluators):
        # A non-string type must raise the clear guard error, not an opaque AttributeError.
        route = respx.post(_EVALUATORS).mock(return_value=httpx.Response(201, json=_evaluator(1)))
        with pytest.raises(InvalidOperationError):
            evaluators.create("eval-1", criteria=_CRIT, result_type="boolean", type=123)
        assert not route.called

    @respx.mock
    def test_create_accepts_padded_llm(self, evaluators):
        # Whitespace/case around a genuine "llm" is normalized, not rejected.
        respx.post(_EVALUATORS).mock(return_value=httpx.Response(201, json=_evaluator(1)))
        assert evaluators.create("eval-1", criteria=_CRIT, result_type="boolean",
                                 type="  LLM  ").evaluator_id == "e1"

    @respx.mock
    def test_create_duplicate_409(self, evaluators):
        respx.post(_EVALUATORS).mock(return_value=httpx.Response(
            409, json={"error": "An evaluator named 'eval-1' already exists for this agent."}))
        with pytest.raises(ConflictError):
            evaluators.create("eval-1", criteria=_CRIT, result_type="boolean")

    @respx.mock
    def test_create_bad_criteria_400(self, evaluators):
        respx.post(_EVALUATORS).mock(return_value=httpx.Response(
            400, json={"error": "Validation failed", "details": {"criteria": ["missing pass_definition"]}}))
        with pytest.raises(ValidationError):
            evaluators.create("eval-1", criteria=[{"name": "x"}], result_type="boolean")

    def test_create_without_agent_id_raises(self, http):
        from lucidicai.core.errors import AgentIdRequiredError
        res = EvaluatorsResource(http, agent_id=None)
        with pytest.raises(AgentIdRequiredError):
            res.create("eval-1", criteria=_CRIT, result_type="boolean")


class TestUpdate:
    @respx.mock
    def test_update_partial_name_only(self, evaluators):
        route = respx.put(f"{_EVALUATORS}/e1").mock(return_value=httpx.Response(
            200, json=_evaluator(1, name="renamed")))
        e = evaluators.update("e1", name="renamed")
        assert isinstance(e, Evaluator) and e.name == "renamed"
        body = json.loads(route.calls.last.request.read())
        assert body["name"] == "renamed"
        # partial update — only the provided field is sent
        assert all(k not in body for k in
                   ("description", "icon", "is_default", "criteria", "result_type", "config"))

    @respx.mock
    def test_update_criteria_and_code_config(self, evaluators):
        route = respx.put(f"{_EVALUATORS}/e1").mock(return_value=httpx.Response(200, json=_evaluator(1)))
        evaluators.update("e1", criteria=_CRIT, is_default=True, config={"user_code": "return 1"})
        body = json.loads(route.calls.last.request.read())
        assert body["criteria"] == _CRIT and body["is_default"] is True
        assert body["config"] == {"user_code": "return 1"}
        assert "name" not in body

    @respx.mock
    def test_update_rename_collision_409(self, evaluators):
        respx.put(f"{_EVALUATORS}/e1").mock(return_value=httpx.Response(
            409, json={"error": "An evaluator named 'taken' already exists for this agent."}))
        with pytest.raises(ConflictError):
            evaluators.update("e1", name="taken")

    @respx.mock
    def test_update_with_no_fields_raises_before_http(self, evaluators):
        # An all-None update would be an empty no-op write — surface the mistake.
        route = respx.put(f"{_EVALUATORS}/e1").mock(return_value=httpx.Response(200, json=_evaluator(1)))
        with pytest.raises(InvalidOperationError):
            evaluators.update("e1")
        assert not route.called


class TestDelete:
    @respx.mock
    def test_delete_204(self, evaluators):
        route = respx.delete(f"{_EVALUATORS}/e1").mock(return_value=httpx.Response(204))
        assert evaluators.delete("e1") is None
        assert route.called

    @respx.mock
    def test_delete_404(self, evaluators):
        respx.delete(f"{_EVALUATORS}/missing").mock(return_value=httpx.Response(
            404, json={"error": "Specified Evaluator not found"}))
        with pytest.raises(NotFoundError):
            evaluators.delete("missing")


class TestWriteAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_acreate(self, evaluators):
        respx.post(_EVALUATORS).mock(return_value=httpx.Response(201, json=_evaluator(1)))
        e = await evaluators.acreate("eval-1", criteria=_CRIT, result_type="boolean")
        assert e.evaluator_id == "e1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_acreate_rejects_code_case_insensitive(self, evaluators):
        route = respx.post(_EVALUATORS).mock(return_value=httpx.Response(201, json=_evaluator(1)))
        with pytest.raises(InvalidOperationError):
            await evaluators.acreate("eval-1", criteria=_CRIT, result_type="boolean", type="CODE")
        assert not route.called

    @respx.mock
    @pytest.mark.asyncio
    async def test_aupdate(self, evaluators):
        respx.put(f"{_EVALUATORS}/e1").mock(return_value=httpx.Response(200, json=_evaluator(1)))
        e = await evaluators.aupdate("e1", description="d")
        assert e.evaluator_id == "e1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_adelete(self, evaluators):
        route = respx.delete(f"{_EVALUATORS}/e1").mock(return_value=httpx.Response(204))
        assert await evaluators.adelete("e1") is None
        assert route.called

