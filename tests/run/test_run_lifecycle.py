"""`with run:`: open the case session (fixtures bound), then flush evals and end it."""
import contextvars
import dataclasses
import json
import time

import httpx
import pytest

import lucidicai as lucidic
from lucidicai import InvalidOperationError
from lucidicai.sdk.context import current_session_id
from lucidicai.sdk.tools.context import _current_mock_context

from .conftest import (
    AGENT_ID, CHECKPOINT_ID, EVAL_RUN_ID, EXPERIMENT_ID, ITEM_ID, SECRET, SESSION_ID, TOOL,
    full_payload, signed,
)


@pytest.fixture
def run(client):
    body, signature = signed(full_payload(style="formal"))
    return lucidic.load(body, signature, secret=SECRET, client=client)


def _ended(backend):
    return [json.loads(c.request.content).get("is_finished") for c in backend["updatesession"].calls]


def test_enter_creates_the_case_session(run, backend):
    with run:
        assert current_session_id.get() == SESSION_ID
    sent = json.loads(backend["initsession"].calls.last.request.content)
    assert sent["session_id"] == SESSION_ID
    assert sent["agent_id"] == AGENT_ID
    assert sent["datasetitem_id"] == ITEM_ID
    assert sent["experiment_id"] == EXPERIMENT_ID
    assert sent["checkpoint_id"] == CHECKPOINT_ID
    assert sent["tags"] == [f"eval_run:{EVAL_RUN_ID}"]
    assert _ended(backend) == [True]


def test_mockable_tools_are_answered_from_the_case_fixtures(run, backend):
    @lucidic.mockable
    def search_contracts(query: str) -> list:
        return ["real result"]

    mock_call = backend.post("/sdk/mock-call").mock(return_value=httpx.Response(
        200, json={"return_value": ["fixture result"], "tier": "PYTHON", "was_mocked": True}))
    with run:
        assert search_contracts("indemnity") == ["fixture result"]
    assert backend["init_fixtures"].call_count == 1
    assert mock_call.call_count == 1


def test_pending_evals_land_before_the_session_ends(run, backend):
    order = []

    def _slow_eval(request):
        time.sleep(0.2)
        order.append("eval")
        return httpx.Response(201, json={})

    backend.post("/sdk/evals").mock(side_effect=_slow_eval)
    backend["updatesession"].mock(side_effect=lambda request: order.append("end") or httpx.Response(200, json={}))
    with run:
        run.client.evals.emit(result=0.9, name="self_score")
    assert order == ["eval", "end"]


def test_agent_exception_propagates_and_the_session_still_ends(run, backend):
    with pytest.raises(RuntimeError, match="agent crashed"):
        with run:
            raise RuntimeError("agent crashed")
    assert _ended(backend) == [True]
    assert current_session_id.get() is None


def test_exit_clears_the_mock_context(run, backend):
    with run:
        assert _current_mock_context().session_id == SESSION_ID
    # a reused pool thread must not inherit this case's fixtures
    assert _current_mock_context() is None


def test_a_run_is_entered_once(run, backend):
    with run:
        pass
    with pytest.raises(InvalidOperationError):
        with run:
            pass


@pytest.mark.asyncio
async def test_async_with(run, backend):
    async with run:
        assert current_session_id.get() == SESSION_ID
        assert _current_mock_context() is not None
    assert backend["initsession"].call_count == 1
    assert _ended(backend) == [True]
    assert _current_mock_context() is None


def test_to_json_round_trips_with_the_frozen_config(run, backend, monkeypatch):
    prompts = backend.get("/sdk/prompts").mock(return_value=httpx.Response(200, json={"prompt_content": "pinned"}))
    run.prompt("system")
    run.tools.list()

    monkeypatch.setenv("LUCIDIC_RUN", run.to_json())
    child = lucidic.load(client=run.client)

    assert child.ids == run.ids
    assert child.input == run.input
    assert dict(child.params) == {"style": "formal"}
    assert child.prompt("system") == "pinned"
    assert child.tools.list() == run.tools.list()
    assert backend["tools"].call_count == 1
    assert prompts.call_count == 1


def test_to_env_feeds_load(run, monkeypatch):
    env = run.to_env()
    assert list(env) == ["LUCIDIC_RUN"]
    monkeypatch.setenv("LUCIDIC_RUN", env["LUCIDIC_RUN"])
    assert lucidic.load().ids == run.ids


def test_child_of_an_open_run_joins_its_session(run, backend, monkeypatch):
    @lucidic.mockable
    def search_contracts(query: str) -> list:
        return ["real result"]

    backend.post("/sdk/mock-call").mock(return_value=httpx.Response(
        200, json={"return_value": ["fixture result"], "tier": "PYTHON", "was_mocked": True}))
    def child_process():
        # a subprocess starts from an empty context and only has LUCIDIC_RUN
        with lucidic.load(client=run.client) as child:
            return current_session_id.get(), search_contracts("x")

    with run:
        monkeypatch.setenv("LUCIDIC_RUN", run.to_json())
        seen = contextvars.Context().run(child_process)
    assert seen == (SESSION_ID, ["fixture result"])
    # the parent created and ended the session; the child did neither
    assert backend["initsession"].call_count == 1
    assert _ended(backend) == [True]
    assert backend["init_fixtures"].call_count == 1


def test_open_run_marks_its_json_attached(run, backend):
    assert json.loads(run.to_json())["_lucidic"]["attached"] is False
    with run:
        snapshot = json.loads(run.to_json())["_lucidic"]
        assert (snapshot["attached"], snapshot["mock"], snapshot["tools"]) == (True, True, [TOOL])
    assert json.loads(run.to_json())["_lucidic"]["attached"] is False


def test_a_run_without_a_client_builds_and_closes_its_own(backend, monkeypatch):
    monkeypatch.setenv("LUCIDIC_API_KEY", "test-key")
    monkeypatch.setenv("LUCIDIC_AGENT_ID", AGENT_ID)
    run = lucidic.load(session_id=SESSION_ID)
    closed = []
    with run:
        monkeypatch.setattr(run.client, "close", lambda: closed.append(True))
    assert closed == [True]
    assert backend["initsession"].call_count == 1


def test_a_failed_enter_closes_the_runs_own_client(backend, monkeypatch):
    monkeypatch.setenv("LUCIDIC_API_KEY", "test-key")
    monkeypatch.setenv("LUCIDIC_AGENT_ID", AGENT_ID)
    backend["initsession"].mock(return_value=httpx.Response(500, json={"error": "boom"}))
    run = lucidic.load(session_id=SESSION_ID)
    closed = []
    monkeypatch.setattr(run.client, "close", lambda: closed.append(True))
    with pytest.raises(lucidic.LucidicError):
        with run:
            pass
    assert closed == [True]


def test_ids_and_params_are_read_only(run):
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.ids.session_id = "other"
    with pytest.raises(TypeError):
        run.params["style"] = "casual"
    with pytest.raises(AttributeError):
        run.input = "other"
