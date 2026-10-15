"""run.session_id: the server's session id, which can differ from the dispatched ids.session_id (LUC-1138)."""
import contextvars
import json

import httpx
import pytest

import lucidicai as lucidic
from lucidicai.sdk.context import current_session_id
from lucidicai.sdk.tools.context import _current_mock_context

from .conftest import SECRET, SESSION_ID, full_payload, signed

SERVER_ID = "5e5e5e5e-1234-4abc-8def-0123456789ab"


@pytest.fixture
def run(client):
    body, signature = signed(full_payload())
    return lucidic.load(body, signature, secret=SECRET, client=client)


@pytest.fixture
def server(backend):
    # InitSession gives a dispatched session a fresh primary key
    backend["initsession"].mock(return_value=httpx.Response(201, json={"session_id": SERVER_ID}))
    return backend


def test_session_id_is_the_server_id_once_open(run, server):
    assert run.session_id is None
    with run:
        assert run.session_id == SERVER_ID
        assert current_session_id.get() == SERVER_ID
    assert run.session_id == SERVER_ID
    assert run.ids.session_id == SESSION_ID


@pytest.mark.asyncio
async def test_async_with_sets_the_server_id(run, server):
    async with run:
        assert run.session_id == SERVER_ID
    assert run.ids.session_id == SESSION_ID


def test_evals_and_updates_by_run_session_id_reach_the_session(run, server):
    evals = server.post("/sdk/evals").mock(return_value=httpx.Response(201, json={}))
    with run:
        run.client.evals.emit(result=0.9, name="score", session_id=run.session_id)
    assert json.loads(evals.calls.last.request.content)["session_id"] == SERVER_ID


def test_tools_call_targets_the_server_session(run, server):
    submit = server.post("/sdk/training-modules/inferences").mock(return_value=httpx.Response(
        201, json={"status": "SUCCEEDED", "inference_run_id": "ir1", "result": {"return_value": "ok"}}))
    with run:
        run.tools.call("get_style_guide", {})
    assert json.loads(submit.calls.last.request.content)["session_id"] == SERVER_ID


def test_prompt_inside_the_run_targets_the_server_session(run, server):
    route = server.get("/sdk/prompts").mock(return_value=httpx.Response(200, json={"prompt_content": "p"}))
    with run:
        run.prompt("system")
    assert route.calls.last.request.url.params["session_id"] == SERVER_ID


def test_exit_clears_the_mock_context_bound_to_the_server_id(run, server):
    with run:
        assert _current_mock_context().session_id == SERVER_ID
    assert _current_mock_context() is None


def test_child_of_an_open_run_joins_the_server_session(run, server, monkeypatch):
    def child_process():
        with lucidic.load(client=run.client) as child:
            return child.session_id, current_session_id.get(), _current_mock_context().session_id

    with run:
        monkeypatch.setenv("LUCIDIC_RUN", run.to_json())
        seen = contextvars.Context().run(child_process)
    assert seen == (SERVER_ID, SERVER_ID, SERVER_ID)
    assert server["initsession"].call_count == 1
