"""run.prompt(): checkpoint-pinned > label > local default, resolved once per run."""
import json

import httpx
import pytest
import respx

import lucidicai as lucidic
from lucidicai import LucidicAI, NotFoundError, ServiceUnavailableError

from .conftest import AGENT_ID, BASE_URL, CHECKPOINT_ID, SESSION_ID, full_payload

_PROMPTS = f"{BASE_URL}/sdk/prompts"
_CREATE = f"{BASE_URL}/sdk/v2/prompts"
_DEFAULT = "You are a careful lawyer."


@pytest.fixture
def run(client):
    return lucidic.load(client=client, **{k: v for k, v in full_payload().items() if k != "input"})


def _missing():
    return respx.get(_PROMPTS).mock(return_value=httpx.Response(404, json={"error": "Prompt not found"}))


@respx.mock
def test_sends_the_run_ids_explicitly(run):
    route = respx.get(_PROMPTS).mock(return_value=httpx.Response(200, json={"prompt_content": "pinned"}))

    assert run.prompt("system", default=_DEFAULT) == "pinned"
    params = route.calls.last.request.url.params
    assert params["checkpoint_id"] == CHECKPOINT_ID
    assert params["session_id"] == SESSION_ID
    assert params["label"] == "production"


@respx.mock
def test_missing_prompt_falls_back_to_the_default(run):
    _missing()
    assert run.prompt("system", default=_DEFAULT) == _DEFAULT


@respx.mock
def test_missing_prompt_without_default_raises(run):
    _missing()
    with pytest.raises(NotFoundError):
        run.prompt("system")


@respx.mock
def test_outage_raises_instead_of_using_the_default(run):
    # a checkpoint run must never silently grade the baseline prompt.
    respx.get(_PROMPTS).mock(return_value=httpx.Response(503, json={"error": "down"}))
    with pytest.raises(ServiceUnavailableError):
        run.prompt("system", default=_DEFAULT)


@respx.mock
def test_each_prompt_is_fetched_once_per_run(run):
    route = respx.get(_PROMPTS).mock(return_value=httpx.Response(200, json={"prompt_content": "v1"}))
    assert run.prompt("system") == "v1"
    route.mock(return_value=httpx.Response(200, json={"prompt_content": "v2"}))
    assert run.prompt("system") == "v1"
    assert route.call_count == 1


@respx.mock
def test_register_creates_a_missing_prompt_from_the_default(run):
    _missing()
    create = respx.post(_CREATE).mock(return_value=httpx.Response(
        201, json={"prompt_id": "p1", "name": "system", "icon": "compass", "preview": ""}))

    assert run.prompt("system", default=_DEFAULT, register=True) == _DEFAULT
    body = json.loads(create.calls.last.request.content)
    assert (body["name"], body["prompt_content"], body["agent_id"]) == ("system", _DEFAULT, AGENT_ID)


@respx.mock
def test_register_tolerates_another_worker_registering_first(run):
    _missing()
    respx.post(_CREATE).mock(return_value=httpx.Response(409, json={"error": "exists"}))
    assert run.prompt("system", default=_DEFAULT, register=True) == _DEFAULT


@respx.mock
def test_register_without_write_scope_still_returns_the_default(run, caplog):
    _missing()
    respx.post(_CREATE).mock(return_value=httpx.Response(
        403, json={"error": "missing scope", "required_scope": "prompt:write"}))
    assert run.prompt("system", default=_DEFAULT, register=True) == _DEFAULT
    assert "could not register" in caplog.text


@respx.mock
def test_production_client_empty_prompt_uses_the_default():
    create = respx.post(_CREATE)
    respx.get(_PROMPTS).mock(return_value=httpx.Response(503, json={"error": "down"}))
    client = LucidicAI(api_key="test-key", agent_id=AGENT_ID, production=True)
    run = lucidic.load(client=client, session_id=SESSION_ID)

    assert run.prompt("system", default=_DEFAULT, register=True) == _DEFAULT
    assert create.call_count == 0


@pytest.mark.asyncio
@respx.mock
async def test_aprompt(run):
    respx.get(_PROMPTS).mock(return_value=httpx.Response(200, json={"prompt_content": "pinned"}))
    assert await run.aprompt("system", default=_DEFAULT) == "pinned"


@respx.mock
def test_register_network_error_still_returns_the_default(run, caplog):
    _missing()
    respx.post(_CREATE).mock(side_effect=httpx.ConnectError("down"))
    assert run.prompt("system", default=_DEFAULT, register=True) == _DEFAULT
    assert "could not register" in caplog.text
