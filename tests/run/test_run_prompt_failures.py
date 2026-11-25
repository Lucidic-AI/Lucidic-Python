"""A checkpoint case must not run on the baseline prompt when the fetch fails, and a case that
fails records its error on the case's session."""
import json
import logging

import httpx
import pytest

import lucidicai as lucidic
from lucidicai import LucidicAI, PromptError, ServiceUnavailableError

from .conftest import AGENT_ID, CHECKPOINT_ID, SESSION_ID, full_payload

_DEFAULT = "You are a careful lawyer."


@pytest.fixture
def production_client():
    c = LucidicAI(api_key="test-key", agent_id=AGENT_ID, production=True, eval_flush_timeout=2)
    yield c
    c.close(eval_flush_timeout=0)


@pytest.fixture(params=[True, False], ids=["production", "default"])
def any_client(request):
    c = LucidicAI(api_key="test-key", agent_id=AGENT_ID, production=request.param, eval_flush_timeout=2)
    yield c
    c.close(eval_flush_timeout=0)


def _load(client, **overrides):
    payload = {k: v for k, v in full_payload(**overrides).items() if k != "input"}
    return lucidic.load(client=client, **{k: v for k, v in payload.items() if v is not None})


@pytest.fixture
def checkpoint_run(any_client):
    return _load(any_client)


@pytest.fixture
def production_checkpoint_run(production_client):
    return _load(production_client)


@pytest.fixture
def baseline_run(production_client):
    return _load(production_client, checkpoint_id=None, experiment_id=None, eval_run_id=None)


def _prompts(backend, **mock):
    return backend.get("/sdk/prompts", name="prompts").mock(**mock)


def _assert_names_the_failure(error, cause_type):
    message = str(error)
    assert "'system'" in message
    assert CHECKPOINT_ID in message
    assert cause_type.__name__ in message


def test_checkpoint_run_raises_when_the_prompt_service_is_down(checkpoint_run, backend):
    _prompts(backend, return_value=httpx.Response(503, json={"error": "down"}))

    with pytest.raises(PromptError) as raised:
        checkpoint_run.prompt("system", default=_DEFAULT)

    _assert_names_the_failure(raised.value, ServiceUnavailableError)
    assert isinstance(raised.value.__cause__, ServiceUnavailableError)


def test_checkpoint_run_raises_when_the_prompt_fetch_times_out(checkpoint_run, backend):
    _prompts(backend, side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(PromptError) as raised:
        checkpoint_run.prompt("system", default=_DEFAULT)

    assert "'system'" in str(raised.value)
    assert CHECKPOINT_ID in str(raised.value)


@pytest.mark.asyncio
async def test_checkpoint_run_aprompt_raises_when_the_prompt_service_is_down(checkpoint_run, backend):
    _prompts(backend, return_value=httpx.Response(503, json={"error": "down"}))

    with pytest.raises(PromptError) as raised:
        await checkpoint_run.aprompt("system", default=_DEFAULT)

    _assert_names_the_failure(raised.value, ServiceUnavailableError)


def test_production_checkpoint_run_still_uses_the_default_for_a_missing_prompt(production_checkpoint_run, backend):
    _prompts(backend, return_value=httpx.Response(404, json={"error": "Prompt not found"}))

    assert production_checkpoint_run.prompt("system", default=_DEFAULT) == _DEFAULT
    assert production_checkpoint_run.prompt("other") == ""


def test_default_client_run_without_checkpoint_raises_the_api_error(client, backend):
    run = _load(client, checkpoint_id=None, experiment_id=None, eval_run_id=None)
    _prompts(backend, return_value=httpx.Response(503, json={"error": "down"}))

    with pytest.raises(ServiceUnavailableError):
        run.prompt("system", default=_DEFAULT)


def test_production_run_without_checkpoint_falls_back_with_a_warning(baseline_run, backend, caplog):
    _prompts(backend, return_value=httpx.Response(503, json={"error": "down"}))

    with caplog.at_level(logging.WARNING, logger="Lucidic"):
        assert baseline_run.prompt("system", default=_DEFAULT) == _DEFAULT

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "'system'" in r.getMessage()]
    assert warnings


@pytest.mark.asyncio
async def test_production_run_without_checkpoint_aprompt_falls_back_with_a_warning(baseline_run, backend, caplog):
    _prompts(backend, return_value=httpx.Response(503, json={"error": "down"}))

    with caplog.at_level(logging.WARNING, logger="Lucidic"):
        assert await baseline_run.aprompt("system", default=_DEFAULT) == _DEFAULT

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "'system'" in r.getMessage()]
    assert warnings


def _record_order(backend):
    order = []

    def _event(request):
        order.append(("event", json.loads(request.content)))
        return httpx.Response(201, json={})

    def _end(request):
        order.append(("end", json.loads(request.content)))
        return httpx.Response(200, json={})

    backend.post("/events", name="events").mock(side_effect=_event)
    backend["updatesession"].mock(side_effect=_end)
    return order


def _assert_error_recorded_before_end(order):
    assert [kind for kind, _ in order] == ["event", "end"]
    event = order[0][1]
    assert event["type"] == "error_traceback"
    assert event["session_id"] == SESSION_ID
    error = event["payload"]["error"]
    assert error.startswith("PromptError")
    assert "'system'" in error and CHECKPOINT_ID in error
    assert "ServiceUnavailableError" in event["payload"]["traceback"]
    assert order[1][1]["is_finished"] is True


def test_failed_case_is_recorded_on_its_session(checkpoint_run, backend):
    _prompts(backend, return_value=httpx.Response(503, json={"error": "down"}))
    order = _record_order(backend)

    with pytest.raises(PromptError):
        with checkpoint_run:
            checkpoint_run.prompt("system", default=_DEFAULT)

    _assert_error_recorded_before_end(order)


@pytest.mark.asyncio
async def test_failed_async_case_is_recorded_on_its_session(checkpoint_run, backend):
    _prompts(backend, return_value=httpx.Response(503, json={"error": "down"}))
    order = _record_order(backend)

    with pytest.raises(PromptError):
        async with checkpoint_run:
            await checkpoint_run.aprompt("system", default=_DEFAULT)

    _assert_error_recorded_before_end(order)


def test_case_error_propagates_when_recording_it_fails(client, backend):
    run = _load(client)
    backend.post("/events", name="events").mock(return_value=httpx.Response(500, json={"error": "boom"}))

    with pytest.raises(RuntimeError, match="agent crashed"):
        with run:
            raise RuntimeError("agent crashed")

    assert backend["events"].call_count == 1
    assert backend["updatesession"].call_count == 1
