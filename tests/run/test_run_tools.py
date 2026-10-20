"""run.tools: the checkpoint's hosted module tools, fetched once, rendered three ways."""
import json

import httpx
import pytest

import lucidicai as lucidic
from lucidicai import InvalidOperationError

from .conftest import CHECKPOINT_ID, SESSION_ID

_NEUTRAL = {"name": "get_style_guide", "description": "style", "input_schema": {"type": "object"}}


@pytest.fixture
def run(client):
    return lucidic.load(client=client, session_id=SESSION_ID, checkpoint_id=CHECKPOINT_ID)


def test_baseline_run_has_no_tools_and_makes_no_request(client, backend):
    run = lucidic.load(client=client, session_id=SESSION_ID)
    assert run.tools.list() == []
    assert run.tools.openai() == []
    assert backend["tools"].call_count == 0


def test_list_is_provider_neutral(run, backend):
    assert run.tools.list() == [_NEUTRAL]
    assert backend["tools"].calls.last.request.url.params["checkpoint_id"] == CHECKPOINT_ID


def test_openai_renders_function_specs(run, backend):
    assert run.tools.openai() == [{"type": "function", "function": {
        "name": "get_style_guide", "description": "style", "parameters": {"type": "object"}}}]


def test_anthropic_renders_tool_specs(run, backend):
    assert run.tools.anthropic() == [_NEUTRAL]


def test_tools_are_fetched_once_per_run(run, backend):
    run.tools.list()
    run.tools.openai()
    with run:
        run.tools.anthropic()
    assert backend["tools"].call_count == 1


def test_call_outside_the_run_raises(run):
    with pytest.raises(InvalidOperationError):
        run.tools.call("get_style_guide", {})


def test_call_runs_the_tool_in_the_run_session(run, backend):
    submit = backend.post("/sdk/training-modules/inferences").mock(return_value=httpx.Response(
        201, json={"status": "SUCCEEDED", "inference_run_id": "ir1", "result": {"return_value": "be terse"}}))
    with run:
        assert run.tools.call("get_style_guide", {"tone": "formal"}) == "be terse"
    body = json.loads(submit.calls.last.request.content)
    assert body["session_id"] == SESSION_ID
    assert body["tool_name"] == "get_style_guide"
    assert body["arguments"] == {"tone": "formal"}


@pytest.mark.asyncio
async def test_acall(run, backend):
    backend.post("/sdk/training-modules/inferences").mock(return_value=httpx.Response(
        201, json={"status": "SUCCEEDED", "inference_run_id": "ir1", "result": {"return_value": 7}}))
    async with run:
        assert await run.tools.acall("get_style_guide") == 7
