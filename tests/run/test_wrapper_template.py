"""examples/lucidic_entry.py and examples/lucidic_runner.py, loaded by path with a stub agent."""
import importlib.util
import json
import sys
import threading
import types
from pathlib import Path

import httpx
import pytest

import lucidicai as lucidic

from .conftest import SECRET, SESSION_ID, full_payload, signed

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
OTHER_IDS = [f"{i}{i}{i}{i}{i}{i}{i}{i}-0000-0000-0000-000000000000" for i in range(1, 4)]


def _load_example(name):
    spec = importlib.util.spec_from_file_location(f"_example_{name}", _EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner_module(monkeypatch):
    monkeypatch.setenv("LUCIDIC_WEBHOOK_SECRET", SECRET)
    return _load_example("lucidic_runner")


def test_entry_runs_the_agent_inside_the_case_session(client, backend, monkeypatch):
    seen = {}

    def their_agent(task, *, system, extra_tools, call_extra_tool):
        seen.update(task=task, system=system, extra_tools=extra_tools,
                    session=lucidic.sdk.context.current_session_id.get())
        return "answer"

    their_agent.SYSTEM_PROMPT = "You are a careful lawyer."
    monkeypatch.setitem(sys.modules, "their_pkg", types.SimpleNamespace(their_agent=their_agent))
    monkeypatch.setattr(lucidic, "LucidicAI", lambda **kwargs: client)
    backend.get("/sdk/prompts").mock(return_value=httpx.Response(404, json={"error": "Prompt not found"}))
    create = backend.post("/sdk/v2/prompts").mock(return_value=httpx.Response(
        201, json={"prompt_id": "p1", "name": "system", "icon": "compass", "preview": ""}))
    entry = _load_example("lucidic_entry")

    body, signature = signed(full_payload())
    assert entry.run_case(lucidic.load(body, signature, secret=SECRET, client=client)) == "answer"

    assert seen["task"] == full_payload()["input"]
    assert seen["system"] == "You are a careful lawyer."
    assert [tool["name"] for tool in seen["extra_tools"]] == ["get_style_guide"]
    assert seen["session"] == SESSION_ID
    assert json.loads(create.calls.last.request.content)["name"] == "system"
    assert backend["updatesession"].call_count == 1


def test_runner_accepts_once_per_session_id(runner_module, client):
    ran = []
    runner = runner_module.Runner(lambda run: ran.append(run.ids.session_id), client, max_workers=2)
    body, signature = signed(full_payload())

    assert runner.handle(body, signature) == 202
    assert runner.handle(body, signature) == 202   # at-least-once re-delivery
    runner._pool.shutdown(wait=True)
    assert ran == [SESSION_ID]


def test_runner_replies_with_the_load_error_status(runner_module, client):
    runner = runner_module.Runner(lambda run: None, client)
    body, _ = signed(full_payload())
    assert runner.handle(body, "sha256=00") == 401
    assert runner.handle(*signed({"session_id": "not-a-uuid"})) == 422


def test_runner_sheds_load_when_the_pool_is_full(runner_module, client):
    release = threading.Event()
    runner = runner_module.Runner(lambda run: release.wait(5), client, max_workers=1)
    try:
        statuses = [runner.handle(*signed(full_payload(session_id=sid))) for sid in OTHER_IDS]
    finally:
        release.set()
        runner._pool.shutdown(wait=True)
    # one running + one queued; the backend re-sends the 503 later
    assert statuses == [202, 202, 503]
