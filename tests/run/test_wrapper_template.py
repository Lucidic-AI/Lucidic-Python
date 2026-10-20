"""examples/lucidic_entry.py and examples/lucidic_runner.py, loaded by path with a stub agent."""
import importlib.util
import json
import runpy
import sys
import types
from pathlib import Path

import httpx

import lucidicai as lucidic

from .conftest import SECRET, SESSION_ID, full_payload, signed

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _load_example(name):
    spec = importlib.util.spec_from_file_location(f"_example_{name}", _EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entry_runs_the_agent_inside_the_case_session(client, backend, monkeypatch, tmp_path):
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
    documents = backend.get(f"/sdk/v2/sessions/{SESSION_ID}/documents").mock(
        return_value=httpx.Response(200, json={"expires_in": 1200, "documents": []}))
    monkeypatch.chdir(tmp_path)
    entry = _load_example("lucidic_entry")

    body, signature = signed(full_payload())
    assert entry.run_case(lucidic.load(body, signature, secret=SECRET, client=client)) == "answer"

    assert seen["task"] == full_payload()["input"]
    assert seen["system"] == "You are a careful lawyer."
    assert [tool["name"] for tool in seen["extra_tools"]] == ["get_style_guide"]
    assert seen["session"] == SESSION_ID
    assert json.loads(create.calls.last.request.content)["name"] == "system"
    assert backend["updatesession"].call_count == 1
    assert documents.call_count == 1


def test_runner_example_serves_the_entry_run_case(client, monkeypatch):
    def run_case(run):
        return None

    served = {}
    monkeypatch.setitem(sys.modules, "lucidic_entry", types.SimpleNamespace(client=client, run_case=run_case))
    monkeypatch.setattr(lucidic, "serve", lambda fn, **kwargs: served.update(fn=fn, **kwargs))
    monkeypatch.setenv("PORT", "9001")
    runpy.run_path(str(_EXAMPLES / "lucidic_runner.py"), run_name="__main__")

    assert served == {"fn": run_case, "host": "0.0.0.0", "port": 9001, "workers": 4, "client": client}
