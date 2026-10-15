"""Shared fixtures for lucidicai.load() / Run tests: hermetic env, isolated context, a stubbed backend."""
import json
import sys

import httpx
import pytest
import respx

from lucidicai import LucidicAI, webhook
from lucidicai.sdk.context import current_client, current_session_id
from lucidicai.sdk.tools.context import current_mock_context
from lucidicai.sdk.tools.registry import _PENDING_BUFFER, _REGISTRY

BASE_URL = "https://stub.lucidic.test"
SECRET = "test-signing-secret"
SESSION_ID = "6f1c2b9e-3d4a-4e5f-8a7b-0c1d2e3f4a5b"
ITEM_ID = "0a0b0c0d-1111-2222-3333-444455556666"
EXPERIMENT_ID = "e0e0e0e0-aaaa-bbbb-cccc-dddddddddddd"
EVAL_RUN_ID = "beefbeef-0000-1111-2222-333344445555"
CHECKPOINT_ID = "c0c0c0c0-9999-8888-7777-666655554444"
AGENT_ID = "aaaa1111-0000-0000-0000-000000000000"
TOOL = {"tool_name": "get_style_guide", "tool_description": "style",
        "input_schema": {"type": "object"}, "edit_site": "system"}


def full_payload(**overrides):
    payload = {
        "session_id": SESSION_ID,
        "datasetitem_id": ITEM_ID,
        "input": {"task": "corporate-ma/x", "documents": ["a.pdf"]},
        "experiment_id": EXPERIMENT_ID,
        "eval_run_id": EVAL_RUN_ID,
        "checkpoint_id": CHECKPOINT_ID,
    }
    payload.update(overrides)
    return payload


def wire(payload) -> bytes:
    # the backend's serialisation (webhook_egress.py::sign_body)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")


def signed(payload, secret=SECRET):
    body = payload if isinstance(payload, bytes) else wire(payload)
    return body, webhook.sign(secret, body)


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    monkeypatch.setenv("LUCIDIC_DEBUG", "false")
    monkeypatch.setenv("LUCIDIC_BASE_URL", BASE_URL)
    monkeypatch.setenv("LUCIDIC_MAX_RETRIES", "0")
    for name in ("LUCIDIC_REGION", "LUCIDIC_RUN", "LUCIDIC_WEBHOOK_SECRET", "LUCIDIC_PRODUCTION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["agent.py"])
    tokens = (current_client.set(None), current_session_id.set(None), current_mock_context.set(None))
    _REGISTRY.clear()
    _PENDING_BUFFER.clear()
    yield
    _REGISTRY.clear()
    _PENDING_BUFFER.clear()
    current_mock_context.reset(tokens[2])
    current_session_id.reset(tokens[1])
    current_client.reset(tokens[0])


@pytest.fixture
def client():
    c = LucidicAI(api_key="test-key", agent_id=AGENT_ID, production=False, eval_flush_timeout=2)
    yield c
    c.close(eval_flush_timeout=0)


@pytest.fixture
def backend():
    """respx router with the session endpoints a `with run:` touches."""
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as router:
        router.post("/initsession", name="initsession").mock(
            return_value=httpx.Response(201, json={"session_id": SESSION_ID}))
        router.put("/updatesession", name="updatesession").mock(
            return_value=httpx.Response(200, json={}))
        router.post("/sdk/session-init-fixtures", name="init_fixtures").mock(
            return_value=httpx.Response(200, json={"initialized": True, "fixture_ids": [],
                                                   "tools_snapshotted": [], "already_initialized": False}))
        router.post("/sdk/agent-tools/sync", name="sync").mock(
            return_value=httpx.Response(200, json={"synced": True, "stats": {}}))
        router.get("/sdk/training-modules/tools", name="tools").mock(
            return_value=httpx.Response(200, json={"checkpoint_id": CHECKPOINT_ID, "tools": [TOOL]}))
        yield router
