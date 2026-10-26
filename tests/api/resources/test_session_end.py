"""Ending and updating a session: no session verdict.

A session's result is an eval, ``client.evals.emit(result, name=..., description=...)``.
The session methods no longer take ``is_successful``, ``is_successful_reason``,
``session_eval`` or ``session_eval_reason``, and ``update(**updates)`` refuses them.
The backend is mocked at the HTTP boundary with respx.
"""
import json
import threading

import httpx
import pytest
import respx

from lucidicai import LucidicAI, Session
from lucidicai.sdk import session as sdk_session
from lucidicai.sdk.context import current_client, current_session_id
from lucidicai.sdk.shutdown_manager import SessionState, ShutdownManager

_BASE_URL = "https://stub.lucidic.test"
_UPDATESESSION = f"{_BASE_URL}/updatesession"
_SESSION_ID = "11111111-2222-3333-4444-555555555555"
_VERDICT_FIELDS = ["is_successful", "is_successful_reason", "session_eval", "session_eval_reason"]
_VALUES = {"is_successful": True, "is_successful_reason": "done", "session_eval": 1.0,
           "session_eval_reason": "done"}


@pytest.fixture(autouse=True)
def _isolate_session_context():
    client_token = current_client.set(None)
    session_token = current_session_id.set(None)
    yield
    current_session_id.reset(session_token)
    current_client.reset(client_token)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    monkeypatch.setenv("LUCIDIC_DEBUG", "false")
    monkeypatch.setenv("LUCIDIC_BASE_URL", _BASE_URL)
    monkeypatch.delenv("LUCIDIC_REGION", raising=False)


@pytest.fixture
def client():
    c = LucidicAI(api_key="test-key", agent_id="aaaa1111-0000-0000-0000-000000000000",
                  production=False)
    yield c
    c.close()


def _body(route):
    body = json.loads(route.calls.last.request.read())
    body.pop("current_time")  # the transport stamps every PUT
    return body


class TestVerdictKwargsRemoved:
    @respx.mock
    @pytest.mark.parametrize("field", _VERDICT_FIELDS)
    @pytest.mark.parametrize("method", ["end", "end_session"])
    def test_sync_end_methods(self, client, method, field):
        route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
        with pytest.raises(TypeError, match=field):
            getattr(client.sessions, method)(session_id=_SESSION_ID, **{field: _VALUES[field]})
        assert not route.called

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field", _VERDICT_FIELDS)
    @pytest.mark.parametrize("method", ["aend", "aend_session"])
    async def test_async_end_methods(self, client, method, field):
        with respx.mock:
            route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(TypeError, match=field):
                await getattr(client.sessions, method)(session_id=_SESSION_ID, **{field: _VALUES[field]})
        assert not route.called

    @respx.mock
    @pytest.mark.parametrize("field", _VERDICT_FIELDS)
    @pytest.mark.parametrize("method", ["update", "end"])
    def test_session_object(self, client, method, field):
        route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
        session = Session(client=client, session_id=_SESSION_ID, auto_end=False)
        with pytest.raises(TypeError, match=field):
            getattr(session, method)(**{field: _VALUES[field]})
        assert not route.called
        assert not session.is_finished

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field", _VERDICT_FIELDS)
    @pytest.mark.parametrize("method", ["aupdate", "aend"])
    async def test_async_session_object(self, client, method, field):
        session = Session(client=client, session_id=_SESSION_ID, auto_end=False)
        with respx.mock:
            route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(TypeError, match=field):
                await getattr(session, method)(**{field: _VALUES[field]})
        assert not route.called

    @pytest.mark.parametrize("field", _VERDICT_FIELDS)
    def test_module_end_session(self, field):
        with pytest.raises(TypeError, match=field):
            sdk_session.end_session(session_id=_SESSION_ID, **{field: _VALUES[field]})

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field", _VERDICT_FIELDS)
    async def test_module_aend_session(self, field):
        with pytest.raises(TypeError, match=field):
            await sdk_session.aend_session(session_id=_SESSION_ID, **{field: _VALUES[field]})


class TestUpdateRefusesVerdictFields:
    @respx.mock
    @pytest.mark.parametrize("field", _VERDICT_FIELDS)
    def test_update(self, client, field):
        route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
        with pytest.raises(TypeError, match=rf"{field}.*client\.evals\.emit\(result, name="):
            client.sessions.update(_SESSION_ID, **{field: _VALUES[field]})
        assert not route.called

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field", _VERDICT_FIELDS)
    async def test_aupdate(self, client, field):
        with respx.mock:
            route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(TypeError, match=rf"{field}.*client\.evals\.emit\(result, name="):
                await client.sessions.aupdate(_SESSION_ID, **{field: _VALUES[field]})
        assert not route.called

    @respx.mock
    def test_names_every_refused_field(self, client):
        respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
        with pytest.raises(TypeError) as err:
            client.sessions.update(_SESSION_ID, task="t", is_successful=False, session_eval=0.0)
        assert "is_successful, session_eval" in str(err.value)

    @respx.mock
    def test_refused_even_in_production(self):
        c = LucidicAI(api_key="test-key", agent_id="aaaa1111-0000-0000-0000-000000000000",
                      production=True)
        try:
            route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(TypeError):
                c.sessions.update(_SESSION_ID, is_successful=True)
            assert not route.called
        finally:
            c.close()

    @respx.mock
    def test_other_fields_still_pass_through(self, client):
        route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
        client.sessions.update(_SESSION_ID, task="new task", tags=["a"])
        assert _body(route) == {"session_id": _SESSION_ID, "task": "new task", "tags": ["a"]}


class TestEndSendsNoVerdict:
    @respx.mock
    def test_end(self, client):
        route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
        client.sessions.end(_SESSION_ID)
        assert _body(route) == {"session_id": _SESSION_ID, "is_finished": True}

    @pytest.mark.asyncio
    async def test_aend(self, client):
        with respx.mock:
            route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
            await client.sessions.aend(_SESSION_ID)
        assert _body(route) == {"session_id": _SESSION_ID, "is_finished": True}

    @respx.mock
    def test_session_end(self, client):
        route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
        session = Session(client=client, session_id=_SESSION_ID, auto_end=False)
        session.end()
        assert session.is_finished
        assert _body(route) == {"session_id": _SESSION_ID, "is_finished": True}

    @respx.mock
    def test_session_update_sends_the_task_only(self, client):
        route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
        Session(client=client, session_id=_SESSION_ID, auto_end=False).update(task="t")
        assert _body(route) == {"session_id": _SESSION_ID, "task": "t"}


@pytest.fixture
def isolated_shutdown_manager(monkeypatch):
    """The ShutdownManager singleton, with only this test's sessions registered."""
    manager = ShutdownManager()
    monkeypatch.setattr(manager, "_clients", {})
    monkeypatch.setattr(manager, "active_sessions", {})
    monkeypatch.setattr(manager, "is_shutting_down", False)
    monkeypatch.setattr(manager, "shutdown_complete", threading.Event())
    return manager


class TestShutdownEndsWithNoVerdict:
    @respx.mock
    def test_end_session_finishes_only(self, client, isolated_shutdown_manager):
        route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
        state = SessionState(session_id=_SESSION_ID, http_client=client._resources)
        isolated_shutdown_manager._end_session(_SESSION_ID, state)
        assert _body(route) == {"session_id": _SESSION_ID, "is_finished": True}

    @respx.mock
    def test_perform_shutdown_finishes_open_sessions(self, client, isolated_shutdown_manager):
        route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
        manager = isolated_shutdown_manager
        manager.active_sessions[_SESSION_ID] = SessionState(
            session_id=_SESSION_ID, http_client=client._resources)
        manager._perform_shutdown()
        assert route.call_count == 1
        assert _body(route) == {"session_id": _SESSION_ID, "is_finished": True}
        assert manager.active_sessions == {}
