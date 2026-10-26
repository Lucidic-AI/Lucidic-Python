"""Ending and updating a session: no session verdict, and the session's emits first.

A session's result is an eval, ``client.evals.emit(result, name=..., description=...)``.
The session methods no longer take ``is_successful``, ``is_successful_reason``,
``session_eval`` or ``session_eval_reason``, and ``update(**updates)`` refuses them.
``end()`` waits (bounded) for that session's pending emits before it finishes the
session, so the backend has the result at the end. The backend is mocked at the HTTP
boundary with respx; slow or held eval posts make the ordering deterministic.
"""
import json
import logging
import threading
import time

import httpx
import pytest
import respx

from lucidicai import LucidicAI, Session
from lucidicai.sdk import session as sdk_session
from lucidicai.sdk.context import current_client, current_session_id
from lucidicai.sdk.shutdown_manager import SessionState, ShutdownManager

_BASE_URL = "https://stub.lucidic.test"
_UPDATESESSION = f"{_BASE_URL}/updatesession"
_INITSESSION = f"{_BASE_URL}/initsession"
_EVALS = f"{_BASE_URL}/sdk/evals"
_SESSION_ID = "11111111-2222-3333-4444-555555555555"
_OTHER_SESSION_ID = "99999999-8888-7777-6666-555555555555"
_SLOW_POST_SECONDS = 0.3
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


# ---- end() waits for the session's pending emits -------------------------------

def _make_client(**kwargs):
    return LucidicAI(api_key="test-key", agent_id="aaaa1111-0000-0000-0000-000000000000",
                     production=False, **kwargs)


def _routes(order, eval_delay=_SLOW_POST_SECONDS, hold=None):
    """Mock the session and eval endpoints, recording the order requests finish in.

    An eval post takes ``eval_delay`` seconds, or waits for ``hold`` (an Event) if given.
    """
    def _eval(request):
        if hold is not None:
            hold.wait(5)
        else:
            time.sleep(eval_delay)
        order.append(("eval", json.loads(request.read())["session_id"]))
        return httpx.Response(201, json={})

    def _update(request):
        body = json.loads(request.read())
        order.append(("end" if body.get("is_finished") else "update", body["session_id"]))
        return httpx.Response(200, json={})

    respx.post(_INITSESSION).mock(
        return_value=httpx.Response(201, json={"session_id": _SESSION_ID}))
    respx.post(_EVALS).mock(side_effect=_eval)
    return respx.put(_UPDATESESSION).mock(side_effect=_update)


class TestEndPostsTheSessionsEmitsFirst:
    @respx.mock
    def test_end(self, client):
        order = []
        _routes(order)
        client.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
        client.sessions.end(_SESSION_ID)
        assert order == [("eval", _SESSION_ID), ("end", _SESSION_ID)]

    @pytest.mark.asyncio
    async def test_aend(self, client):
        order = []
        with respx.mock:
            _routes(order)
            client.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
            await client.sessions.aend(_SESSION_ID)
        assert order == [("eval", _SESSION_ID), ("end", _SESSION_ID)]

    @respx.mock
    def test_end_from_context_flushes_the_context_session(self, client):
        order = []
        _routes(order)
        session = client.sessions.create(session_name="case", session_id=_SESSION_ID)
        client.evals.emit(0.9, name="score")  # session from context
        client.sessions.end()
        assert order == [("eval", _SESSION_ID), ("end", _SESSION_ID)]
        session._unbind_context()

    @respx.mock
    def test_session_end(self, client):
        order = []
        _routes(order)
        session = Session(client=client, session_id=_SESSION_ID, auto_end=False)
        client.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
        session.end()
        assert order == [("eval", _SESSION_ID), ("end", _SESSION_ID)]

    @respx.mock
    def test_with_block_exit(self, client):
        order = []
        _routes(order)
        with client.sessions.create(session_name="case", session_id=_SESSION_ID):
            client.evals.emit(True, name="task_passed")
        assert order == [("eval", _SESSION_ID), ("end", _SESSION_ID)]

    @pytest.mark.asyncio
    async def test_async_with_block_exit(self, client):
        order = []
        with respx.mock:
            _routes(order)
            async with await client.sessions.acreate(session_name="case", session_id=_SESSION_ID):
                client.evals.emit(True, name="task_passed")
        assert order == [("eval", _SESSION_ID), ("end", _SESSION_ID)]

    @respx.mock
    def test_does_not_wait_for_another_sessions_emits(self, client):
        order = []
        hold = threading.Event()
        _routes(order, hold=hold)
        client.evals.emit(True, name="task_passed", session_id=_OTHER_SESSION_ID)
        try:
            started = time.monotonic()
            client.sessions.end(_SESSION_ID)
            elapsed = time.monotonic() - started
            assert order == [("end", _SESSION_ID)]
        finally:
            hold.set()
        assert elapsed < 1.0  # the default wait is 5 s
        assert client.evals.flush(timeout=5) is True


class TestEndWaitIsBounded:
    @respx.mock
    def test_timeout_warns_and_ends_anyway(self, client, caplog):
        order = []
        hold = threading.Event()
        update = _routes(order, hold=hold)
        client.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
        try:
            with caplog.at_level(logging.WARNING, logger="Lucidic"):
                started = time.monotonic()
                client.sessions.end(_SESSION_ID, eval_flush_timeout=0.2)
                elapsed = time.monotonic() - started
            # the session finished while the eval was still in flight
            assert order == [("end", _SESSION_ID)]
            assert update.call_count == 1
        finally:
            hold.set()
        assert client.evals.flush(timeout=5) is True  # the post lands while mocked
        assert 0.15 <= elapsed < 1.0
        assert "still in flight for session" in caplog.text

    @pytest.mark.asyncio
    async def test_aend_timeout_warns_and_ends_anyway(self, client, caplog):
        order = []
        hold = threading.Event()
        with respx.mock:
            _routes(order, hold=hold)
            client.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
            try:
                with caplog.at_level(logging.WARNING, logger="Lucidic"):
                    started = time.monotonic()
                    await client.sessions.aend(_SESSION_ID, eval_flush_timeout=0.2)
                    elapsed = time.monotonic() - started
                assert order == [("end", _SESSION_ID)]
            finally:
                hold.set()
            assert client.evals.flush(timeout=5) is True
        assert 0.15 <= elapsed < 1.0
        assert "still in flight for session" in caplog.text

    @respx.mock
    def test_the_client_setting_is_the_default(self):
        c = _make_client(eval_flush_timeout=0.2)
        order = []
        hold = threading.Event()
        _routes(order, hold=hold)
        try:
            c.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
            started = time.monotonic()
            c.sessions.end(_SESSION_ID)
            elapsed = time.monotonic() - started
        finally:
            hold.set()
            assert c.evals.flush(timeout=5) is True
            c.close()
        assert 0.15 <= elapsed < 1.0

    @respx.mock
    def test_session_end_keyword(self, client):
        order = []
        hold = threading.Event()
        _routes(order, hold=hold)
        session = Session(client=client, session_id=_SESSION_ID, auto_end=False)
        client.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
        try:
            started = time.monotonic()
            session.end(eval_flush_timeout=0.2)
            elapsed = time.monotonic() - started
        finally:
            hold.set()
        assert client.evals.flush(timeout=5) is True
        assert 0.15 <= elapsed < 1.0
        assert session.is_finished

    @respx.mock
    def test_zero_skips_the_wait(self, client):
        order = []
        hold = threading.Event()
        _routes(order, hold=hold)
        client.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
        try:
            started = time.monotonic()
            client.sessions.end(_SESSION_ID, eval_flush_timeout=0)
            elapsed = time.monotonic() - started
            assert order == [("end", _SESSION_ID)]
        finally:
            hold.set()
        assert client.evals.flush(timeout=5) is True
        assert elapsed < 0.15

    @respx.mock
    @pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), "5", True])
    def test_invalid_timeout_is_refused_before_any_request(self, client, bad):
        update = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
        with pytest.raises(ValueError):
            client.sessions.end(_SESSION_ID, eval_flush_timeout=bad)
        assert not update.called

    @respx.mock
    def test_close_waits_once_not_once_more_per_session(self):
        c = _make_client(eval_flush_timeout=0.2)
        order = []
        hold = threading.Event()
        _routes(order, hold=hold)
        try:
            c.sessions.create(session_name="case", session_id=_SESSION_ID)
            c.evals.emit(True, name="task_passed")
            started = time.monotonic()
            c.close()
            elapsed = time.monotonic() - started
        finally:
            hold.set()
            current_session_id.set(None)
        assert c.evals.flush(timeout=5) is True
        # close() waits its 0.2 s for every session's evals, then ends each
        # session without a second wait; the held eval lands after the end.
        assert 0.15 <= elapsed < 0.35
        assert order == [("end", _SESSION_ID), ("eval", _SESSION_ID)]


class TestFlushOneSession:
    @respx.mock
    def test_waits_only_for_that_sessions_posts(self, client):
        order = []
        hold = threading.Event()
        _routes(order, hold=hold)
        client.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
        try:
            assert client.evals.flush(timeout=0.1, session_id=_OTHER_SESSION_ID) is True
            assert client.evals.flush(timeout=0.1, session_id=_SESSION_ID) is False
        finally:
            hold.set()
        assert client.evals.flush(timeout=5, session_id=_SESSION_ID) is True
