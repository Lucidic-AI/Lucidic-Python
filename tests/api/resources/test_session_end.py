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


def _spy_flush(monkeypatch, c):
    """Record the session_id of every evals.flush call on client ``c``; returns the list."""
    calls = []
    real_flush = c.evals.flush

    def _flush(timeout=5.0, session_id=None):
        calls.append(session_id)
        return real_flush(timeout, session_id=session_id)

    monkeypatch.setattr(c.evals, "flush", _flush)
    return calls


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
    def test_close_waits_once_not_once_more_per_session(self, monkeypatch):
        c = _make_client(eval_flush_timeout=0.2)
        order = []
        hold = threading.Event()
        _routes(order, hold=hold)
        flushes, end_timeouts = _spy_flush(monkeypatch, c), []
        real_end = c.sessions.end

        def _end(session_id=None, **kwargs):
            end_timeouts.append(kwargs.get("eval_flush_timeout"))
            return real_end(session_id, **kwargs)

        monkeypatch.setattr(c.sessions, "end", _end)
        try:
            c.sessions.create(session_name="case", session_id=_SESSION_ID)
            c.evals.emit(True, name="task_passed")
            started = time.monotonic()
            c.close()
            elapsed = time.monotonic() - started
            close_flushes = list(flushes)
        finally:
            hold.set()
            current_session_id.set(None)
        assert c.evals.flush(timeout=5) is True
        # one wait for every session's evals, then each session ends without one
        assert close_flushes == [None]
        assert end_timeouts == [0]
        assert order == [("end", _SESSION_ID), ("eval", _SESSION_ID)]
        assert elapsed < 1.0  # smoke bound only


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


# ---- the helpful TypeError on every end/update ---------------------------------

class TestVerdictKwargsPointAtEmit:
    @respx.mock
    @pytest.mark.parametrize("call", [
        lambda c, s: c.sessions.end(_SESSION_ID, is_successful=True),
        lambda c, s: s.update(is_successful=True),
        lambda c, s: s.end(session_eval=1.0),
    ])
    def test_sync(self, client, call):
        route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
        session = Session(client=client, session_id=_SESSION_ID, auto_end=False)
        with pytest.raises(TypeError, match=r"no longer accepts .*client\.evals\.emit\(result, name="):
            call(client, session)
        assert not route.called
        assert not session.is_finished

    @pytest.mark.asyncio
    @pytest.mark.parametrize("call", [
        lambda c, s: c.sessions.aend(_SESSION_ID, is_successful=True),
        lambda c, s: s.aupdate(is_successful_reason="ok"),
        lambda c, s: s.aend(session_eval_reason="ok"),
    ])
    async def test_async(self, client, call):
        session = Session(client=client, session_id=_SESSION_ID, auto_end=False)
        with respx.mock:
            route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(TypeError, match=r"no longer accepts .*client\.evals\.emit"):
                await call(client, session)
        assert not route.called

    @respx.mock
    @pytest.mark.parametrize("call, method", [
        (lambda c, s: c.sessions.end(_SESSION_ID, sesion_id="typo"), "sessions.end"),
        (lambda c, s: s.update(taks="typo"), "Session.update"),
        (lambda c, s: s.end(timeout=1), "Session.end"),
    ])
    def test_other_unknown_keywords_still_raise(self, client, call, method):
        route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))
        session = Session(client=client, session_id=_SESSION_ID, auto_end=False)
        with pytest.raises(TypeError, match=rf"{method}\(\) got an unexpected keyword argument"):
            call(client, session)
        assert not route.called


# ---- concurrent ends and emits -------------------------------------------------

class TestConcurrentEnds:
    @respx.mock
    def test_end_waits_for_its_own_posts_among_concurrent_emits(self, client):
        order = []
        hold_b = threading.Event()

        def _eval(request):
            sid = json.loads(request.read())["session_id"]
            if sid == _OTHER_SESSION_ID:
                hold_b.wait(5)
            else:
                time.sleep(0.05)
            order.append(("eval", sid))
            return httpx.Response(201, json={})

        respx.post(_EVALS).mock(side_effect=_eval)
        respx.put(_UPDATESESSION).mock(side_effect=lambda request: order.append(
            ("end", json.loads(request.read())["session_id"])) or httpx.Response(200, json={}))
        sids = [_SESSION_ID] * 4 + [_OTHER_SESSION_ID] * 4
        barrier = threading.Barrier(len(sids))

        def _emit(sid):
            barrier.wait()
            client.evals.emit(True, name="task_passed", session_id=sid)

        emitters = [threading.Thread(target=_emit, args=(sid,)) for sid in sids]
        for t in emitters:
            t.start()
        for t in emitters:
            t.join(5)
        try:
            client.sessions.end(_SESSION_ID)
            seen = list(order)
            still_pending = sorted(client.evals._pending.values())
        finally:
            hold_b.set()
        assert client.evals.flush(timeout=5) is True
        assert seen == [("eval", _SESSION_ID)] * 4 + [("end", _SESSION_ID)]
        assert still_pending == [_OTHER_SESSION_ID] * 4

    @pytest.mark.asyncio
    async def test_aend_does_not_wait_for_another_sessions_emits(self, client):
        order = []
        hold = threading.Event()
        with respx.mock:
            _routes(order, hold=hold)
            client.evals.emit(True, name="task_passed", session_id=_OTHER_SESSION_ID)
            try:
                await client.sessions.aend(_SESSION_ID)
                assert order == [("end", _SESSION_ID)]
                assert list(client.evals._pending.values()) == [_OTHER_SESSION_ID]
            finally:
                hold.set()
            assert client.evals.flush(timeout=5) is True

    @respx.mock
    def test_two_sessions_ending_at_once_each_wait_for_their_own(self, client):
        order = []
        holds = {_SESSION_ID: threading.Event(), _OTHER_SESSION_ID: threading.Event()}

        def _eval(request):
            sid = json.loads(request.read())["session_id"]
            holds[sid].wait(5)
            order.append(("eval", sid))
            return httpx.Response(201, json={})

        respx.post(_EVALS).mock(side_effect=_eval)
        respx.put(_UPDATESESSION).mock(side_effect=lambda request: order.append(
            ("end", json.loads(request.read())["session_id"])) or httpx.Response(200, json={}))
        for sid in holds:
            client.evals.emit(True, name="task_passed", session_id=sid)
        enders = {sid: threading.Thread(target=client.sessions.end, args=(sid,)) for sid in holds}
        try:
            for t in enders.values():
                t.start()
            holds[_OTHER_SESSION_ID].set()
            enders[_OTHER_SESSION_ID].join(5)
            # B ended once its own post landed; A is still waiting for A's
            assert order == [("eval", _OTHER_SESSION_ID), ("end", _OTHER_SESSION_ID)]
            assert enders[_SESSION_ID].is_alive()
        finally:
            for hold in holds.values():
                hold.set()
            for t in enders.values():
                t.join(5)
        assert order[2:] == [("eval", _SESSION_ID), ("end", _SESSION_ID)]

    @pytest.mark.asyncio
    async def test_concurrent_aends_end_on_time_with_the_executor_busy(self, client):
        import asyncio
        import concurrent.futures

        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(executor)
        busy, release = threading.Event(), threading.Event()
        blocker = loop.run_in_executor(None, lambda: (busy.set(), release.wait(5)))
        sids = [f"{n:08d}-0000-4000-8000-000000000000" for n in range(4)]
        order = []
        with respx.mock:
            update = _routes(order, hold=release)
            for sid in sids:
                client.evals.emit(True, name="task_passed", session_id=sid)

            async def _timed(sid):
                started = time.monotonic()
                await client.sessions.aend(sid, eval_flush_timeout=0.2)
                return time.monotonic() - started

            try:
                assert busy.wait(2)  # the executor's only worker is taken
                took = await asyncio.gather(*(_timed(sid) for sid in sids))
            finally:
                release.set()
            await blocker
            assert client.evals.flush(timeout=5) is True
        executor.shutdown(wait=True)
        assert update.call_count == len(sids)
        # each ended after its own 0.2 s, none queued behind the busy worker (5 s)
        assert max(took) < 1.0


# ---- edge cases of the wait ----------------------------------------------------

class TestWaitEdgeCases:
    @respx.mock
    def test_a_client_set_to_zero_neither_waits_nor_warns(self, monkeypatch, caplog):
        c = _make_client(eval_flush_timeout=0)
        order = []
        hold = threading.Event()
        _routes(order, hold=hold)
        flushes = _spy_flush(monkeypatch, c)
        try:
            c.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
            with caplog.at_level(logging.WARNING, logger="Lucidic"):
                c.sessions.end(_SESSION_ID)
            assert order == [("end", _SESSION_ID)]
            assert flushes == []  # end() did not wait at all
        finally:
            hold.set()
        assert c.evals.flush(timeout=5) is True
        c.close(eval_flush_timeout=0)
        assert "still in flight" not in caplog.text

    @pytest.mark.asyncio
    async def test_a_client_set_to_zero_aend_neither_waits_nor_warns(self, caplog):
        c = _make_client(eval_flush_timeout=0)
        order = []
        hold = threading.Event()
        with respx.mock:
            _routes(order, hold=hold)
            c.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
            try:
                with caplog.at_level(logging.WARNING, logger="Lucidic"):
                    await c.sessions.aend(_SESSION_ID)
                assert order == [("end", _SESSION_ID)]
            finally:
                hold.set()
            assert c.evals.flush(timeout=5) is True
        c.close(eval_flush_timeout=0)
        assert "still in flight" not in caplog.text

    @pytest.mark.asyncio
    async def test_a_cancelled_aend_still_ends_the_session(self, client):
        import asyncio

        order = []
        hold = threading.Event()
        with respx.mock:
            update = _routes(order, hold=hold)
            session = await client.sessions.acreate(session_name="case", session_id=_SESSION_ID)
            client.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
            task = asyncio.ensure_future(client.sessions.aend(_SESSION_ID))
            try:
                await asyncio.sleep(0.1)  # aend is waiting for the held post
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert update.call_count == 1 and order == [("end", _SESSION_ID)]
                assert _SESSION_ID not in client._sessions
            finally:
                hold.set()
                session._unbind_context()
            assert client.evals.flush(timeout=5) is True


# ---- finish() waits for the listed sessions' emits --------------------------------

_FINISH = f"{_BASE_URL}/sdk/v2/sessions/finish"


def _finish_route(order):
    def _finish(request):
        order.append(("finish", tuple(json.loads(request.read())["session_ids"])))
        return httpx.Response(200, json={"num_sessions_finished": 2})

    return respx.post(_FINISH).mock(side_effect=_finish)


class TestFinishWaitsForTheSessionsEmits:
    @respx.mock
    def test_finish(self, client, monkeypatch):
        order = []
        _routes(order)
        _finish_route(order)
        flushes = _spy_flush(monkeypatch, client)
        client.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
        client.evals.emit(False, name="task_passed", session_id=_OTHER_SESSION_ID)
        assert client.sessions.finish([_SESSION_ID, _OTHER_SESSION_ID]) == 2
        assert sorted(order[:2]) == [("eval", _SESSION_ID), ("eval", _OTHER_SESSION_ID)]
        assert order[2] == ("finish", (_SESSION_ID, _OTHER_SESSION_ID))
        # one wait, under one deadline, for both sessions
        assert flushes == [[_SESSION_ID, _OTHER_SESSION_ID]]

    @pytest.mark.asyncio
    async def test_afinish(self, client):
        order = []
        with respx.mock:
            _routes(order)
            _finish_route(order)
            client.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
            assert await client.sessions.afinish([_SESSION_ID]) == 2
        assert order == [("eval", _SESSION_ID), ("finish", (_SESSION_ID,))]

    @respx.mock
    def test_does_not_wait_for_unlisted_sessions(self, client):
        order = []
        hold = threading.Event()
        _routes(order, hold=hold)
        _finish_route(order)
        client.evals.emit(True, name="task_passed", session_id=_OTHER_SESSION_ID)
        try:
            client.sessions.finish([_SESSION_ID])
            assert order == [("finish", (_SESSION_ID,))]
        finally:
            hold.set()
        assert client.evals.flush(timeout=5) is True

    @respx.mock
    def test_zero_skips_the_wait(self, client, monkeypatch):
        order = []
        hold = threading.Event()
        _routes(order, hold=hold)
        _finish_route(order)
        flushes = _spy_flush(monkeypatch, client)
        client.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
        try:
            client.sessions.finish([_SESSION_ID], eval_flush_timeout=0)
            assert order == [("finish", (_SESSION_ID,))]
            assert flushes == []
        finally:
            hold.set()
        assert client.evals.flush(timeout=5) is True

    @respx.mock
    @pytest.mark.parametrize("bad", [-1, float("nan"), "5", True])
    def test_invalid_timeout_is_refused_before_any_request(self, client, bad):
        route = respx.post(_FINISH).mock(return_value=httpx.Response(200, json={}))
        with pytest.raises(ValueError):
            client.sessions.finish([_SESSION_ID], eval_flush_timeout=bad)
        assert not route.called


# ---- shutdown waits for a session's emits before finishing it ----------------------

class TestShutdownWaitsForTheSessionsEmits:
    @respx.mock
    def test_a_pending_emit_lands_first(self, client, isolated_shutdown_manager):
        order = []
        _routes(order)
        manager = isolated_shutdown_manager
        manager.active_sessions[_SESSION_ID] = SessionState(
            session_id=_SESSION_ID, http_client=client._resources)
        client.evals.emit(True, name="task_passed", session_id=_SESSION_ID)
        manager._perform_shutdown()
        assert order == [("eval", _SESSION_ID), ("end", _SESSION_ID)]

    @respx.mock
    def test_the_wait_is_bounded_by_the_shutdown_budget(self, client, isolated_shutdown_manager, monkeypatch):
        order = []
        hold = threading.Event()
        _routes(order, hold=hold)
        manager = isolated_shutdown_manager
        monkeypatch.setattr(manager, "eval_flush_budget", 0.2)
        for sid in (_SESSION_ID, _OTHER_SESSION_ID):
            manager.active_sessions[sid] = SessionState(session_id=sid, http_client=client._resources)
            client.evals.emit(True, name="task_passed", session_id=sid)
        try:
            started = time.monotonic()
            manager._perform_shutdown()
            elapsed = time.monotonic() - started
            # both sessions finished, though their evals were still in flight
            assert sorted(order) == [("end", _SESSION_ID), ("end", _OTHER_SESSION_ID)]
        finally:
            hold.set()
        assert client.evals.flush(timeout=5) is True
        assert elapsed < 1.0  # one 0.2 s budget for both sessions

    @respx.mock
    def test_a_failing_flush_never_raises(self, client, isolated_shutdown_manager, monkeypatch):
        route = respx.put(_UPDATESESSION).mock(return_value=httpx.Response(200, json={}))

        def _broken(*args, **kwargs):
            raise RuntimeError("flush failed")

        state = SessionState(session_id=_SESSION_ID, http_client=client._resources)
        with monkeypatch.context() as patched:
            patched.setattr(client, "_flush_evals", _broken)
            isolated_shutdown_manager._end_session(_SESSION_ID, state)
        assert route.call_count == 1
