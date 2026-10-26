"""evals.emit() posts from a daemon thread; close() must not drop it.

Before the fix, ``LucidicAI.close()`` returned without waiting for pending
emit threads, so a process exiting right after ``client.evals.emit(...)``
could lose the eval silently (found 2026-10-12). These tests mock the
backend at the HTTP boundary with respx and slow the eval POST down so the
race is deterministic.
"""
import asyncio
import threading
import time

import httpx
import pytest
import respx

from lucidicai import LucidicAI
from lucidicai.sdk.context import current_client, current_session_id

_BASE_URL = "https://stub.lucidic.test"
_EVALS_ENDPOINT = f"{_BASE_URL}/sdk/evals"
_INITSESSION_ENDPOINT = f"{_BASE_URL}/initsession"
_UPDATESESSION_ENDPOINT = f"{_BASE_URL}/updatesession"
_SLOW_POST_SECONDS = 0.3
_SESSION_ID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def _isolate_session_context():
    # sessions.create binds current_client/current_session_id and close() does
    # not unbind them; reset so a closed client never leaks into later tests.
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
    c = LucidicAI(
        api_key="test-key",
        agent_id="aaaa1111-0000-0000-0000-000000000000",
        production=True,
    )
    yield c
    c.close()


def _slow_eval_route(delivered, delay=_SLOW_POST_SECONDS, status=201):
    """An eval POST that takes ``delay`` seconds, then records its body."""

    def _respond(request):
        time.sleep(delay)
        delivered.append(request.read())
        return httpx.Response(status, json={})

    return respx.post(_EVALS_ENDPOINT).mock(side_effect=_respond)


@respx.mock
def test_close_waits_for_an_in_flight_emit(client):
    delivered = []
    _slow_eval_route(delivered)

    client.evals.emit(result=0.75, name="accuracy", session_id=_SESSION_ID)
    client.close()

    # the eval reached the backend before close() returned.
    assert len(delivered) == 1
    assert b'"result":0.75' in delivered[0].replace(b" ", b"")


@pytest.mark.asyncio
async def test_aclose_waits_for_an_in_flight_emit(client):
    delivered = []
    with respx.mock:
        _slow_eval_route(delivered)
        client.evals.emit(result=1, name="score", session_id=_SESSION_ID)
        await client.aclose()
    assert len(delivered) == 1


@respx.mock
def test_close_posts_pending_evals_before_ending_open_sessions(client):
    order = []
    respx.post(_INITSESSION_ENDPOINT).mock(
        return_value=httpx.Response(201, json={"session_id": _SESSION_ID})
    )

    def _eval(request):
        time.sleep(_SLOW_POST_SECONDS)
        order.append("eval")
        return httpx.Response(201, json={})

    def _end(request):
        order.append("end")
        return httpx.Response(200, json={})

    respx.post(_EVALS_ENDPOINT).mock(side_effect=_eval)
    respx.put(_UPDATESESSION_ENDPOINT).mock(side_effect=_end)

    client.sessions.create(session_name="case", session_id=_SESSION_ID)
    client.evals.emit(result=0.5, name="score")
    client.close()

    assert order == ["eval", "end"]


@respx.mock
def test_flush_is_bounded_by_its_timeout(client):
    release = threading.Event()

    def _hang(request):
        release.wait(5)
        return httpx.Response(201, json={})

    respx.post(_EVALS_ENDPOINT).mock(side_effect=_hang)
    client.evals.emit(result=0.1, name="score", session_id=_SESSION_ID)
    try:
        started = time.monotonic()
        drained = client.evals.flush(timeout=0.2)
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert drained is False
    # floor: it really waited; ceiling: it did not wait for the post.
    assert elapsed >= 0.15
    assert elapsed < 2.0


@respx.mock
def test_flush_timeout_bounds_the_whole_wait_not_each_post(client):
    release = threading.Event()

    def _hang(request):
        release.wait(5)
        return httpx.Response(201, json={})

    respx.post(_EVALS_ENDPOINT).mock(side_effect=_hang)
    for score in (0.1, 0.2, 0.3):
        client.evals.emit(result=score, name="score", session_id=_SESSION_ID)
    try:
        started = time.monotonic()
        drained = client.evals.flush(timeout=0.3)
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert drained is False
    # three hung posts share one 0.3 s budget (0.9 s if each got its own).
    assert elapsed >= 0.25
    assert elapsed < 0.6


@respx.mock
def test_flush_reports_drained_once_the_post_finishes(client):
    delivered = []
    _slow_eval_route(delivered)
    client.evals.emit(result=3, name="score", session_id=_SESSION_ID)

    assert client.evals.flush(timeout=5) is True
    assert len(delivered) == 1


@respx.mock
def test_failed_post_does_not_leave_the_emit_pending(client):
    delivered = []
    _slow_eval_route(delivered, delay=0.05, status=500)
    client.evals.emit(result=2, name="score", session_id=_SESSION_ID)

    assert client.evals.flush(timeout=5) is True
    # a second flush has nothing left to wait for.
    started = time.monotonic()
    assert client.evals.flush(timeout=5) is True
    assert time.monotonic() - started < 1.0


def test_an_emit_whose_thread_cannot_start_is_not_left_pending(client, monkeypatch):
    def _refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", _refuse)
    with pytest.raises(RuntimeError):
        client.evals.emit(result=1.0, name="score", session_id=_SESSION_ID)
    monkeypatch.undo()

    assert client.evals.flush(timeout=1) is True


def test_flush_with_nothing_pending_returns_at_once(client):
    started = time.monotonic()
    assert client.evals.flush(timeout=5) is True
    assert time.monotonic() - started < 1.0


# ---- configurable timeout + shutdown budget (verify-sdk follow-up) -----------

def _hang_evals(release):
    def _hang(request):
        release.wait(5)
        return httpx.Response(201, json={})

    return respx.post(_EVALS_ENDPOINT).mock(side_effect=_hang)


def _make_client(**kwargs):
    return LucidicAI(api_key="test-key", agent_id="aaaa1111-0000-0000-0000-000000000000",
                     production=True, **kwargs)


@respx.mock
def test_client_setting_bounds_the_close_wait():
    release = threading.Event()
    _hang_evals(release)
    c = _make_client(eval_flush_timeout=0.2)
    c.evals.emit(result=0.4, name="score", session_id=_SESSION_ID)
    try:
        started = time.monotonic()
        c.close()
        elapsed = time.monotonic() - started
    finally:
        release.set()
    assert 0.15 <= elapsed < 1.0


@respx.mock
def test_close_keyword_overrides_the_client_setting(client):
    release = threading.Event()
    _hang_evals(release)
    client.evals.emit(result=0.4, name="score", session_id=_SESSION_ID)
    try:
        started = time.monotonic()
        client.close(eval_flush_timeout=0.2)
        elapsed = time.monotonic() - started
    finally:
        release.set()
    # the client default is 5 s; the keyword wins.
    assert 0.15 <= elapsed < 1.0


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), "5", True])
def test_invalid_eval_flush_timeout_is_rejected(bad):
    with pytest.raises(ValueError):
        LucidicAI(api_key="test-key", agent_id="aaaa1111-0000-0000-0000-000000000000",
                  production=False, eval_flush_timeout=bad)


@pytest.fixture
def isolated_shutdown_manager(monkeypatch):
    """The ShutdownManager singleton, with only this test's clients registered."""
    from lucidicai.sdk.shutdown_manager import ShutdownManager

    manager = ShutdownManager()
    monkeypatch.setattr(manager, "_clients", {})
    monkeypatch.setattr(manager, "active_sessions", {})
    monkeypatch.setattr(manager, "shutdown_complete", threading.Event())
    return manager


@respx.mock
def test_shutdown_shares_one_eval_flush_deadline_across_clients(isolated_shutdown_manager, monkeypatch):
    manager = isolated_shutdown_manager
    monkeypatch.setattr(manager, "eval_flush_budget", 0.3)
    release = threading.Event()
    _hang_evals(release)
    clients = [_make_client() for _ in range(3)]
    for c in clients:
        c.evals.emit(result=0.5, name="score", session_id=_SESSION_ID)
    try:
        started = time.monotonic()
        manager._perform_shutdown()
        elapsed = time.monotonic() - started
    finally:
        release.set()
    # three hung clients share one 0.3 s budget (0.9 s if each got its own,
    # 15 s with the 5 s client default).
    assert elapsed >= 0.25
    assert elapsed < 0.6
    assert manager._clients == {}


@respx.mock
def test_shutdown_never_waits_longer_than_a_clients_own_setting(isolated_shutdown_manager):
    release = threading.Event()
    _hang_evals(release)
    c = _make_client(eval_flush_timeout=0.2)
    c.evals.emit(result=0.5, name="score", session_id=_SESSION_ID)
    try:
        started = time.monotonic()
        isolated_shutdown_manager._perform_shutdown()
        elapsed = time.monotonic() - started
    finally:
        release.set()
    # the shared budget is 5 s; the client's own 0.2 s is the tighter bound.
    assert 0.15 <= elapsed < 1.0


@pytest.mark.asyncio
async def test_aclose_warns_when_the_flush_is_skipped(client, monkeypatch, caplog):
    import logging

    loop = asyncio.get_running_loop()

    def _refuse(*args, **kwargs):
        raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(loop, "run_in_executor", _refuse)
    caplog.set_level(logging.DEBUG, logger="Lucidic")
    await client.aclose()

    skipped = [r for r in caplog.records if "flush" in r.getMessage().lower()
               and "cannot schedule" in r.getMessage()]
    assert skipped and all(r.levelno == logging.WARNING for r in skipped)


@pytest.mark.asyncio
async def test_aclose_keyword_bounds_the_wait(client):
    release = threading.Event()
    with respx.mock:
        _hang_evals(release)
        client.evals.emit(result=0.4, name="score", session_id=_SESSION_ID)
        try:
            started = time.monotonic()
            await client.aclose(eval_flush_timeout=0.2)
            elapsed = time.monotonic() - started
        finally:
            release.set()
    assert 0.15 <= elapsed < 1.0


def test_close_rejects_an_invalid_keyword_timeout(client):
    with pytest.raises(ValueError):
        client.close(eval_flush_timeout=-1)
