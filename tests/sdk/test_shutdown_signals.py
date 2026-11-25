"""Signal and excepthook listeners, and the span flush before end/close.

The listener tests use a fresh ShutdownManager with ``_handle_shutdown`` recorded,
and restore the process's SIGINT/SIGTERM handlers and ``sys.excepthook`` afterwards.
The flush tests record ``force_flush`` against the HTTP calls (respx).
"""
import asyncio
import atexit
import json
import logging
import signal
import sys
import threading
import time

import httpx
import pytest
import respx

from lucidicai import LucidicAI
from lucidicai.sdk.context import current_client, current_session_id
from lucidicai.sdk.shutdown_manager import ShutdownManager
from lucidicai.telemetry import telemetry_manager as tm
from lucidicai.telemetry.telemetry_manager import TelemetryManager

_SIGNALS = (signal.SIGINT, signal.SIGTERM)
_BASE_URL = "https://stub.lucidic.test"
_UPDATESESSION = f"{_BASE_URL}/updatesession"
_INITSESSION = f"{_BASE_URL}/initsession"
_SESSION_ID = "11111111-2222-3333-4444-555555555555"
_AGENT_ID = "aaaa1111-0000-0000-0000-000000000000"
_SLOW_FLUSH_SECONDS = 1.0
_MAX_LOOP_GAP_SECONDS = 0.2


@pytest.fixture(autouse=True)
def _restore_process_hooks():
    handlers = {s: signal.getsignal(s) for s in _SIGNALS}
    hook = sys.excepthook
    yield
    for s, h in handlers.items():
        signal.signal(s, h)
    sys.excepthook = hook


@pytest.fixture
def calls():
    return []


@pytest.fixture
def manager(monkeypatch, calls):
    monkeypatch.setattr(ShutdownManager, "_instance", None)
    monkeypatch.setattr(atexit, "register", lambda fn: calls.append(("atexit", fn)))
    mgr = ShutdownManager()
    monkeypatch.setattr(mgr, "_handle_shutdown", lambda trigger: calls.append(("shutdown", trigger)))
    return mgr


@pytest.fixture
def no_raise_signal(monkeypatch, calls):
    monkeypatch.setattr(signal, "raise_signal", lambda s: calls.append(("raise_signal", s)))


def _install(manager, previous):
    for s in _SIGNALS:
        signal.signal(s, previous)
    manager._ensure_listeners_registered()


class TestInstall:
    def test_off_the_main_thread_registers_atexit_but_not_signals(self, manager, calls):
        before = {s: signal.getsignal(s) for s in _SIGNALS}
        errors = []

        def register():
            try:
                manager._ensure_listeners_registered()
            except Exception as e:
                errors.append(e)

        thread = threading.Thread(target=register)
        thread.start()
        thread.join()
        assert errors == []
        assert {s: signal.getsignal(s) for s in _SIGNALS} == before
        assert [c[0] for c in calls] == ["atexit"]

        manager._ensure_listeners_registered()
        assert all(signal.getsignal(s) == manager._signal_handler for s in _SIGNALS)
        assert [c[0] for c in calls] == ["atexit"]  # once per process

    @pytest.mark.parametrize("previous", [signal.SIG_DFL, lambda s, f: None])
    def test_installs_over_a_default_or_python_handler(self, manager, previous):
        _install(manager, previous)
        assert all(signal.getsignal(s) == manager._signal_handler for s in _SIGNALS)

    def test_leaves_an_ignored_signal_ignored(self, manager):
        _install(manager, signal.SIG_IGN)
        assert all(signal.getsignal(s) == signal.SIG_IGN for s in _SIGNALS)

    def test_leaves_a_handler_set_outside_python_alone(self, manager, monkeypatch):
        installed = []
        monkeypatch.setattr(signal, "getsignal", lambda s: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: installed.append(s))
        manager._ensure_listeners_registered()
        assert installed == []

    def test_a_repeated_excepthook_capture_does_not_chain_to_itself(self, manager, calls):
        sys.excepthook = lambda *exc: calls.append(("app_hook", exc[0]))
        manager._ensure_listeners_registered()
        manager.listeners_registered = False  # as when two threads register at once
        manager._ensure_listeners_registered()
        sys.excepthook(ValueError, ValueError("boom"), None)
        assert calls[-1] == ("app_hook", ValueError)


class TestSignalHandler:
    @pytest.mark.parametrize("signum", _SIGNALS)
    def test_calls_the_app_handler_after_the_shutdown(self, manager, calls, signum):
        _install(manager, lambda s, f: calls.append(("app", s)))
        manager._signal_handler(signum, None)  # returns: no exit of our own
        assert calls[-2:] == [("shutdown", f"signal_{signum}"), ("app", signum)]

    def test_sigint_with_the_default_handler_raises_keyboard_interrupt(self, manager, calls):
        _install(manager, signal.default_int_handler)
        with pytest.raises(KeyboardInterrupt):
            manager._signal_handler(signal.SIGINT, None)
        assert calls[-1] == ("shutdown", f"signal_{signal.SIGINT}")
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    @pytest.mark.parametrize("signum", _SIGNALS)
    def test_sig_dfl_is_restored_and_the_signal_re_raised(self, manager, calls, no_raise_signal, signum):
        _install(manager, signal.SIG_DFL)
        manager._signal_handler(signum, None)
        assert calls[-2:] == [("shutdown", f"signal_{signum}"), ("raise_signal", signum)]
        assert signal.getsignal(signum) == signal.SIG_DFL

    @pytest.mark.parametrize("signum", _SIGNALS)
    def test_chained_from_a_wrapper_exits_and_leaves_the_wrapper_installed(
        self, manager, calls, no_raise_signal, signum
    ):
        _install(manager, signal.SIG_DFL)
        ours = signal.getsignal(signum)

        def wrapper(s, f):
            ours(s, f)

        signal.signal(signum, wrapper)
        with pytest.raises(SystemExit) as exc:
            wrapper(signum, None)
        assert exc.value.code == 128 + signum
        assert ("raise_signal", signum) not in calls
        assert signal.getsignal(signum) is wrapper

    def test_one_shot_a_second_signal_goes_to_the_previous_handler(self, manager, calls):
        _install(manager, lambda s, f: calls.append(("app", s)))
        signal.raise_signal(signal.SIGTERM)
        signal.raise_signal(signal.SIGTERM)
        assert [c for c in calls if c[0] != "atexit"] == [
            ("shutdown", f"signal_{signal.SIGTERM}"), ("app", signal.SIGTERM), ("app", signal.SIGTERM),
        ]


def test_excepthook_chains_to_the_hook_it_replaced(manager, calls):
    sys.excepthook = lambda *exc: calls.append(("app_hook", exc[0]))
    manager._ensure_listeners_registered()
    sys.excepthook(ValueError, ValueError("boom"), None)
    assert calls[-2:] == [("shutdown", "uncaught_exception"), ("app_hook", ValueError)]


def test_excepthook_passes_keyboard_interrupt_straight_through(manager, calls, monkeypatch, caplog):
    import lucidicai.sdk.event as sdk_event

    monkeypatch.setattr(sdk_event, "create_event", lambda **kw: calls.append(("event", kw["type"])))
    sys.excepthook = lambda *exc: calls.append(("app_hook", exc[0]))
    manager._ensure_listeners_registered()
    with caplog.at_level(logging.ERROR, logger="Lucidic"):
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert [c for c in calls if c[0] != "atexit"] == [("app_hook", KeyboardInterrupt)]
    assert "Uncaught exception" not in caplog.text


class TestFlushSpans:
    def test_warns_when_the_flush_is_incomplete(self, monkeypatch, caplog):
        monkeypatch.setattr(TelemetryManager, "is_telemetry_initialized", property(lambda self: True))
        monkeypatch.setattr(TelemetryManager, "force_flush", lambda self, timeout_millis: False)
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            tm.flush_spans()
        assert "incomplete" in caplog.text

    def test_returns_within_the_timeout_when_the_export_hangs(self, monkeypatch, caplog):
        release = threading.Event()
        monkeypatch.setattr(TelemetryManager, "is_telemetry_initialized", property(lambda self: True))
        monkeypatch.setattr(TelemetryManager, "force_flush", lambda self, timeout_millis: release.wait(10))
        start = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            tm.flush_spans(timeout_millis=200)
        release.set()
        assert time.monotonic() - start < 2
        assert "incomplete" in caplog.text

    def test_skips_while_an_earlier_flush_is_still_stuck(self, monkeypatch, calls):
        release = threading.Event()
        monkeypatch.setattr(TelemetryManager, "is_telemetry_initialized", property(lambda self: True))
        monkeypatch.setattr(TelemetryManager, "force_flush",
                            lambda self, timeout_millis: calls.append("flush") or release.wait(10))
        try:
            tm.flush_spans(timeout_millis=200)
            start = time.monotonic()
            tm.flush_spans(timeout_millis=200)
            assert time.monotonic() - start < 0.1
            assert calls == ["flush"]
        finally:
            release.set()

    def test_concurrent_healthy_flushes_each_run(self, monkeypatch, calls):
        monkeypatch.setattr(TelemetryManager, "is_telemetry_initialized", property(lambda self: True))
        monkeypatch.setattr(TelemetryManager, "force_flush",
                            lambda self, timeout_millis: calls.append("flush") or time.sleep(0.1) or True)
        threads = [threading.Thread(target=tm.flush_spans) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert calls == ["flush", "flush"]

    def test_never_raises(self, monkeypatch):
        def boom(self, timeout_millis):
            raise RuntimeError("exporter down")

        monkeypatch.setattr(TelemetryManager, "is_telemetry_initialized", property(lambda self: True))
        monkeypatch.setattr(TelemetryManager, "force_flush", boom)
        tm.flush_spans()

    def test_skips_when_telemetry_is_off(self, monkeypatch, calls):
        monkeypatch.setattr(TelemetryManager, "is_telemetry_initialized", property(lambda self: False))
        monkeypatch.setattr(TelemetryManager, "force_flush", lambda self, timeout_millis: calls.append("flush"))
        tm.flush_spans()
        assert calls == []


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LUCIDIC_DEBUG", "false")
    monkeypatch.setenv("LUCIDIC_BASE_URL", _BASE_URL)
    monkeypatch.delenv("LUCIDIC_REGION", raising=False)
    client_token = current_client.set(None)
    session_token = current_session_id.set(None)
    c = LucidicAI(api_key="test-key", agent_id=_AGENT_ID, production=False)
    yield c
    c.close()
    current_session_id.reset(session_token)
    current_client.reset(client_token)


@pytest.fixture
def backend(calls):
    with respx.mock:
        respx.post(_INITSESSION).mock(side_effect=lambda request: httpx.Response(
            200, json={"session_id": json.loads(request.content).get("session_id", _SESSION_ID)}))
        respx.put(_UPDATESESSION).mock(
            side_effect=lambda request: calls.append(("finish",)) or httpx.Response(200, json={}))
        yield


def _recorded(calls):
    return [c for c in calls if c[0] in ("flush", "finish", "unregister")]


@pytest.fixture
def ordered(monkeypatch, calls, backend):
    monkeypatch.setattr(TelemetryManager, "is_telemetry_initialized", property(lambda self: True))
    monkeypatch.setattr(TelemetryManager, "force_flush",
                        lambda self, timeout_millis=5000: calls.append(("flush", timeout_millis)) or True)
    monkeypatch.setattr(TelemetryManager, "unregister_client",
                        lambda self, client_id: calls.append(("unregister",)))
    return calls


class TestSpansFlushBeforeFinish:
    def test_sessions_end(self, client, ordered):
        client.sessions.end(_SESSION_ID)
        assert _recorded(ordered) == [("flush", 5000), ("finish",)]

    @pytest.mark.asyncio
    async def test_sessions_aend(self, client, ordered):
        await client.sessions.aend(_SESSION_ID)
        assert _recorded(ordered) == [("flush", 5000), ("finish",)]

    def test_session_end(self, client, ordered):
        session = client.sessions.create(session_name="s")
        session.end()
        assert _recorded(ordered) == [("flush", 5000), ("finish",)]

    @pytest.mark.parametrize("closer", ["close", "aclose"])
    def test_close_flushes_before_finishing_and_unregistering(self, client, ordered, closer):
        import asyncio

        client.sessions.create(session_name="s")
        client._providers = ["openai"]  # so close() unregisters from telemetry
        result = getattr(client, closer)()
        if closer == "aclose":
            asyncio.run(result)
        assert _recorded(ordered) == [("flush", 5000), ("finish",), ("unregister",)]

    @pytest.mark.parametrize("closer", ["close", "aclose"])
    def test_close_flushes_once_for_all_its_sessions(self, client, ordered, closer):
        for name in ("a", "b"):
            client.sessions.create(session_name=name)
        result = getattr(client, closer)()
        if closer == "aclose":
            asyncio.run(result)
        assert _recorded(ordered) == [("flush", 5000), ("finish",), ("finish",)]

    def test_with_session_flushes_once(self, client, ordered):
        with client.sessions.create(session_name="s"):
            pass
        assert _recorded(ordered) == [("flush", 5000), ("finish",)]

    @pytest.mark.asyncio
    async def test_async_with_session_flushes_once(self, client, ordered):
        async with client.sessions.create(session_name="s"):
            pass
        assert _recorded(ordered) == [("flush", 5000), ("finish",)]

    def test_shutdown_flushes_once_then_ends_every_session(self, client, ordered, monkeypatch):
        monkeypatch.setattr(ShutdownManager, "_instance", None)
        mgr = ShutdownManager()
        mgr._clients[client._client_id] = client
        for name in ("a", "b"):
            client.sessions.create(session_name=name)
        mgr._perform_shutdown()
        assert _recorded(ordered) == [("flush", 5000), ("finish",), ("finish",)]


@pytest.fixture
def slow_flush(monkeypatch, backend):
    monkeypatch.setattr(TelemetryManager, "is_telemetry_initialized", property(lambda self: True))
    monkeypatch.setattr(TelemetryManager, "force_flush",
                        lambda self, timeout_millis=5000: time.sleep(_SLOW_FLUSH_SECONDS) or True)


async def _max_loop_gap(work):
    """Run work() while a ticker records the longest stall of the event loop."""
    gaps, done = [], asyncio.Event()

    async def tick():
        last = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    ticker = asyncio.ensure_future(tick())
    await asyncio.sleep(0.02)
    try:
        await work()
    finally:
        done.set()
        await ticker
    return max(gaps)


class TestAsyncEndsKeepTheLoopFree:
    @pytest.mark.asyncio
    async def test_sessions_aend(self, client, slow_flush):
        assert await _max_loop_gap(lambda: client.sessions.aend(_SESSION_ID)) < _MAX_LOOP_GAP_SECONDS

    @pytest.mark.asyncio
    async def test_aclose(self, client, slow_flush):
        client.sessions.create(session_name="s")
        assert await _max_loop_gap(client.aclose) < _MAX_LOOP_GAP_SECONDS

    @pytest.mark.asyncio
    async def test_async_with_session(self, client, slow_flush):
        session = client.sessions.create(session_name="s")

        async def work():
            async with session:
                pass

        assert await _max_loop_gap(work) < _MAX_LOOP_GAP_SECONDS


@pytest.mark.asyncio
async def test_aclose_warning_names_the_flush(client, caplog):
    def boom(*args):
        raise RuntimeError("executor gone")

    client._flush_evals = boom
    try:
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            await client.aclose()
    finally:
        del client._flush_evals
    assert "Flush skipped on async close" in caplog.text
