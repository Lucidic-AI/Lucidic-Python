"""What reaches the backend when a real client's process ends.

Each test runs ``child.py`` in a subprocess against a recording fake backend and
checks the captured Lucidic requests: initsession, the LLM event, then is_finished.
"""
import signal
import time

import pytest

from lucidicai import runner

from .child import MANY_SESSIONS
from .harness import assert_session_sequence, lucidic_requests

SIGINT, SIGTERM = signal.SIGINT, signal.SIGTERM
STUCK_EXPORT_SECONDS = 40
SPAN_FLUSH_BOUND_SECONDS = 5


@pytest.mark.parametrize("scenario", [
    "exit",               # session left open
    "with",
    "manual_end",
    "sessions_end",
    "close",              # os._exit right after close()
    "last_event_exit",    # the LLM call is the last statement
    "async_aend",
    "async_aclose",       # os._exit right after aclose()
])
def test_normal_endings_deliver_the_event_before_finish(capture, scenario):
    result = capture(scenario)
    assert result.returncode == 0, result.output
    assert_session_sequence(result)


def test_manual_end_in_a_long_lived_process_posts_the_event_before_finish(capture):
    # the batch timer would deliver it, but after is_finished
    result = capture("manual_end_linger")
    assert result.returncode == 0, result.output
    assert_session_sequence(result)


def test_ctrl_c_flushes_then_raises_keyboard_interrupt(capture):
    result = capture("sigint", [("READY", SIGINT)])
    assert_session_sequence(result)
    assert result.returncode == -SIGINT, result.output
    assert "KeyboardInterrupt" in result.output


def test_ctrl_c_while_end_exports_does_not_stall_the_shutdown(capture):
    # the signal handler interrupts end()'s flush; the shutdown must not wait on that export
    start = time.monotonic()
    result = capture("sigint_during_end", [("READY", SIGINT)], event_delay=3)
    assert time.monotonic() - start < 20, result.output  # the shutdown wait is 30 s
    assert_session_sequence(result)
    assert result.returncode == -SIGINT, result.output


def test_second_ctrl_c_quits_promptly_while_an_export_hangs(capture):
    result = capture("sigint", [("READY", SIGINT), (None, SIGINT)], event_delay=STUCK_EXPORT_SECONDS)
    assert result.exit_after_last_signal < 5, result.output
    assert result.returncode == -SIGINT, result.output


def test_sigterm_with_a_stuck_export_finishes_every_session_within_one_flush(capture):
    result = capture("many_sessions_sigterm", [("READY", SIGTERM)], event_delay=STUCK_EXPORT_SECONDS)
    reqs = lucidic_requests(result.requests)
    started = {r["body"]["session_id"] for r in reqs if r["path"].endswith("/initsession")}
    finished = {r["body"]["session_id"] for r in reqs
                if r["path"].endswith("/updatesession") and (r["body"] or {}).get("is_finished")}
    assert len(started) == MANY_SESSIONS and finished == started, result.output
    assert result.exit_after_last_signal < SPAN_FLUSH_BOUND_SECONDS + 5, result.output
    assert result.returncode == -SIGTERM, result.output


def test_sigterm_flushes_then_terminates_by_signal(capture):
    result = capture("sigterm", [("READY", SIGTERM)])
    assert_session_sequence(result)
    assert result.returncode == -SIGTERM, result.output


def test_app_handler_installed_before_the_client_still_runs(capture):
    result = capture("app_handler_before", [("READY", SIGTERM)])
    assert_session_sequence(result)
    assert "APP_HANDLER" in result.output
    assert result.returncode == 7, result.output


def test_app_handler_installed_after_the_client_runs_and_atexit_flushes(capture):
    result = capture("app_handler_after", [("READY", SIGTERM)])
    assert_session_sequence(result)
    assert "APP_HANDLER" in result.output
    assert result.returncode == 7, result.output


def test_client_built_off_the_main_thread_works_and_flushes_at_exit(capture):
    result = capture("thread_client")
    assert "THREAD_ERRORS=[]" in result.output, result.output
    assert result.returncode == 0, result.output
    assert_session_sequence(result)


def test_client_built_off_the_main_thread_then_sigterm_is_a_plain_termination(capture):
    # documented limitation: no handler can be installed from that thread
    result = capture("thread_client_sigterm", [("READY", SIGTERM)])
    assert "THREAD_ERRORS=[]" in result.output, result.output
    assert result.returncode == -SIGTERM, result.output
    reqs = lucidic_requests(result.requests)
    assert [r["path"] for r in reqs] == ["/api/initsession"], result.output


@pytest.mark.skipif(not hasattr(runner, "_drain_on_signal"), reason="needs serve()'s drain handler")
def test_second_signal_under_serve_drain_flushes_and_force_quits(capture):
    # serve's force quit chains to our handler, which must not re-raise the signal;
    # it then exits with 128 + signum
    result = capture("serve_second_signal", [("READY", SIGTERM), ("DRAINING", SIGTERM)])
    assert_session_sequence(result)
    assert result.returncode == 128 + SIGTERM, result.output
