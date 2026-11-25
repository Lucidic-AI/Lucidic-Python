"""What reaches the backend when a real client's process ends (B-04, B-05/06).

Each test runs ``child.py`` in a subprocess against a recording fake backend and
checks the captured Lucidic requests: initsession, the LLM event, then is_finished.
"""
import signal
import time

import pytest

from lucidicai import runner

from .harness import assert_session_sequence, lucidic_requests

SIGINT, SIGTERM = signal.SIGINT, signal.SIGTERM


@pytest.mark.parametrize("scenario", [
    "exit",               # 1: session left open
    "with",               # 2: guard, correct today
    "manual_end",         # 3
    "sessions_end",       # 4
    "close",              # 5: os._exit right after close()
    "last_event_exit",    # 10: B-05 repro, the LLM call is the last statement
    "async_aend",         # 11
    "async_aclose",       # 11: os._exit right after aclose()
])
def test_normal_endings_deliver_the_event_before_finish(capture, scenario):
    result = capture(scenario)
    assert result.returncode == 0, result.output
    assert_session_sequence(result)


def test_manual_end_in_a_long_lived_process_posts_the_event_before_finish(capture):
    # 3b: the batch timer would deliver it, but after is_finished
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


@pytest.mark.skipif(not hasattr(runner, "_drain_on_signal"), reason="needs #103's _drain_on_signal")
def test_second_signal_under_serve_drain_flushes_and_force_quits(capture):
    # 12: force_quit chains to our handler, which must not re-raise the signal;
    # it then exits with 128 + signum
    result = capture("serve_second_signal", [("READY", SIGTERM), ("DRAINING", SIGTERM)])
    assert_session_sequence(result)
    assert result.returncode == 128 + SIGTERM, result.output
