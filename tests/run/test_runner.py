"""lucidicai.Runner / lucidicai.serve: the stdlib webhook runner."""
import http.client
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

import lucidicai as lucidic
from lucidicai import webhook
from lucidicai.runner import MIN_DEDUPE_TTL, _drain_on_signal, _http_server

from .conftest import AGENT_ID, SECRET, SESSION_ID, full_payload, signed, wire

OTHER_IDS = [f"{i}{i}{i}{i}{i}{i}{i}{i}-0000-0000-0000-000000000000" for i in range(1, 4)]


def _headers(signature):
    return {webhook.SIGNATURE_HEADER: signature}


def _post(runner, payload, **sign_kwargs):
    body = wire(payload)
    return runner.handle(body, _headers(webhook.sign(SECRET, body, **sign_kwargs)))


def _accept_soon(runner, session_id, timeout=5):
    """Re-send like the backend while the runner answers 503."""
    deadline = time.time() + timeout
    status, _ = _post(runner, full_payload(session_id=session_id))
    while status == 503 and time.time() < deadline:
        time.sleep(0.01)
        status, _ = _post(runner, full_payload(session_id=session_id))
    return status


@pytest.fixture
def make_runner(client):
    runners = []

    def make(run_case=lambda run: None, **kwargs):
        runner = lucidic.Runner(run_case, kwargs.pop("secret", SECRET), client=client, **kwargs)
        runners.append(runner)
        return runner

    yield make
    for runner in runners:
        runner.close()


def test_accepts_once_then_replies_duplicate(make_runner):
    ran = []
    runner = make_runner(lambda run: ran.append(run.ids.session_id), workers=2)

    assert _post(runner, full_payload()) == (202, b"accepted")
    assert _post(runner, full_payload()) == (200, b"duplicate")   # at-least-once re-delivery
    runner.close()
    assert ran == [SESSION_ID]


def test_stale_signature_is_401_and_run_case_never_runs(make_runner):
    ran = []
    runner = make_runner(ran.append)

    status, _ = _post(runner, full_payload(), t=int(time.time()) - 301)
    assert status == 401
    # nothing was claimed, so a fresh delivery of the same case is accepted
    assert _post(runner, full_payload())[0] == 202
    runner.close()
    assert len(ran) == 1


def test_load_errors_map_to_their_status(make_runner):
    runner = make_runner()
    body = wire(full_payload())
    assert runner.handle(body, _headers("t=0,v1=00"))[0] == 401
    assert runner.handle(body, {})[0] == 401
    assert _post(runner, {"session_id": "not-a-uuid"})[0] == 422


def test_no_secret_is_412(make_runner):
    ran = []
    runner = make_runner(ran.append, secret=None)
    assert runner.has_secret is False
    assert _post(runner, full_payload())[0] == 412
    assert ran == []


def test_secret_is_read_from_env_once_with_a_rotation_list(make_runner, monkeypatch):
    monkeypatch.setenv("LUCIDIC_WEBHOOK_SECRET", f"old-secret, {SECRET}")
    runner = make_runner(secret=None)
    monkeypatch.delenv("LUCIDIC_WEBHOOK_SECRET")
    assert _post(runner, full_payload())[0] == 202


def test_body_over_the_cap_is_413(make_runner):
    ran = []
    runner = make_runner(ran.append, max_body=10)
    assert _post(runner, full_payload())[0] == 413
    assert ran == []


def test_headers_are_case_insensitive(make_runner):
    body, signature = signed(full_payload())
    assert make_runner().handle(body, {"x-lucidic-signature": signature})[0] == 202


def test_full_is_503_and_the_retry_is_accepted_later(make_runner):
    release = threading.Event()
    runner = make_runner(lambda run: release.wait(5), workers=1)
    try:
        statuses = [_post(runner, full_payload(session_id=sid))[0] for sid in OTHER_IDS]
    finally:
        release.set()
    # one running + one queued; the 503 left no claim behind
    assert statuses == [202, 202, 503]
    assert _accept_soon(runner, OTHER_IDS[2]) == 202


def test_redelivery_of_an_accepted_case_is_200_even_when_full(make_runner):
    release = threading.Event()
    runner = make_runner(lambda run: release.wait(5), workers=1)
    try:
        assert [_post(runner, full_payload(session_id=sid))[0] for sid in OTHER_IDS[:2]] == [202, 202]
        assert _post(runner, full_payload(session_id=OTHER_IDS[0])) == (200, b"duplicate")
        assert _post(runner, full_payload(session_id=OTHER_IDS[2]))[0] == 503
    finally:
        release.set()


class _FlakyStore:
    """Raises on the first ``fails`` claims, then accepts."""

    def __init__(self, fails):
        self.fails = fails

    def claim(self, key):
        if self.fails:
            self.fails -= 1
            raise ConnectionError("store down")
        return True

    def release(self, key):
        pass


def test_a_failing_dedupe_store_does_not_leak_slots(make_runner):
    runner = make_runner(workers=1, dedupe=_FlakyStore(fails=3))
    for _ in range(3):
        with pytest.raises(ConnectionError):
            _post(runner, full_payload())
    # both slots are still free
    assert [_post(runner, full_payload(session_id=sid))[0] for sid in OTHER_IDS[:2]] == [202, 202]


def test_a_failing_run_case_is_logged_and_frees_its_slot(make_runner, caplog):
    def boom(run):
        raise RuntimeError("agent crashed")

    runner = make_runner(boom, workers=1)
    # three cases through two slots, so each failure released its slot
    assert [_accept_soon(runner, sid) for sid in OTHER_IDS] == [202, 202, 202]
    runner.close()
    assert caplog.text.count("failed") == 3


def test_custom_dedupe_store_is_used(make_runner):
    class Store:
        def __init__(self):
            self.claims, self.released = [], []

        def claim(self, key):
            self.claims.append(key)
            return False

        def release(self, key):
            self.released.append(key)

    store = Store()
    ran = []
    runner = make_runner(ran.append, dedupe=store)
    assert _post(runner, full_payload()) == (200, b"duplicate")
    assert store.claims == [SESSION_ID]
    assert ran == []


def test_memory_dedupe_expires_after_ttl():
    now = [0.0]
    store = lucidic.MemoryDedupe(ttl=MIN_DEDUPE_TTL, clock=lambda: now[0])
    assert store.claim("a") is True
    assert store.claim("a") is False
    now[0] = MIN_DEDUPE_TTL + 1
    assert store.claim("a") is True
    store.release("a")
    assert store.claim("a") is True


def test_memory_dedupe_ttl_must_outlast_the_retries():
    with pytest.raises(ValueError):
        lucidic.MemoryDedupe(ttl=MIN_DEDUPE_TTL - 1)


def test_serve_refuses_to_start_without_a_secret(monkeypatch):
    def no_bind(*args, **kwargs):
        raise AssertionError("bound a port with no secret")

    monkeypatch.setattr("lucidicai.runner._http_server", no_bind)
    with pytest.raises(lucidic.RunSecretMissingError):
        lucidic.serve(lambda run: None, port=0)


@pytest.fixture
def live(make_runner):
    ran = []
    runner = make_runner(ran.append, max_body=4096)
    server = _http_server(runner, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1], ran
    server.shutdown()
    server.server_close()


def _request(port, body, headers):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.putrequest("POST", "/")
        for name, value in headers.items():
            conn.putheader(name, value)
        conn.endheaders()
        if body:
            conn.send(body)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def test_http_accepts_a_signed_delivery(live):
    port, _ = live
    body, signature = signed(full_payload())
    headers = {"Content-Length": str(len(body)), webhook.SIGNATURE_HEADER: signature}
    assert _request(port, body, headers) == (202, b"accepted")


def test_http_stale_delivery_is_401(live):
    port, ran = live
    body = wire(full_payload())
    signature = webhook.sign(SECRET, body, t=int(time.time()) - 301)
    headers = {"Content-Length": str(len(body)), webhook.SIGNATURE_HEADER: signature}
    assert _request(port, body, headers)[0] == 401
    assert ran == []


@pytest.mark.parametrize(
    "length, status",
    [(None, 411), ("12x", 400), ("-1", 400), ("\u00b2", 400), ("9" * 5000, 400), ("4097", 413)],
)
def test_http_content_length_is_checked_before_reading(live, length, status):
    port, ran = live
    headers = {} if length is None else {"Content-Length": length}
    assert _request(port, b"", headers)[0] == status
    assert ran == []


def test_http_store_error_is_500_and_retryable(make_runner):
    runner = make_runner(dedupe=_FlakyStore(fails=1))
    server = _http_server(runner, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        body, signature = signed(full_payload())
        headers = {"Content-Length": str(len(body)), webhook.SIGNATURE_HEADER: signature}
        assert _request(port, body, headers)[0] == 500
        assert _request(port, body, headers) == (202, b"accepted")
    finally:
        server.shutdown()
        server.server_close()


def test_drain_on_signal_needs_the_main_thread():
    errors = []

    def off_main():
        try:
            with _drain_on_signal(lambda: None):
                pass
        except RuntimeError as e:
            errors.append(e)

    thread = threading.Thread(target=off_main)
    thread.start()
    thread.join()
    assert len(errors) == 1


def test_drain_on_signal_hands_a_second_signal_back_and_restores_on_exit():
    seen, stops = [], []

    def before(signum, frame):
        seen.append(signum)

    old = signal.signal(signal.SIGTERM, before)
    try:
        with _drain_on_signal(lambda: stops.append(1)) as stopped:
            signal.raise_signal(signal.SIGTERM)
            assert stopped.is_set() and stops == [1] and seen == []
            signal.raise_signal(signal.SIGTERM)
            assert seen == [signal.SIGTERM] and stops == [1]
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        assert signal.getsignal(signal.SIGTERM) is before
    finally:
        signal.signal(signal.SIGTERM, old)


# A LucidicAI() arms the ShutdownManager (end sessions, sys.exit(0) on SIGTERM); serve must win.
_SERVE_CHILD = f"""
import logging, os, sys, time
import lucidicai as lucidic
logging.getLogger("lucidicai.runner").addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger("lucidicai.runner").setLevel(logging.INFO)
client = lucidic.LucidicAI(api_key="k", agent_id="{AGENT_ID}", production=False)
release = sys.argv[1]

def run_case(run):
    print("case started", flush=True)
    while not os.path.exists(release):
        time.sleep(0.01)
    print("case done", flush=True)

lucidic.serve(run_case, port=0, secret="{SECRET}", client=client)
print("serve returned", flush=True)
"""


def _lines(proc):
    lines = queue.Queue()

    def pump():
        for line in proc.stdout:
            lines.put(line.rstrip("\n"))

    threading.Thread(target=pump, daemon=True).start()
    return lines


def _wait_for(lines, prefix, timeout=15):
    deadline = time.time() + timeout
    while True:
        line = lines.get(timeout=max(0.01, deadline - time.time()))
        if line.startswith(prefix):
            return line


def _refused(port):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        return False
    except ConnectionRefusedError:
        return True


def test_sigterm_drains_the_running_case_and_serve_returns(tmp_path):
    release = tmp_path / "release"
    proc = subprocess.Popen([sys.executable, "-c", _SERVE_CHILD, str(release)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        lines = _lines(proc)
        port = int(_wait_for(lines, "lucidic runner listening on").rsplit(":", 1)[1])
        body, signature = signed(full_payload())
        headers = {"Content-Length": str(len(body)), webhook.SIGNATURE_HEADER: signature}
        assert _request(port, body, headers) == (202, b"accepted")
        _wait_for(lines, "case started")

        proc.send_signal(signal.SIGTERM)
        deadline = time.time() + 10
        while not _refused(port):
            assert time.time() < deadline, "server still accepting after SIGTERM"
            time.sleep(0.05)
        body, signature = signed(full_payload(session_id=OTHER_IDS[0]))
        with pytest.raises(ConnectionRefusedError):
            _request(port, body, {"Content-Length": str(len(body)), webhook.SIGNATURE_HEADER: signature})
        assert proc.poll() is None, "exited before the accepted case finished"

        release.touch()
        assert _wait_for(lines, "case") == "case done"
        _wait_for(lines, "serve returned")
        assert proc.wait(timeout=15) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
