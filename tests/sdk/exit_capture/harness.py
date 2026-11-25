"""Runs a child scenario against the recording fake backend and checks what reached it."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
AGENT_ID = "aaaa1111-0000-0000-0000-000000000000"
RAW_API_KEY = "secret-key"
CHILD_TIMEOUT_SECONDS = 60
LINE_TIMEOUT_SECONDS = 30
SIGNAL_SETTLE_SECONDS = 0.3


@dataclass(frozen=True)
class Capture:
    returncode: int
    output: str
    requests: List[dict]
    raw_log: str


def _pump(stream, lines):
    for line in stream:
        lines.put(line)
    lines.put(None)


def _wait_for(lines, marker, seen):
    deadline = time.monotonic() + LINE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=0.1)
        except queue.Empty:
            continue
        if line is None:
            raise AssertionError(f"child exited before {marker!r}:\n{''.join(seen)}")
        seen.append(line)
        if marker in line:
            return
    raise AssertionError(f"timed out waiting for {marker!r}:\n{''.join(seen)}")


def _drain(lines, seen):
    while True:
        line = lines.get(timeout=CHILD_TIMEOUT_SECONDS)
        if line is None:
            return
        seen.append(line)


def _start_backend(log_path, event_delay):
    backend = subprocess.Popen([sys.executable, str(HERE / "fake_backend.py"), str(log_path)],
                               stdout=subprocess.PIPE, text=True,
                               env=dict(os.environ, FAKE_EVENT_DELAY=str(event_delay)))
    port = int(backend.stdout.readline().strip().split("=", 1)[1])
    return backend, port


def _child_env(port):
    return dict(
        os.environ,
        LUCIDIC_BASE_URL=f"http://127.0.0.1:{port}/api",
        LUCIDIC_DEBUG="false",
        OPENAI_BASE_URL=f"http://127.0.0.1:{port}/v1",
        PYTHONPATH=str(REPO),
    )


def run_scenario(tmp_path, scenario: str, signals: Sequence[Tuple[str, int]] = (),
                 event_delay: float = 0) -> Capture:
    """Run ``child.py scenario``; for each (marker, sig), send sig once marker is printed."""
    log_path = tmp_path / "requests.jsonl"
    log_path.touch()
    backend, port = _start_backend(log_path, event_delay)
    child = None
    try:
        child = subprocess.Popen([sys.executable, str(HERE / "child.py"), scenario],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 env=_child_env(port), cwd=str(tmp_path))
        lines, seen = queue.Queue(), []
        threading.Thread(target=_pump, args=(child.stdout, lines), daemon=True).start()
        for marker, sig in signals:
            _wait_for(lines, marker, seen)
            time.sleep(SIGNAL_SETTLE_SECONDS)
            child.send_signal(sig)
        _drain(lines, seen)
        returncode = child.wait(timeout=CHILD_TIMEOUT_SECONDS)
        time.sleep(SIGNAL_SETTLE_SECONDS)  # let the backend finish writing
    finally:
        for proc in (child, backend):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
    raw_log = log_path.read_text()
    requests = [json.loads(line) for line in raw_log.splitlines()]
    return Capture(returncode, "".join(seen), requests, raw_log)


def lucidic_requests(requests):
    return [r for r in requests if not r["path"].startswith("/v1/")]


def summary(requests):
    return [(r["method"], r["path"], (r["body"] or {}).get("type") or (r["body"] or {}).get("is_finished"))
            for r in lucidic_requests(requests)]


def assert_session_sequence(capture: Capture, n_events: int = 1) -> None:
    """initsession first, n unique llm events for that session, one is_finished PUT last."""
    reqs = lucidic_requests(capture.requests)
    context = f"{summary(capture.requests)}\n{capture.output}"
    assert reqs, context
    inits = [r for r in reqs if r["path"].endswith("/initsession")]
    assert len(inits) == 1 and reqs[0] is inits[0], context
    assert inits[0]["method"] == "POST" and inits[0]["body"]["agent_id"] == AGENT_ID, context
    session_id = inits[0]["body"]["session_id"]

    events = [(i, r) for i, r in enumerate(reqs) if r["path"].endswith("/events")]
    assert len(events) == n_events, context
    assert all(r["method"] == "POST" and r["body"]["session_id"] == session_id
               and r["body"]["type"] == "llm_generation" for _, r in events), context
    event_ids = [r["body"]["client_event_id"] for _, r in events]
    assert len(set(event_ids)) == n_events, context

    finishes = [(i, r) for i, r in enumerate(reqs)
                if r["path"].endswith("/updatesession") and (r["body"] or {}).get("is_finished")]
    assert len(finishes) == 1, context
    finish_index, finish = finishes[0]
    assert finish["method"] == "PUT" and finish["body"]["session_id"] == session_id, context
    assert finish_index == len(reqs) - 1, context
    assert all(i < finish_index for i, _ in events), context

    assert all(r["headers"].get("x-agent-id") == AGENT_ID and "authorization" in r["headers"]
               for r in reqs), context
    assert RAW_API_KEY not in capture.raw_log
