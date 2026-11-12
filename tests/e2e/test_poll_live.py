"""The real lucidicai.poll against a real backend: one pull-mode case ends COMPLETED.

Skipped unless LUCIDIC_E2E_BASE_URL (e.g. http://127.0.0.1:8000/api) and
LUCIDIC_E2E_API_KEY are set; run it against the backend conformance stack before
each 4.1.x release (README, "Releasing"). It mirrors the backend's
smoke_tests/conformance.py pull_round_trip, with the SDK poller doing the claim.
"""
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

import pytest

from lucidicai import LucidicAI

BASE_URL = os.environ.get("LUCIDIC_E2E_BASE_URL")
API_KEY = os.environ.get("LUCIDIC_E2E_API_KEY")
TIMEOUT = 150

pytestmark = pytest.mark.skipif(
    not (BASE_URL and API_KEY), reason="set LUCIDIC_E2E_BASE_URL and LUCIDIC_E2E_API_KEY")

# poll() needs the main thread, so it runs in its own process
POLLER = """
import logging, os, sys, time
import lucidicai as lucidic

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(name)s %(levelname)s %(message)s")
logging.getLogger("lucidicai.runner").setLevel(logging.DEBUG)

def run_case(run):
    with run:
        print("CASE_STARTED", run.session_id, flush=True)
        time.sleep(4)  # past a few 1 s heartbeats

lucidic.poll(run_case, [os.environ["E2E_AGENT_ID"]], workers=1, _heartbeat_seconds=1)
print("POLL_RETURNED", flush=True)
"""


def _wait_for(lines, text, proc, timeout=TIMEOUT):
    deadline = time.time() + timeout
    while not any(text in line for line in lines):
        assert proc.poll() is None, "the poller exited:\n" + "".join(lines)
        assert time.time() < deadline, f"no {text!r} within {timeout} s:\n" + "".join(lines)
        time.sleep(0.2)


def test_poll_completes_a_pull_case():
    nonce = uuid.uuid4().hex[:12]
    admin = LucidicAI(api_key=API_KEY, base_url=BASE_URL, production=False)
    agent_id = admin.agents.create(f"e2e-poll-{nonce}").agent_id
    client = LucidicAI(api_key=API_KEY, base_url=BASE_URL, agent_id=agent_id, production=False)
    dataset_id = client.datasets.create(f"e2e-poll-{nonce}")["dataset_id"]
    client.datasets.create_item(dataset_id, "case", {"e2e": nonce})
    run = client.datasets.start_run(dataset_id, repeats=1, window=1, repeat_deadline_seconds=120)
    assert run.delivery == "pull"

    env = {k: v for k, v in os.environ.items() if not k.startswith("LUCIDIC_")}
    env.update(LUCIDIC_BASE_URL=BASE_URL, LUCIDIC_API_KEY=API_KEY, E2E_AGENT_ID=agent_id)
    proc = subprocess.Popen([sys.executable, "-c", POLLER], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    reader = threading.Thread(target=lambda: lines.extend(iter(proc.stdout.readline, "")), daemon=True)
    reader.start()
    try:
        _wait_for(lines, "CASE_STARTED", proc)
        _wait_for(lines, "heartbeat 200", proc, timeout=30)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(60) == 0, "".join(lines)
    finally:
        if proc.poll() is None:
            proc.kill()
    reader.join(5)
    assert any("POLL_RETURNED" in line for line in lines), "".join(lines)

    deadline = time.time() + TIMEOUT
    while True:
        cases = admin._http.get(f"sdk/v2/experiments/{run.experiment_id}/outcomes").get("cases") or []
        found = [case.get("delivery") for case in cases]
        if found == ["COMPLETED"]:
            break
        assert not any(d in ("FAILED", "TIMEOUT") for d in found), cases
        assert time.time() < deadline, f"the case is {found}, not COMPLETED"
        time.sleep(3)
    client.close(eval_flush_timeout=0)
    admin.close(eval_flush_timeout=0)
