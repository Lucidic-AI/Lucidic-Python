"""lucidicai.poll / Poller: the pull-mode poller, against a respx backend."""
import json
import logging
import os
import signal
import threading
import time
from contextlib import contextmanager

import httpx
import pytest

import lucidicai as lucidic
from lucidicai import LucidicAI
from lucidicai.api.models.dataset import EvalRun
from lucidicai.api.models.evosim import EvoSim
from lucidicai.api.resources.evosim import PULL_DELIVERY_WARNING
from lucidicai.core.config import REGION_URLS, Region
from lucidicai.runner import CLAIM_PATH, NO_PULL_QUEUE, Poller

from .conftest import AGENT_ID, CHECKPOINT_ID, EVAL_RUN_ID, EXPERIMENT_ID, ITEM_ID, SESSION_ID, full_payload

JOB_ID = "10b10b10-0000-4000-8000-000000000001"
OTHER_JOB = "10b10b10-0000-4000-8000-000000000002"
OTHER_AGENT = "bbbb2222-0000-0000-0000-000000000000"
HEARTBEAT = f"/sdk/v2/rollout-jobs/{JOB_ID}/heartbeat"


def job(job_id=JOB_ID, agent_id=AGENT_ID, **payload):
    return {
        "v": 1, "job_id": job_id, "session_id": SESSION_ID, "agent_id": agent_id,
        "evosim_id": "e1e1e1e1-0000-0000-0000-000000000000", "tmi_id": "a1a1a1a1-0000-0000-0000-000000000000",
        "module_key": "eval_run", "rollout": 0, "claim_expires_at": "2026-11-11T00:02:00+00:00",
        "payload": full_payload(**payload),
    }


def claim_reply(*jobs, poll_after=5, agent_ids=(AGENT_ID,)):
    return httpx.Response(200, json={"jobs": list(jobs), "poll_after_seconds": poll_after,
                                     "agent_ids": list(agent_ids)})


def replies(*items):
    """A respx side effect: each item in turn (an exception is raised), then empty claims."""
    queue = list(items)

    def effect(request):
        item = queue.pop(0) if queue else claim_reply()
        if isinstance(item, Exception):
            raise item
        return item

    return effect


def error(status, **headers):
    return httpx.Response(status, json={"error": f"status {status}"}, headers=headers)


def wait_until(condition, timeout=5.0):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline, "timed out"
        time.sleep(0.01)


@pytest.fixture
def claim(backend):
    return backend.post(f"/{CLAIM_PATH}", name="claim")


@pytest.fixture
def heartbeat(backend):
    return backend.post(HEARTBEAT, name="heartbeat").mock(
        return_value=httpx.Response(200, json={"job_id": JOB_ID, "status": "RUNNING"}))


@pytest.fixture
def make_poller(client):
    pollers = []

    def make(run_case=lambda run: None, stop_after=None, **kwargs):
        """``stop_after=n``: the injected sleep records each wait and stops the poller at the n-th."""
        sleeps = []
        if stop_after is not None:
            def sleep(seconds):
                sleeps.append(seconds)
                if len(sleeps) >= stop_after:
                    poller.stop()
            kwargs["_sleep"] = sleep
        kwargs.setdefault("client", client)
        poller = Poller(run_case, **kwargs)
        poller.sleeps = sleeps
        pollers.append(poller)
        return poller

    yield make
    for poller in pollers:
        poller.stop()


@contextmanager
def running(poller):
    thread = threading.Thread(target=poller.run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        poller.stop()
        thread.join(10)
        assert not thread.is_alive()


@pytest.fixture
def closed(monkeypatch):
    seen = []
    original = LucidicAI.close

    def close(self, *args, **kwargs):
        seen.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LucidicAI, "close", close)
    return seen


# ==================== jobs become Runs ====================

def test_a_claimed_job_becomes_a_run_with_its_ids_and_the_per_agent_client(
        make_poller, client, claim, backend, closed):
    claim.side_effect = replies(claim_reply(job(temperature=0.2)))
    seen = {}

    def run_case(run):
        with run:
            seen.update(ids=run.ids, input=run.input, params=dict(run.params), client=run.client)

    poller = make_poller(run_case, stop_after=1, workers=3)
    poller.run()

    assert seen["ids"] == lucidic.RunIds(SESSION_ID, ITEM_ID, EXPERIMENT_ID, CHECKPOINT_ID, EVAL_RUN_ID)
    assert seen["input"] == full_payload()["input"]
    assert seen["params"] == {"temperature": 0.2}   # tuner params only, no job metadata
    built = seen["client"]
    assert built is not client and built.agent_id == AGENT_ID and built._production is False
    assert json.loads(claim.calls[0].request.content)["agent_ids"] == [AGENT_ID]
    assert json.loads(claim.calls[0].request.content)["max_jobs"] == 3
    assert backend["initsession"].call_count == 1 and backend["updatesession"].call_count == 1
    assert built in closed and client not in closed


def test_it_claims_at_most_ten_and_waits_poll_after_seconds(make_poller, claim):
    claim.side_effect = replies(claim_reply(poll_after=7), claim_reply(poll_after=0), claim_reply(poll_after=3))
    poller = make_poller(stop_after=2, workers=25)
    poller.run()

    assert [json.loads(c.request.content)["max_jobs"] for c in claim.calls] == [10, 10, 10]
    assert poller.sleeps == [7, 3]   # 0: claim again at once


def test_heartbeats_while_a_job_runs_and_stop_on_409(make_poller, claim, heartbeat):
    claim.side_effect = replies(claim_reply(job()))
    heartbeat.mock(side_effect=[httpx.Response(200, json={}), error(409)])
    release = threading.Event()

    def run_case(run):
        with run:
            release.wait(5)

    poller = make_poller(run_case, _heartbeat_seconds=0.01)
    with running(poller):
        wait_until(lambda: heartbeat.call_count == 2)
        time.sleep(0.1)
        assert heartbeat.call_count == 2   # no more after the 409
        release.set()
    assert heartbeat.calls[0].request.extensions["timeout"]["read"] == 5


@pytest.mark.parametrize("returns", [False, True], ids=["raises", "returns"])
def test_run_case_that_never_enters_still_opens_and_ends_the_session(make_poller, claim, backend, caplog, returns):
    claim.side_effect = replies(claim_reply(job()))

    def run_case(run):
        if not returns:
            raise RuntimeError("boom before with run")

    poller = make_poller(run_case, stop_after=1)
    with caplog.at_level(logging.ERROR, logger="lucidicai.runner"):
        poller.run()

    assert backend["initsession"].call_count == 1
    assert json.loads(backend["initsession"].calls[0].request.content)["session_id"] == SESSION_ID
    assert backend["updatesession"].call_count == 1
    if returns:
        assert any("returned without `with run:`" in r.getMessage() and JOB_ID in r.getMessage()
                   for r in caplog.records)


@pytest.mark.parametrize("route,status", [("tools", 500), ("initsession", 400), ("initsession", 500)])
def test_an_error_entering_the_run_is_logged_and_the_job_dropped(
        make_poller, claim, backend, heartbeat, caplog, route, status):
    backend[route].mock(return_value=error(status))
    claim.side_effect = replies(claim_reply(job()))

    def run_case(run):
        with run:
            pass

    poller = make_poller(run_case, stop_after=2)
    first_sleep = poller._sleep

    def sleep(seconds):
        wait_until(lambda: not poller._jobs)   # the job finished before the next claim
        first_sleep(seconds)

    poller._sleep = sleep
    with caplog.at_level(logging.ERROR, logger="lucidicai.runner"):
        poller.run()

    assert claim.call_count == 2   # kept polling
    never = [r for r in caplog.records if "never opened" in r.getMessage()]
    assert never and never[0].levelno == logging.ERROR and JOB_ID in never[0].getMessage()
    poller._heartbeat_round()
    assert heartbeat.call_count == 0


def test_poller_clients_are_never_production(make_poller, monkeypatch, claim, backend, caplog):
    monkeypatch.setenv("LUCIDIC_PRODUCTION", "true")
    prod = LucidicAI(api_key="test-key", agent_id=AGENT_ID, production=True)
    try:
        poller = make_poller(client=prod, stop_after=1)
        assert all(c._production is False for c in poller._clients.values())
        backend["initsession"].mock(return_value=error(500))
        claim.side_effect = replies(claim_reply(job()))
        poller._run_case = lambda run: run.__enter__()
        with caplog.at_level(logging.ERROR, logger="lucidicai.runner"):
            poller.run()
        assert any("never opened" in r.getMessage() for r in caplog.records)
    finally:
        prod.close(eval_flush_timeout=0)


# ==================== errors and backoff ====================

def test_backoff_on_429_503_500_and_transport_errors_is_capped_at_60(make_poller, claim):
    claim.side_effect = replies(
        error(429, **{"Retry-After": "7"}), error(503), error(500), httpx.ConnectError("down"),
        error(500), error(500), error(500), error(500),
        claim_reply(),
    )
    poller = make_poller(stop_after=9)
    poller.run()

    assert poller.sleeps == [7, 2, 4, 8, 16, 32, 60, 60, 5]


def test_retry_after_zero_still_backs_off(make_poller, claim):
    claim.side_effect = replies(*[error(429, **{"Retry-After": "0"})] * 3, claim_reply())
    poller = make_poller(stop_after=4)
    poller.run()

    assert poller.sleeps == [1, 2, 4, 5]


def test_401_and_403_retry_every_60_seconds(make_poller, claim):
    claim.side_effect = replies(error(401), error(403), error(401), claim_reply())
    poller = make_poller(stop_after=4)
    poller.run()

    assert poller.sleeps == [60, 60, 60, 5]


def test_claim_404_says_the_backend_needs_the_pull_queue(make_poller, claim, caplog):
    claim.side_effect = replies(error(404), error(404))
    poller = make_poller(stop_after=2)
    with caplog.at_level(logging.ERROR, logger="lucidicai.runner"):
        poller.run()

    assert poller.sleeps == [60, 60]
    assert sum(NO_PULL_QUEUE in r.getMessage() for r in caplog.records) == 2


def test_validation_error_and_a_plain_value_error_are_logged_and_backed_off(make_poller, claim, caplog):
    claim.side_effect = replies(error(400))
    poller = make_poller(stop_after=3)
    real_post = poller._claimer._http.post
    calls = []

    def post(path, data=None):
        calls.append(path)
        if len(calls) == 2:
            raise ValueError("a bug")
        return real_post(path, data)

    poller._claimer._http.post = post
    with caplog.at_level(logging.ERROR, logger="lucidicai.runner"):
        poller.run()

    assert poller.sleeps == [1, 2, 5]
    failed = [r for r in caplog.records if r.getMessage() == "claim failed"]
    assert len(failed) == 2 and all(r.exc_info for r in failed)


@pytest.mark.parametrize("extra", [{"poll_after_seconds": None}, {"agent_ids": 7}], ids=["null-delay", "bad-agents"])
def test_a_bad_optional_field_still_runs_the_leased_jobs(make_poller, claim, extra):
    body = {"jobs": [job()], "poll_after_seconds": 3, "agent_ids": [AGENT_ID], **extra}
    claim.side_effect = replies(httpx.Response(200, json=body))
    ran = []
    poller = make_poller(lambda run: ran.append(run), stop_after=1)
    poller.run()

    assert len(ran) == 1
    assert poller.sleeps == [5 if extra.get("poll_after_seconds", 3) is None else 3]


def test_a_job_already_in_flight_or_without_an_id_is_not_started(make_poller, claim, backend, caplog):
    no_id = job()
    del no_id["job_id"]
    claim.side_effect = replies(claim_reply(job(), job(), no_id))
    release = threading.Event()
    ran = []

    def run_case(run):
        ran.append(run)
        with run:
            release.wait(5)

    poller = make_poller(run_case, workers=4)
    with caplog.at_level(logging.ERROR, logger="lucidicai.runner"):
        with running(poller):
            wait_until(lambda: len(ran) == 1)
            time.sleep(0.05)
            release.set()
    assert len(ran) == 1
    assert sum("already in flight" in r.getMessage() for r in caplog.records) == 2


def test_a_bad_reply_body_is_backed_off(make_poller, claim):
    claim.side_effect = replies(httpx.Response(200, json={"poll_after_seconds": 5}))
    poller = make_poller(stop_after=2)
    poller.run()

    assert poller.sleeps == [1, 5]


def test_dropped_agents_warn_once_each_and_none_served_errors_once_a_minute(make_poller, claim, caplog):
    clock = iter([0, 30, 61])
    claim.side_effect = replies(
        claim_reply(agent_ids=[AGENT_ID]), claim_reply(agent_ids=[AGENT_ID]),
        claim_reply(agent_ids=[]), claim_reply(agent_ids=[]), claim_reply(agent_ids=[]),
    )
    poller = make_poller(stop_after=5, agent_ids=[AGENT_ID, OTHER_AGENT], _clock=lambda: next(clock))
    with caplog.at_level(logging.WARNING, logger="lucidicai.runner"):
        poller.run()

    dropped = [r for r in caplog.records if r.levelno == logging.WARNING and "dropped agent" in r.getMessage()]
    assert [OTHER_AGENT in r.getMessage() for r in dropped] == [True, False]   # OTHER once, then AGENT once
    assert sum(r.levelno == logging.ERROR and "never receive a case" in r.getMessage()
               for r in caplog.records) == 2   # at clock 0 and 61, not 30


# ==================== heartbeats ====================

def test_heartbeat_404_drops_only_that_job_and_errors_keep_the_thread_alive(make_poller, client, backend, heartbeat):
    poller = make_poller()
    built = poller._claimer
    other = backend.post(f"/sdk/v2/rollout-jobs/{OTHER_JOB}/heartbeat").mock(return_value=error(404))
    poller._jobs.update({JOB_ID: built, OTHER_JOB: built})

    poller._heartbeat_round()
    assert poller._jobs == {JOB_ID: built, OTHER_JOB: None}
    assert heartbeat.calls.last.request.extensions["timeout"] == {
        "connect": 5.0, "read": 5.0, "write": 5.0, "pool": 5.0}

    heartbeat.mock(side_effect=httpx.ConnectError("down"))
    poller._heartbeat_seconds = 0.01
    poller._beat.start()
    wait_until(lambda: heartbeat.call_count >= 4)
    assert poller._beat.is_alive() and poller._jobs[JOB_ID] is built
    heartbeat.mock(side_effect=ValueError("a bug"))
    count = heartbeat.call_count
    wait_until(lambda: heartbeat.call_count >= count + 2)
    assert poller._beat.is_alive() and other.call_count == 1
    poller._jobs.clear()
    poller.close()


# ==================== startup ====================

def test_startup_fails_without_agent_ids(monkeypatch):
    monkeypatch.delenv("LUCIDIC_AGENT_ID", raising=False)
    no_agent = LucidicAI(api_key="test-key", production=False)
    try:
        with pytest.raises(ValueError, match="agent_ids"):
            Poller(lambda run: None, client=no_agent)
        with pytest.raises(ValueError, match="agent_ids"):
            lucidic.poll(lambda run: None)
    finally:
        no_agent.close(eval_flush_timeout=0)


def test_no_base_url_raises_unless_require_base_url_is_off(monkeypatch):
    monkeypatch.delenv("LUCIDIC_BASE_URL")
    monkeypatch.delenv("LUCIDIC_REQUIRE_BASE_URL", raising=False)
    monkeypatch.setenv("LUCIDIC_API_KEY", "test-key")
    with pytest.raises(ValueError, match="no base URL"):
        Poller(lambda run: None, [AGENT_ID])

    saas = Poller(lambda run: None, [AGENT_ID], require_base_url=False)
    assert saas.base_url == REGION_URLS[Region.US]
    saas.close()

    fell_back = LucidicAI(api_key="test-key", agent_id=AGENT_ID, production=False)
    try:
        with pytest.raises(ValueError, match="fell back to Lucidic SaaS"):
            Poller(lambda run: None, client=fell_back)
    finally:
        fell_back.close(eval_flush_timeout=0)


def test_poll_needs_the_main_thread():
    errors = []
    thread = threading.Thread(target=lambda: errors.append(pytest.raises(RuntimeError, lucidic.poll, lambda r: None)))
    thread.start()
    thread.join()
    assert errors and "main thread" in str(errors[0].value)


def test_per_agent_clients_copy_the_passed_clients_settings(claim, backend, closed):
    passed = LucidicAI(api_key="test-key", agent_id=AGENT_ID, production=False, providers=["openai"],
                       eval_flush_timeout=1.5, auto_end=False)
    events = backend.post("/events").mock(return_value=httpx.Response(201, json={"event_id": "ev1"}))
    claim.side_effect = replies(claim_reply(job()))

    def run_case(run):
        from lucidicai.telemetry.telemetry_manager import get_telemetry_manager

        with run:
            tracer = get_telemetry_manager().tracer_provider.get_tracer("test")
            with tracer.start_as_current_span("openai.chat", attributes={
                "gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o",
                "gen_ai.prompt.0.role": "user", "gen_ai.prompt.0.content": "hi",
                "gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": "hello",
            }):
                pass

    try:
        stops = []
        poller = Poller(run_case, client=passed, _sleep=lambda s: stops.append(poller.stop()))
        built = poller._claimer
        assert built._providers == ["openai"] and built.eval_flush_timeout == 1.5
        assert built._config.auto_end is False and built._config.api_key == "test-key"
        poller.run()
        wait_until(lambda: events.call_count >= 1)
        assert json.loads(events.calls[0].request.content)["session_id"] == SESSION_ID
        assert built in closed and passed not in closed
    finally:
        passed.close(eval_flush_timeout=0)


def enter(run):
    with run:
        pass


@pytest.mark.parametrize("pass_client", [True, False], ids=["client", "env"])
def test_module_level_mockable_tools_sync_for_every_agent(claim, backend, monkeypatch, pass_client):
    @lucidic.mockable
    def lookup(x: str) -> str:
        """look up"""
        return x

    monkeypatch.setenv("LUCIDIC_API_KEY", "test-key")
    passed = LucidicAI(agent_id=AGENT_ID, production=False) if pass_client else None
    other_session = "5e55e55e-0000-4000-8000-000000000002"
    claim.side_effect = replies(claim_reply(
        job(), job(OTHER_JOB, OTHER_AGENT, session_id=other_session), agent_ids=[AGENT_ID, OTHER_AGENT]))
    try:
        poller = Poller(enter, [AGENT_ID, OTHER_AGENT], client=passed, _sleep=lambda s: poller.stop())
        poller.run()
        bodies = [json.loads(c.request.content) for c in backend["sync"].calls]
        assert sorted(b["agent_id"] for b in bodies) == sorted([AGENT_ID, OTHER_AGENT])
        assert all([t["name"] for t in b["tools"]] == ["lookup"] for b in bodies)
    finally:
        if passed:
            passed.close(eval_flush_timeout=0)


# ==================== signals ====================

def test_sigterm_drains_the_job_heartbeating_it_and_poll_returns(client, claim, heartbeat, caplog):
    claim.side_effect = replies(claim_reply(job()))
    finished = []
    beats = {}

    def run_case(run):
        with run:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.1)
            beats["at_signal"] = heartbeat.call_count
            time.sleep(0.3)
            finished.append(run.ids.session_id)

    with caplog.at_level(logging.INFO, logger="lucidicai.runner"):
        lucidic.poll(run_case, workers=1, client=client, _heartbeat_seconds=0.02)

    assert finished == [SESSION_ID]
    assert claim.call_count == 1   # no claim after the signal
    assert heartbeat.call_count > beats["at_signal"]   # heartbeats went on during the drain
    assert any("draining" in r.getMessage() and JOB_ID in r.getMessage() for r in caplog.records)


# ==================== start_run, train, models ====================

def test_start_run_without_webhook_url_sends_none_and_warns_on_pull(client, backend, caplog):
    route = backend.post("/sdk/v2/datasets/ds1/runs").mock(return_value=httpx.Response(
        202, json={"eval_run_id": EVAL_RUN_ID, "status": "running", "delivery": "pull"}))
    with caplog.at_level(logging.WARNING, logger="Lucidic"):
        run = client.datasets.start_run("ds1", repeats=1)

    assert "webhook_url" not in json.loads(route.calls[0].request.content)
    assert run.delivery == "pull"
    assert [r.getMessage() for r in caplog.records if r.getMessage() == PULL_DELIVERY_WARNING] == [
        PULL_DELIVERY_WARNING]


async def _astart(client):
    return await client.datasets.astart_run("ds1", webhook_url="https://runner.example/hook")


def test_start_run_with_a_webhook_is_silent(client, backend, caplog):
    import asyncio

    route = backend.post("/sdk/v2/datasets/ds1/runs").mock(return_value=httpx.Response(
        202, json={"eval_run_id": EVAL_RUN_ID, "delivery": "push"}))
    with caplog.at_level(logging.WARNING, logger="Lucidic"):
        asyncio.run(_astart(client))

    assert json.loads(route.calls[0].request.content)["webhook_url"] == "https://runner.example/hook"
    assert PULL_DELIVERY_WARNING not in caplog.text


@pytest.mark.parametrize("evosim,warns", [
    ({"evosim_id": "x", "delivery": "pull"}, True), ({"evosim_id": "x", "delivery": "push"}, False), ("x", False),
], ids=["pull", "push", "not-a-dict"])
def test_train_warns_on_pull_only(client, backend, caplog, tmp_path, evosim, warns):
    backend.post("/sdk/evosim/training-run").mock(return_value=httpx.Response(201, json={"evosim": evosim}))
    config = tmp_path / "train.json"
    config.write_text(json.dumps({"name": "t"}))
    with caplog.at_level(logging.WARNING, logger="Lucidic"):
        assert client.evosims.train(config) == {"evosim": evosim}

    assert (PULL_DELIVERY_WARNING in caplog.text) is warns


def test_models_parse_the_delivery_fields():
    fields = {"delivery": "pull", "queued": 3, "claimed": 1, "oldest_queued_seconds": 42}
    for model, key in ((EvoSim, "evosim_id"), (EvalRun, "eval_run_id")):
        obj = model.from_dict({key: "x", **fields})
        assert {k: getattr(obj, k) for k in fields} == fields
