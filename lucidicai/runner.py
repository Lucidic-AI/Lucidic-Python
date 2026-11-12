"""Eval-run runners: ``serve`` (push: webhooks) and ``poll`` (pull: claim jobs).

    lucidic.serve(run_case, port=8080)  # host defaults to 127.0.0.1
    lucidic.poll(run_case)              # outbound only; see poll() and Poller

``serve``: verify, dedupe, bounded pool, reply 202 fast.

Pass ``host="0.0.0.0"`` to listen on all interfaces (e.g. in a container).

The backend waits 10 s for a reply, re-sends on 5xx and timeouts for about 5
minutes, and may deliver a case twice (signing and re-send need a backend
release that includes World Evals, LUC-997). So a request only verifies,
dedupes and queues; ``run_case(run)`` runs later on a pool of ``workers`` threads.

- Capacity: ``workers * 2`` cases run or wait at once; beyond that the reply is
  503 and the backend re-sends later. Keep ``workers * 2`` at least the EvoSim
  dispatch window, or cases spend their retries on 503s.
- Dedupe is by ``session_id`` for ``ttl`` seconds. The default store is in
  memory, so it resets on restart (cases already acked with 202 then end as
  rollout timeouts); pass a shared store when running several replicas. A
  store's ``claim`` runs under the runner's lock, so keep it fast.
- SIGTERM or SIGINT drains: ``serve`` stops accepting, waits for the cases
  already acked with 202, then returns (instead of the client's exit-on-signal).
  A second signal force quits: the previous handler runs (the client still ends
  its sessions), then the process exits with 128 + the signal number, even with
  cases still running.
- Deliveries signed more than 300 s from this host's clock get 401, so keep the
  clock synced (NTP).
"""
import logging
import os
import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol, Tuple

import httpx

from .core.errors import (
    APIError,
    AuthError,
    ConflictError,
    InsufficientScopeError,
    NotFoundError,
    RateLimitError,
    RunLoadError,
    RunSecretMissingError,
    ServiceUnavailableError,
)
from .run import Run, load
from .run.load import _run_from_payload, webhook_keys
from .webhook import SECRET_ENV_VAR, SIGNATURE_HEADER
from .webhook.signing import Secret

if TYPE_CHECKING:
    from .client import LucidicAI

logger = logging.getLogger("lucidicai.runner")

DEFAULT_WORKERS = 4
DEFAULT_MAX_BODY = 1024 * 1024
DEFAULT_DEDUPE_TTL = 3600
MIN_DEDUPE_TTL = 900
READ_TIMEOUT = 10

Response = Tuple[int, bytes]


class DedupeStore(Protocol):
    def claim(self, key: str) -> bool:
        """True when ``key`` was not claimed yet (and is now)."""

    def release(self, key: str) -> None:
        """Drop a claim so a re-delivery is accepted."""


class MemoryDedupe:
    """In-process dedupe; claims expire after ``ttl`` seconds."""

    def __init__(self, ttl: float = DEFAULT_DEDUPE_TTL, clock: Callable[[], float] = time.monotonic):
        if ttl < MIN_DEDUPE_TTL:
            raise ValueError(f"dedupe ttl must be at least {MIN_DEDUPE_TTL} s, longer than the backend's retries")
        self._ttl = ttl
        self._clock = clock
        self._expires: Dict[str, float] = {}
        self._lock = threading.Lock()

    def claim(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            self._expires = {k: t for k, t in self._expires.items() if t > now}
            if key in self._expires:
                return False
            self._expires = {**self._expires, key: now + self._ttl}
            return True

    def release(self, key: str) -> None:
        with self._lock:
            self._expires = {k: t for k, t in self._expires.items() if k != key}


def _header(headers: Mapping[str, str], name: str) -> Optional[str]:
    value = headers.get(name)
    if value is None:
        value = next((v for k, v in headers.items() if k.lower() == name.lower()), None)
    return value


class Runner:
    """``handle(body, headers)`` returns ``(status, body)`` to reply with; see the module doc."""

    def __init__(
        self,
        run_case: Callable[[Run], Any],
        secret: Optional[Secret] = None,
        *,
        client: Optional["LucidicAI"] = None,
        workers: int = DEFAULT_WORKERS,
        max_body: int = DEFAULT_MAX_BODY,
        dedupe: Optional[DedupeStore] = None,
    ):
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self._run_case = run_case
        self._keys = webhook_keys(secret)  # read once, at boot
        self._client = client
        self.max_body = max_body
        self._dedupe = MemoryDedupe() if dedupe is None else dedupe
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="lucidic-runner")
        # running + queued cases
        self._slots = threading.BoundedSemaphore(workers * 2)
        self._lock = threading.Lock()

    @property
    def has_secret(self) -> bool:
        return bool(self._keys)

    def handle(self, body: bytes, headers: Mapping[str, str]) -> Response:
        if len(body) > self.max_body:
            return 413, f"body exceeds {self.max_body} bytes".encode()
        try:
            run = load(body, _header(headers, SIGNATURE_HEADER), secret=self._keys, client=self._client)
        except RunLoadError as e:
            logger.warning(f"rejected delivery: {e}")
            return e.status_code, str(e).encode()
        session_id = run.ids.session_id
        # claim then slot under one lock: a re-delivery is 200 even when full, and a 503 leaves no claim
        with self._lock:
            if not self._dedupe.claim(session_id):
                return 200, b"duplicate"
            if not self._slots.acquire(blocking=False):
                self._dedupe.release(session_id)
                return 503, b"runner full, retry later"
            try:
                self._pool.submit(self._work, run)
            except BaseException as e:
                self._dedupe.release(session_id)
                self._slots.release()
                if isinstance(e, RuntimeError):
                    return 503, b"runner shutting down"
                raise
        return 202, b"accepted"

    def _work(self, run: Run) -> None:
        try:
            self._run_case(run)
        except Exception:
            logger.exception(f"case {run.ids.session_id} failed")
        finally:
            self._slots.release()

    def close(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)


def _http_server(runner: Runner, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        timeout = READ_TIMEOUT

        def do_POST(self) -> None:
            try:
                status, reply = self._read_and_handle()
            except Exception:
                # e.g. a custom dedupe store failed; 500 lets the backend retry
                logger.exception("delivery failed")
                self.close_connection = True
                status, reply = 500, b"internal error"
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def _read_and_handle(self) -> Response:
            declared = self.headers.get("Content-Length")
            if declared is None:
                self.close_connection = True
                return 411, b"Content-Length required"
            # ascii digits only, and short enough for int()
            if not (declared.isascii() and declared.isdigit() and len(declared) <= 15):
                self.close_connection = True
                return 400, b"invalid Content-Length"
            length = int(declared)
            if length > runner.max_body:
                self.close_connection = True
                return 413, f"body exceeds {runner.max_body} bytes".encode()
            return runner.handle(self.rfile.read(length), self.headers)

        def log_message(self, format: str, *args: Any) -> None:
            logger.debug(format % args)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def _require_main_thread() -> None:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("signal handlers can only be installed on the main thread")


@contextmanager
def _drain_on_signal(stop: Callable[[], None]) -> Iterator[threading.Event]:
    """On the first SIGTERM/SIGINT, set the yielded event and call ``stop()`` (from the signal handler).

    Install it after the clients are built: it replaces their exit-on-signal handlers
    for the block and puts them back on exit. A second signal force quits: it runs
    the previous handler, then ``os._exit(128 + signum)``, since a handler that only
    raises (sys.exit, KeyboardInterrupt) would still wait for the pool's threads at
    interpreter exit. ``stop`` must not block.
    """
    _require_main_thread()
    stopped = threading.Event()
    signals = (signal.SIGTERM, signal.SIGINT)
    previous = {s: signal.getsignal(s) for s in signals}

    def restore() -> None:
        for s, h in previous.items():
            # None: a handler not set from Python; SIG_DFL is the closest
            signal.signal(s, signal.SIG_DFL if h is None else h)

    def force_quit(signum: int, frame: Any) -> None:
        logger.info(f"received signal {signum} again, force quitting")
        try:
            if callable(previous[signum]):
                previous[signum](signum, frame)
        finally:
            os._exit(128 + signum)

    def on_signal(signum: int, frame: Any) -> None:
        for s in signals:  # first, so a quick second signal already force quits
            signal.signal(s, force_quit)
        logger.info(f"received signal {signum}, draining")
        stopped.set()
        stop()

    for s in signals:
        signal.signal(s, on_signal)
    try:
        yield stopped
    finally:
        restore()


def serve(
    run_case: Callable[[Run], Any],
    host: str = "127.0.0.1",
    port: int = 8080,
    workers: int = DEFAULT_WORKERS,
    *,
    secret: Optional[Secret] = None,
    client: Optional["LucidicAI"] = None,
    max_body: int = DEFAULT_MAX_BODY,
    dedupe: Optional[DedupeStore] = None,
) -> None:
    """Serve eval-run webhooks until SIGTERM/SIGINT, calling ``run_case(run)`` for each new case.

    Call it from the main thread. On a signal it stops accepting, waits for the
    accepted cases to finish and returns; a second signal force quits.

    Raises:
        RunSecretMissingError: no ``secret=`` and no ``$LUCIDIC_WEBHOOK_SECRET`` (before binding).
        RuntimeError: called off the main thread (signal handlers need it; before binding).
    """
    _require_main_thread()
    runner = Runner(run_case, secret, client=client, workers=workers, max_body=max_body, dedupe=dedupe)
    if not runner.has_secret:
        runner.close()
        raise RunSecretMissingError(f"no webhook secret: pass secret= or set {SECRET_ENV_VAR}")
    server = _http_server(runner, host, port)
    logger.info(f"lucidic runner listening on {host}:{server.server_address[1]}")
    # shutdown() blocks until serve_forever returns, so not from the handler on this thread
    stop = lambda: threading.Thread(target=server.shutdown, daemon=True).start()  # noqa: E731
    with _drain_on_signal(stop):  # also around close(), so a second signal force quits the drain
        try:
            server.serve_forever()
        finally:
            server.server_close()
            runner.close()  # waits for the accepted cases


# ==================== Pull mode ====================

CLAIM_PATH = "sdk/v2/rollout-jobs/claim"
MAX_CLAIM = 10                # the backend's max_jobs cap
DEFAULT_POLL_AFTER = 5.0      # a claim reply without poll_after_seconds
HEARTBEAT_SECONDS = 30.0      # leases: 120 s for a claimed case, 180 s for a running one
HEARTBEAT_TIMEOUT = 5.0       # per request, so slow calls cannot cost the other jobs their lease
MAX_BACKOFF = 60.0
SLOW_RETRY = 60.0             # 401, 403, and a backend with no pull queue
NO_AGENTS_LOG_EVERY = 60.0
NO_PULL_QUEUE = (
    "the backend has no pull queue (claim answered 404); it needs a backend release "
    "with the pull rollout queue (LUC-1279) or later"
)


def _canonical(agent_id: Any) -> str:
    try:
        return str(uuid.UUID(str(agent_id)))
    except ValueError:
        return str(agent_id)


def _poll_agent_ids(agent_ids: Optional[Iterable[str]], client: Optional["LucidicAI"]) -> List[str]:
    if isinstance(agent_ids, str):
        agent_ids = [agent_ids]
    if not agent_ids and client is not None and client.agent_id:
        agent_ids = [client.agent_id]
    ids = list(dict.fromkeys(_canonical(a) for a in agent_ids or () if a))
    if not ids:
        raise ValueError("poll needs agent_ids= or a client with an agent_id")
    return ids


def _poll_clients(
    ids: List[str], client: Optional["LucidicAI"], require_base_url: bool,
) -> Dict[str, "LucidicAI"]:
    """One client per agent, always built here; settings from ``client``, else from env."""
    from .client import LucidicAI

    shared: Dict[str, Any] = {}
    if client is not None:
        network = client._config.network
        # a URL resolved from a region (LUCIDIC_REGION, or the default) is a SaaS fallback
        if require_base_url and network.region is not None:
            raise ValueError(
                f"require_base_url is on, but the client fell back to Lucidic SaaS ({network.base_url}): "
                "give it base_url= or set LUCIDIC_BASE_URL, or pass poll(require_base_url=False) to poll SaaS"
            )
        shared = {
            "api_key": client._config.api_key,
            "base_url": network.base_url,
            "providers": list(client._providers),
            "eval_flush_timeout": client._eval_flush_timeout,
            "auto_end": client._config.auto_end,
        }
    clients: Dict[str, LucidicAI] = {}
    try:
        for agent_id in ids:
            # production=False: a failed InitSession raises instead of faking a session
            clients[agent_id] = LucidicAI(
                agent_id=agent_id, production=False, require_base_url=require_base_url, **shared)
    except BaseException:
        for built in clients.values():
            built.close(eval_flush_timeout=0)
        raise
    # Module-level @mockable tools drain into the first client built (the passed one, else ours),
    # so give every agent's client that registry: each syncs its tools when its session starts.
    source = client or next(iter(clients.values()))
    for built in clients.values():
        built.tools._registry.update(source.tools._registry)
    return clients


class Poller:
    """Claims pull-mode jobs, runs each as a Run on ``workers`` threads, heartbeats them; see ``poll``."""

    def __init__(
        self,
        run_case: Callable[[Run], Any],
        agent_ids: Optional[Iterable[str]] = None,
        workers: int = DEFAULT_WORKERS,
        *,
        client: Optional["LucidicAI"] = None,
        require_base_url: bool = True,
        _sleep: Optional[Callable[[float], Any]] = None,
        _clock: Callable[[], float] = time.monotonic,
        _heartbeat_seconds: float = HEARTBEAT_SECONDS,
    ):
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self._run_case = run_case
        self._agent_ids = _poll_agent_ids(agent_ids, client)
        self._workers = workers
        self._stop = threading.Event()
        self._freed = threading.Event()
        self._beat_stop = threading.Event()
        self._sleep = _sleep or self._stop.wait
        self._clock = _clock
        self._heartbeat_seconds = _heartbeat_seconds
        self._lock = threading.Lock()
        # job id -> the client that heartbeats it; None once its heartbeats stopped
        self._jobs: Dict[str, Optional["LucidicAI"]] = {}
        self._failures = 0
        self._warned_dropped: set = set()
        self._no_agents_logged_at: Optional[float] = None
        self._clients = _poll_clients(self._agent_ids, client, require_base_url)
        self._claimer = next(iter(self._clients.values()))
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="lucidic-poller")
        self._beat = threading.Thread(target=self._heartbeat_loop, name="lucidic-poller-heartbeat", daemon=True)

    @property
    def base_url(self) -> str:
        return self._claimer._config.network.base_url

    def stop(self) -> None:
        """Stop claiming. Never blocks, so a signal handler may call it."""
        self._stop.set()
        self._freed.set()

    def run(self) -> None:
        """Claim and run jobs until ``stop()``, then drain and close (``close``)."""
        logger.info(f"lucidic poller polling {self.base_url} for agents {self._agent_ids} with {self._workers} workers")
        self._beat.start()
        try:
            while not self._stop.is_set():
                self._freed.clear()
                free = min(MAX_CLAIM, self._workers - len(self._jobs))
                if free <= 0:
                    self._freed.wait()  # a job finished, or stop()
                    continue
                delay = self._claim(free)
                if delay > 0 and not self._stop.is_set():
                    self._sleep(delay)
        finally:
            self.close()

    def close(self) -> None:
        """Wait for the jobs in flight (still heartbeating them), then close the poller's clients."""
        with self._lock:
            in_flight = list(self._jobs)
        logger.info(f"lucidic poller draining, jobs in flight: {in_flight}")
        self._pool.shutdown(wait=True)
        self._beat_stop.set()
        if self._beat.is_alive():
            self._beat.join()
        for built in self._clients.values():
            try:
                built.close()
            except Exception as e:
                logger.debug(f"error closing a poller client: {e}")

    # ---- claim ----

    def _claim(self, free: int) -> float:
        """Claim up to ``free`` jobs and start them. Returns the seconds to wait before the next claim."""
        try:
            body = self._claimer._http.post(CLAIM_PATH, {"agent_ids": self._agent_ids, "max_jobs": free})
            jobs = list(body["jobs"])
        except RateLimitError as e:
            return self._backoff("claim throttled (429)", e.retry_after)
        except ServiceUnavailableError as e:
            return self._backoff("backend unavailable (503)", e.retry_after)
        except (AuthError, InsufficientScopeError) as e:
            logger.error(f"claim refused ({e.status_code}; the key needs rollout-job:claim): {e}; "
                         f"retrying in {SLOW_RETRY:g} s")
            return SLOW_RETRY
        except NotFoundError:
            logger.error(f"{NO_PULL_QUEUE}; retrying in {SLOW_RETRY:g} s")
            return SLOW_RETRY
        except (APIError, httpx.TransportError) as e:
            return self._backoff(f"claim failed ({e!r})")
        except Exception:
            # another API error, a bad reply, a bug: keep polling
            logger.exception("claim failed")
            return self._backoff("claim failed")
        self._failures = 0
        for job in jobs:  # leased already: start them before reading the optional fields
            self._start(job)
        try:
            self._check_served(body.get("agent_ids"))
        except Exception:
            logger.exception("claim reply has bad agent_ids")
        delay = body.get("poll_after_seconds")
        return float(delay) if isinstance(delay, (int, float)) and delay >= 0 else DEFAULT_POLL_AFTER

    def _backoff(self, what: str, retry_after: Optional[float] = None) -> float:
        self._failures += 1
        # a Retry-After below the exponential step (e.g. 0) must not turn into a busy loop
        delay = min(MAX_BACKOFF, max(retry_after or 0.0, 2.0 ** (self._failures - 1)))
        logger.warning(f"{what}; retrying in {delay:g} s")
        return delay

    def _check_served(self, served: Optional[List[str]]) -> None:
        if served is None:
            return
        served_ids = {_canonical(a) for a in served}
        for agent_id in self._agent_ids:
            if agent_id not in served_ids and agent_id not in self._warned_dropped:
                self._warned_dropped.add(agent_id)
                logger.warning(f"the backend dropped agent {agent_id}: not an agent of the key's org, "
                               "or outside the key's agent binding")
        if not served_ids:
            now = self._clock()
            if self._no_agents_logged_at is None or now - self._no_agents_logged_at >= NO_AGENTS_LOG_EVERY:
                self._no_agents_logged_at = now
                logger.error(f"the backend serves none of the agents {self._agent_ids}: "
                             "this poller can never receive a case")

    def _start(self, job: Any) -> None:
        job_id = job.get("job_id") if isinstance(job, dict) else None
        with self._lock:
            duplicate = job_id is None or job_id in self._jobs
        if duplicate:
            # no id, or re-leased to us while still in flight: let the lease lapse
            logger.error(f"job {job_id}: missing job_id or already in flight; not starting it")
            return
        try:
            client = self._clients[_canonical(job["agent_id"])]
            run = _run_from_payload(dict(job["payload"]), {}, client)
        except Exception:
            logger.exception(f"job {job_id}: cannot start it; its lease lapses and the backend re-claims the case")
            return
        with self._lock:
            self._jobs[job_id] = client
        self._pool.submit(self._work, job_id, run)

    # ---- work ----

    def _work(self, job_id: str, run: Run) -> None:
        try:
            error: Optional[Exception] = None
            try:
                self._run_case(run)
            except Exception as e:
                logger.exception(f"job {job_id}: run_case raised")
                error = e
            if run.session_id is None:  # None until the server opened the session
                self._end_unopened(job_id, run, error)
        finally:
            with self._lock:
                self._jobs.pop(job_id, None)
            self._freed.set()

    def _end_unopened(self, job_id: str, run: Run, error: Optional[Exception]) -> None:
        """Open and end the session of a case ``run_case`` never opened, so the case ends."""
        if run._entered:
            # Run.__enter__ failed (module-tools fetch or InitSession), and a Run is entered once
            self._never_opened(job_id, error)
            return
        if error is None:
            logger.error(f"job {job_id}: run_case returned without `with run:`; ending its session")
            error = RuntimeError("run_case returned without `with run:`")
        try:
            with run:
                raise error
        except Exception as e:
            if e is not error:
                self._never_opened(job_id, e)

    @staticmethod
    def _never_opened(job_id: str, error: Optional[BaseException]) -> None:
        logger.error(f"job {job_id}: its session never opened ({error!r}); no more heartbeats, so the lease "
                     "lapses and the backend re-claims the case")

    # ---- heartbeat ----

    def _heartbeat_loop(self) -> None:
        while not self._beat_stop.wait(self._heartbeat_seconds):
            self._heartbeat_round()

    def _heartbeat_round(self) -> None:
        with self._lock:
            jobs = [(job_id, c) for job_id, c in self._jobs.items() if c is not None]
        for job_id, client in jobs:
            try:
                client._http.request("POST", f"sdk/v2/rollout-jobs/{job_id}/heartbeat", json={},
                                     timeout=HEARTBEAT_TIMEOUT)
                logger.debug(f"job {job_id}: heartbeat 200")
            except (NotFoundError, ConflictError) as e:
                logger.warning(f"job {job_id}: heartbeat {e.status_code}, no longer heartbeating it: {e}")
                with self._lock:
                    if job_id in self._jobs:
                        self._jobs[job_id] = None
            except Exception as e:
                logger.warning(f"job {job_id}: heartbeat failed, retrying next round: {e!r}")


def poll(
    run_case: Callable[[Run], Any],
    agent_ids: Optional[Iterable[str]] = None,
    workers: int = DEFAULT_WORKERS,
    *,
    client: Optional["LucidicAI"] = None,
    require_base_url: bool = True,
    _heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> None:
    """Claim pull-mode eval-run cases and call ``run_case(run)`` for each, until SIGTERM/SIGINT.

    The pull-mode twin of ``serve``, with the same ``run_case`` contract; it needs
    outbound access only. Call it from the main thread.

    - ``agent_ids`` default to ``[client.agent_id]``. The poller builds its own
      client per agent (``run.client``), copying ``client``'s api key, base URL,
      providers, ``eval_flush_timeout`` and ``auto_end`` (else env defaults), with
      ``production=False``: errors from ``run.client`` calls raise. It closes them
      on exit, never ``client``.
    - It claims up to the free workers (at most 10) and heartbeats the jobs in
      flight every 30 s. Errors back off (up to 60 s; 401, 403 and a backend
      without the pull queue retry every 60 s); it never exits on one.
    - A case whose ``run_case`` raised or returned before ``with run:`` still gets
      its session opened and ended. One whose session failed to open is logged and
      left to lapse; the backend re-claims it.
    - SIGTERM or SIGINT stops claiming, waits for the jobs in flight (still
      heartbeating them), closes the clients and returns. A second signal force quits.
    - ``require_base_url`` (default True): with no ``LUCIDIC_BASE_URL`` and no
      ``client`` with its own base URL, raise instead of polling Lucidic SaaS. SaaS
      users set the SaaS URL or pass ``require_base_url=False``.

    Raises (at startup):
        ValueError: no agent id, no base URL (see above), or ``workers`` < 1.
        RuntimeError: called off the main thread.
    """
    _require_main_thread()
    poller = Poller(run_case, agent_ids, workers, client=client, require_base_url=require_base_url,
                    _heartbeat_seconds=_heartbeat_seconds)
    with _drain_on_signal(poller.stop):  # also around the drain, so a second signal force quits it
        poller.run()
