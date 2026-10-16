"""A webhook runner for eval runs: verify, dedupe, bounded pool, reply 202 fast.

    lucidic.serve(run_case, host="0.0.0.0", port=8080)

The backend waits 10 s for a reply, re-sends on 5xx and timeouts for about 5
minutes, and may deliver a case twice. So a request only verifies, dedupes and
queues; ``run_case(run)`` runs later on a pool of ``workers`` threads.

- Capacity: ``workers * 2`` cases run or wait at once; beyond that the reply is
  503 and the backend re-sends later. Keep ``workers * 2`` at least the EvoSim
  dispatch window, or cases spend their retries on 503s.
- Dedupe is by ``session_id`` for ``ttl`` seconds. The default store is in
  memory, so it resets on restart (cases already acked with 202 then end as
  rollout timeouts); pass a shared store when running several replicas. A
  store's ``claim`` runs under the runner's lock, so keep it fast.
- Stopping the server drains: cases already acked with 202 still run.
- Deliveries signed more than 300 s from this host's clock get 401, so keep the
  clock synced (NTP).
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

from .core.errors import RunLoadError, RunSecretMissingError
from .run import Run, load
from .run.load import webhook_keys
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
    """Serve eval-run webhooks until interrupted, calling ``run_case(run)`` for each new case.

    Raises:
        RunSecretMissingError: no ``secret=`` and no ``$LUCIDIC_WEBHOOK_SECRET`` (before binding).
    """
    runner = Runner(run_case, secret, client=client, workers=workers, max_body=max_body, dedupe=dedupe)
    if not runner.has_secret:
        runner.close()
        raise RunSecretMissingError(f"no webhook secret: pass secret= or set {SECRET_ENV_VAR}")
    server = _http_server(runner, host, port)
    logger.info(f"lucidic runner listening on {host}:{server.server_address[1]}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        runner.close(wait=False)
