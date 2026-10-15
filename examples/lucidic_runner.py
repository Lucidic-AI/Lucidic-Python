"""A stdlib webhook runner for lucidic_entry.run_case: verify, dedupe, bounded pool, 202 fast.

The backend waits 10 s for a reply, re-sends on 5xx and timeouts, and may
deliver a case twice. So the handler only verifies and queues, replies 202 in
well under a second, and runs the case on a bounded pool.

    LUCIDIC_WEBHOOK_SECRET=... LUCIDIC_RUNNER_WORKERS=4 python lucidic_runner.py
"""
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

import lucidicai as lucidic

logger = logging.getLogger("lucidic_runner")


class Runner:
    """``handle(body, signature)`` returns the HTTP status to reply with."""

    def __init__(self, run_case: Callable[[lucidic.Run], Any], client: lucidic.LucidicAI, max_workers: int = 4):
        self._run_case = run_case
        self._client = client
        self._pool = ThreadPoolExecutor(max_workers)
        # running + queued cases; beyond this reply 503 and the backend re-sends later
        self._slots = threading.BoundedSemaphore(max_workers * 2)
        self._seen = set()  # in-memory dedupe; use a shared store when running several replicas
        self._lock = threading.Lock()

    def handle(self, body: bytes, signature: Optional[str]) -> int:
        try:
            run = lucidic.load(body, signature, client=self._client)
        except lucidic.RunLoadError as e:
            logger.warning(f"rejected delivery: {e}")
            return e.status_code
        with self._lock:
            if run.ids.session_id in self._seen:
                return 202  # a re-delivery of a case already taken
            if not self._slots.acquire(blocking=False):
                return 503
            self._seen.add(run.ids.session_id)
        self._pool.submit(self._work, run)
        return 202

    def _work(self, run: lucidic.Run) -> None:
        try:
            self._run_case(run)
        except Exception:
            logger.exception(f"case {run.ids.session_id} failed")
        finally:
            self._slots.release()


def serve(runner: Runner, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(runner.handle(body, self.headers.get(lucidic.webhook.SIGNATURE_HEADER)))
            self.end_headers()

    ThreadingHTTPServer(("", port), Handler).serve_forever()


if __name__ == "__main__":
    from lucidic_entry import client, run_case

    serve(Runner(run_case, client, int(os.environ.get("LUCIDIC_RUNNER_WORKERS", "4"))),
          int(os.environ.get("PORT", "8080")))
