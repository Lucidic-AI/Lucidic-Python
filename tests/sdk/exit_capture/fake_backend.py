"""Recording fake Lucidic + OpenAI backend.

Prints ``PORT=<n>`` and appends one JSON line per request to ``argv[1]``.
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG_PATH = sys.argv[1]
LOG_LOCK = threading.Lock()
EVENT_DELAY_SECONDS = float(os.environ.get("FAKE_EVENT_DELAY", "0"))
COMPLETION = {
    "id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _parse(raw):
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return raw.decode(errors="replace")


def _reply(path, body):
    body = body if isinstance(body, dict) else {}
    if path.endswith("/chat/completions"):
        return COMPLETION
    if path.endswith("/initsession"):
        return {"session_id": body.get("session_id")}
    if path.endswith("/events"):
        return {"event_id": body.get("client_event_id")}
    return {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _handle(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = _parse(self.rfile.read(length) if length else b"")
        headers = {k.lower(): ("<redacted>" if k.lower() == "authorization" else v)
                   for k, v in self.headers.items()}
        record = {"method": self.command, "path": self.path, "headers": headers, "body": body}
        with LOG_LOCK, open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.path.endswith("/events"):
            time.sleep(EVENT_DELAY_SECONDS)
        out = json.dumps(_reply(self.path, body)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    do_GET = do_POST = do_PUT = do_PATCH = _handle


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(f"PORT={server.server_address[1]}", flush=True)
    server.serve_forever()
