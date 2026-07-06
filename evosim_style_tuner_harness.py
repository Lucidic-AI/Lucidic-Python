"""LUC-797 Phase 2 — webhook harness: the client "agent" that gets rolled out.

Run a tiny HTTP server and point the EvoSim's ``webhook_url`` at it. On each rollout
dispatch the backend POSTs ``{"session_id": ..., ...params}``; this harness then:

  1. starts a Lucidic session with that session_id — the backend binds the in-training
     candidate checkpoint to it server-side via the LUC-790 pre-link (the SDK does not
     know the checkpoint id);
  2. fetches the target prompt (``client.prompts.get`` auto-sends the session context,
     so with LUC-795 it resolves the CANDIDATE prompt version, not the base label);
  3. calls the style_tuner tool (an inference against the candidate artifact);
  4. ends the session — which signals rollout completion back to the training loop.

Requires an SDK build that includes LUC-795 (the checkpoint-scoped prompt fetch): the
``luc-795-sdk-checkpoint-scoped-prompt-fetch`` branch or a ``lucidicai>=3.7.2`` release.

Usage (env points the client at your local sst-dev backend):
    LUCIDIC_API_KEY=... LUCIDIC_AGENT_ID=... LUCIDIC_BASE_URL=http://localhost:8000 \
      TARGET_PROMPT=style_tuner_demo_prompt python3 evosim_style_tuner_harness.py --port 8799
"""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import lucidicai as lai

TARGET_PROMPT = os.environ.get("TARGET_PROMPT", "style_tuner_demo_prompt")

# api_key / agent_id / base_url are read from the LUCIDIC_* env vars.
client = lai.LucidicAI(providers=[])


def _run_session(session_id: str, params: dict) -> None:
    try:
        session = client.sessions.create(
            session_id=session_id, session_name=f"rollout::{session_id[:8]}"
        )
        try:
            prompt = client.prompts.get(TARGET_PROMPT)  # -> candidate version via session
            style = client.training_modules.call("get_style_guide", {})  # inference vs candidate
            print(
                f"[harness] session={session_id[:8]} "
                f"prompt={prompt.content[:70]!r} style={style}"
            )
        finally:
            session.end()  # -> finishSession -> signals rollout completion
    except Exception as exc:  # a rollout session must never crash the harness
        print(f"[harness] session {session_id[:8]} failed: {exc}")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            body = {}
        session_id = body.get("session_id")
        # respond fast; run the session off-thread so we don't stall dispatch pacing.
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')
        if session_id:
            threading.Thread(
                target=_run_session, args=(str(session_id), body), daemon=True
            ).start()

    def log_message(self, *args) -> None:  # quiet the default access log
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    args = ap.parse_args()
    print(f"[harness] listening on :{args.port} (POST any path), target_prompt={TARGET_PROMPT}")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
