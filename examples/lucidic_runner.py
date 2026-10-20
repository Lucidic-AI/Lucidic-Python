"""Serve lucidic_entry.run_case as the eval-run webhook (see lucidicai/runner.py).

    LUCIDIC_WEBHOOK_SECRET=... LUCIDIC_RUNNER_WORKERS=4 PORT=8080 python lucidic_runner.py

``LUCIDIC_WEBHOOK_SECRET`` may hold two comma-separated values while a secret
rotates. Without one the runner refuses to start.
"""
import os

import lucidicai as lucidic

if __name__ == "__main__":
    from lucidic_entry import client, run_case

    lucidic.serve(
        run_case,
        host="0.0.0.0",                                 # reachable from Lucidic, not only localhost
        port=int(os.environ.get("PORT", "8080")),
        workers=int(os.environ.get("LUCIDIC_RUNNER_WORKERS", "4")),
        client=client,
    )
