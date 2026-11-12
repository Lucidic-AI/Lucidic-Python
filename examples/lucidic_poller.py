"""Poll for lucidic_entry.run_case's eval-run cases (pull mode; see lucidicai/runner.py).

    LUCIDIC_BASE_URL=https://lucidic.internal.example.com/api LUCIDIC_RUNNER_WORKERS=4 python lucidic_poller.py

Outbound access only: no port to open, no webhook secret. Start the eval run without
a webhook_url. SaaS users set LUCIDIC_BASE_URL to https://backend.lucidic.ai/api.
"""
import os

import lucidicai as lucidic

if __name__ == "__main__":
    from lucidic_entry import client, run_case

    lucidic.poll(
        run_case,
        workers=int(os.environ.get("LUCIDIC_RUNNER_WORKERS", "4")),
        client=client,                                  # polls for client.agent_id
    )
