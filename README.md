# Lucidic AI Python SDK

`lucidicai` is the official Python observability SDK that instruments AI agents to send telemetry (sessions, events, traces) to the [Lucidic AI](https://lucidic.ai) platform for monitoring, evaluation, and optimization.

## Installation

```bash
pip install lucidicai
```

## Quick Start

```python
from lucidicai import LucidicAI
from openai import OpenAI

client = LucidicAI(
    api_key="...",          # or set LUCIDIC_API_KEY
    agent_id="...",         # or set LUCIDIC_AGENT_ID
    providers=["openai"],   # auto-instrument these providers via OpenTelemetry
)

with client.sessions.create(session_name="My Session") as session:
    openai_client = OpenAI()
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
    )
    # The OpenAI call is captured automatically as an event on this session.

client.close()
```

`LucidicAI` and `Session` both support context-manager use; the session is
bound to a `contextvar` so concurrent / async work in the same context is
attributed correctly. With the default `auto_end=True`, sessions also end on
process exit if you forget to close them explicitly.

## Configuration

Constructor arguments take precedence over environment variables.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LUCIDIC_API_KEY` | Yes | — | API authentication key |
| `LUCIDIC_AGENT_ID` | Yes | — | Agent identifier |
| `LUCIDIC_DEBUG` | No | `false` | Debug logging |
| `LUCIDIC_VERBOSE` | No | `false` | Verbose event logging |
| `LUCIDIC_PRODUCTION` | No | `false` | Suppress SDK errors in production |
| `LUCIDIC_AUTO_END` | No | `true` | Auto-end sessions on process exit |
| `LUCIDIC_REGION` | No | `"us"` | `"us"` or `"india"` |
| `LUCIDIC_BASE_URL` | No | — | Custom API URL (overrides region) |
| `LUCIDIC_REQUIRE_BASE_URL` | No | `false` | Refuse to fall back to Lucidic SaaS (see below) |
| `LUCIDIC_WEBHOOK_SECRET` | For webhook runs | — | Eval-run webhook signing secret (comma-separated while rotating) |

### Self-hosted Lucidic (BYOC)

Without a base URL the SDK sends telemetry (prompts, outputs, tool calls) to
Lucidic SaaS at `https://backend.lucidic.ai/api`. When Lucidic runs in your own
cloud account, set the URL and turn on the guard so a missing URL fails loudly
instead of sending data out of your account:

```bash
export LUCIDIC_BASE_URL="https://lucidic.internal.example.com/api"
export LUCIDIC_REQUIRE_BASE_URL=true
```

With the guard on, the SDK raises `ValueError` at `LucidicAI(...)` (or at the
first config read on the older module-level paths) if neither `base_url=` nor
`LUCIDIC_BASE_URL` is set, or if a region (`region=` or `LUCIDIC_REGION`) is
set, since a region only selects a Lucidic SaaS URL. It raises even with
`LUCIDIC_PRODUCTION=true`, before any HTTP client exists. `true`, `1`, `yes`
and `on` turn it on; `false`, `0`, `no`, `off` or unset leave it off; any
other value raises. `LUCIDIC_DEBUG=true` still wins and uses
`http://localhost:8000/api`, which never leaves the machine.

`LucidicAI(require_base_url=True)` turns the guard on for that one client;
`require_base_url=False` doesn't turn the environment variable off. A client the
SDK builds for you, such as `lucidicai.load()` without `client=`, reads only the
environment, so in a deployment prefer the environment variable.

## Supported Providers

OpenAI, Anthropic, LangChain, Google Generative AI (Gemini), Vertex AI, AWS Bedrock, Cohere, Groq, LiteLLM, Pydantic AI, OpenAI Agents.

Pass the provider names you want auto-instrumented:

```python
client = LucidicAI(providers=["openai", "anthropic"])
```

## Resources

The `LucidicAI` instance exposes resource handlers as properties:
`client.sessions`, `client.events`, `client.experiments`, `client.prompts`,
`client.datasets`, `client.feature_flags`, `client.evals`.

Use `@client.event` to track an arbitrary function as a nested event on the
current session.

## Eval runs over a webhook

Lucidic sends each eval case to your runner as a signed webhook. `lucidicai.serve`
verifies it, dedupes re-deliveries, replies `202` at once and calls your
function on a bounded thread pool (stdlib only). Signing and re-send need a
backend release that includes World Evals (LUC-997).

```python
import lucidicai as lucidic

client = lucidic.LucidicAI(providers=["openai"])

def run_case(run: lucidic.Run):
    with run:                    # the case's session; ends and flushes evals on exit
        return my_agent(run.input)

# host defaults to 127.0.0.1; 0.0.0.0 listens on all interfaces
lucidic.serve(run_case, host="0.0.0.0", port=8080, workers=4, client=client)
```

Call `serve` from the main thread. On SIGTERM or SIGINT it stops accepting,
waits for the cases already accepted, and returns. A second signal force quits:
the previous handler runs (the client still ends its sessions), then the process
exits with 128 + the signal number, even with cases still running.

Set `LUCIDIC_WEBHOOK_SECRET` (or pass `secret=`); without one `serve` refuses to
start. Replies: `202` accepted, `200` duplicate, `401` bad or stale signature
(keep the clock NTP-synced), `412` no secret, `413` body over `max_body`
(1 MiB), `503` full (`workers * 2` cases running or queued; the backend
re-sends). Dedupe is in memory for an hour by default; pass `dedupe=` a store
with `claim(key) -> bool` and `release(key)` to share it across replicas. In
your own server, call `lucidic.Runner(run_case).handle(body, headers)`.

## Eval runs by polling

Start an eval run without a `webhook_url` and it runs in pull mode: Lucidic
queues the cases and `lucidicai.poll` claims them. The poller needs outbound
access only, so it suits a runner behind a firewall, on SaaS and on a
self-hosted (BYOC) Lucidic alike. Push mode (a `webhook_url` and `serve`) stays
available. The backend needs a release with the pull queue (LUC-1279).

```python
import lucidicai as lucidic

client = lucidic.LucidicAI(providers=["openai"])

def run_case(run: lucidic.Run):  # the same run_case as for serve
    with run:
        return my_agent(run.input)

lucidic.poll(run_case, workers=4, client=client)  # polls for client.agent_id
```

```python
run = client.datasets.start_run(dataset_id, repeats=3)   # no webhook_url: pull mode
```

A run started without a `webhook_url` logs a warning: its cases wait for a
poller and fail as never delivered at the deadline. `EvalRun` and `EvoSim` carry
`delivery` (`"pull"` or `"push"`) and the queue counts `queued`, `claimed` and
`oldest_queued_seconds`.

- `agent_ids=` lists the agents to poll for; it defaults to `[client.agent_id]`.
  Agents the key cannot serve (another org's, or outside the key's agent binding)
  are logged as dropped.
- The poller builds its own client per agent (`run.client`), copying the passed
  client's API key, base URL, providers, `eval_flush_timeout` and `auto_end`, or
  reading the environment when no client is passed. These clients have
  `production=False`, so a failed session start is reported instead of hidden, and
  errors from `run.client` calls inside `run_case` (evals, prompts, events) raise
  instead of being logged. The poller closes them on exit, never the passed client.
- It claims up to the number of free workers (at most 10), heartbeats every job in
  flight every 30 s, and backs off on errors (up to 60 s; 401, 403 and a backend
  without the pull queue retry every 60 s). It never exits on an error.
- A case whose `run_case` raised or returned before `with run:` still has its
  session opened and ended. If the session fails to open, the case is logged and
  its lease lapses, so the backend claims it again.
- `poll` requires a base URL: with neither `LUCIDIC_BASE_URL` nor a client with
  its own `base_url` it raises at startup instead of polling Lucidic SaaS. On SaaS,
  set `LUCIDIC_BASE_URL=https://backend.lucidic.ai/api` or pass
  `require_base_url=False`.
- Call it from the main thread. SIGTERM or SIGINT stops claiming, waits for the
  jobs in flight (still heartbeating them), closes the clients and returns. A
  second signal force quits.

## Releasing

Release a 4.1.x only after the backend with the pull queue is deployed, and after
`tests/e2e/test_poll_live.py` passes against the backend conformance stack: in
AnalyticsAPIBackend, `docker compose -f docker-compose.conformance.yml -p conformance up -d --wait`,
then `manage.py upgrade` and the conformance bootstrap for an API key (see
`smoke_tests/conformance.py`), then here
`LUCIDIC_E2E_BASE_URL=http://127.0.0.1:8000/api LUCIDIC_E2E_API_KEY=... pytest tests/e2e`.

## Documentation

A full reference is on the way. For now, the source under `lucidicai/` is the
authoritative documentation — start at `client.py` and the resource modules in
`lucidicai/api/resources/`.

## License

MIT.
