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
| `LUCIDIC_WEBHOOK_SECRET` | For webhook runs | — | Eval-run webhook signing secret (comma-separated while rotating) |

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
function on a bounded thread pool (stdlib only):

```python
import lucidicai as lucidic

client = lucidic.LucidicAI(providers=["openai"])

def run_case(run: lucidic.Run):
    with run:                    # the case's session; ends and flushes evals on exit
        return my_agent(run.input)

lucidic.serve(run_case, host="0.0.0.0", port=8080, workers=4, client=client)
```

Set `LUCIDIC_WEBHOOK_SECRET` (or pass `secret=`); without one `serve` refuses to
start. Replies: `202` accepted, `200` duplicate, `401` bad or stale signature
(keep the clock NTP-synced), `412` no secret, `413` body over `max_body`
(1 MiB), `503` full (`workers * 2` cases running or queued; the backend
re-sends). Dedupe is in memory for an hour by default; pass `dedupe=` a store
with `claim(key) -> bool` and `release(key)` to share it across replicas. In
your own server, call `lucidic.Runner(run_case).handle(body, headers)`.

## Documentation

A full reference is on the way. For now, the source under `lucidicai/` is the
authoritative documentation — start at `client.py` and the resource modules in
`lucidicai/api/resources/`.

## License

MIT.
