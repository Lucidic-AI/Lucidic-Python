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

## Documentation

A full reference is on the way. For now, the source under `lucidicai/` is the
authoritative documentation — start at `client.py` and the resource modules in
`lucidicai/api/resources/`.

## License

MIT.
