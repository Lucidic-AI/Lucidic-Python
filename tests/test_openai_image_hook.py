"""Inline base64 images in prompts become a visible placeholder instead of dropping the prompt."""
import asyncio
import inspect
import json

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lucidicai.telemetry import telemetry_init
from lucidicai.telemetry.extract import extract_prompts

PNG_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


class FakeSpan:
    def __init__(self, attributes):
        self.name = "chat"
        self.attributes = attributes
        self.events = []


def test_image_hook_returns_placeholder():
    assert asyncio.run(telemetry_init._omit_base64_image("1", "2", "m.png", "AAAA")) == "[image omitted]"


def test_openai_instrumentor_gets_async_image_hook(monkeypatch):
    import opentelemetry.instrumentation.openai as openai_instrumentation

    created = []

    class RecordingInstrumentor:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def instrument(self, **kwargs):
            pass

    monkeypatch.setattr(openai_instrumentation, "OpenAIInstrumentor", RecordingInstrumentor)
    monkeypatch.setattr(telemetry_init, "_global_instrumentors", {})

    telemetry_init.instrument_providers(["openai"], TracerProvider(), {})

    [kwargs] = created
    assert inspect.iscoroutinefunction(kwargs.get("upload_base64_image"))


@pytest.mark.parametrize("image_part", [
    {"type": "uri", "modality": "image", "uri": "[image omitted]"},
    {"type": "blob", "modality": "image", "mime_type": "image/png", "content": "iVBORw0KGgo="},
])
def test_image_parts_render_as_placeholder(image_part):
    span = FakeSpan({
        "gen_ai.input.messages": json.dumps([
            {"role": "user", "parts": [{"type": "text", "content": "What is in this picture?"}, image_part]},
        ]),
    })
    assert extract_prompts(span, span.attributes) == [
        {"role": "user", "content": "What is in this picture? [image omitted]"}
    ]


def test_openai_prompt_with_base64_image_keeps_text(monkeypatch):
    import openai
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor

    def handler(request):
        return httpx.Response(200, json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "A red pixel."}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    spans = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    monkeypatch.setattr(telemetry_init, "_global_instrumentors", {})
    telemetry_init.instrument_providers(["openai"], provider, {})
    try:
        client = openai.OpenAI(
            api_key="test",
            base_url="http://llm.mock/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "What is in this picture?"},
                {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
            ]}],
        )
    finally:
        OpenAIInstrumentor().uninstrument()

    [span] = spans.get_finished_spans()
    attributes = dict(span.attributes)
    assert extract_prompts(span, attributes) == [
        {"role": "user", "content": "What is in this picture? [image omitted]"}
    ]
