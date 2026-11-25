"""LLM events built by the span exporter keep the tool calls the model asked for."""
import json
from types import SimpleNamespace

from opentelemetry.trace import StatusCode

from lucidicai.sdk.event_builder import EventBuilder
from lucidicai.telemetry.extract import _format_tool_calls
from lucidicai.telemetry.lucidic_exporter import LucidicSpanExporter


class FakeStatus:
    status_code = StatusCode.UNSET
    description = None


class FakeSpan:
    def __init__(self, attributes):
        self.name = "openai.chat"
        self.attributes = attributes
        self.events = []
        self.status = FakeStatus()
        self.start_time = 1_000_000_000
        self.end_time = 2_000_000_000


class FakeEvents:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return "evt-1"


ROUTING = {"lucidic.session_id": "sess-1", "lucidic.client_id": "c1"}
PROMPT_NEW = json.dumps([{"role": "user", "parts": [{"type": "text", "content": "Weather in Paris?"}]}])
CALL_NEW = {"name": "get_weather", "arguments": {"city": "Paris"}}
CALL_LEGACY = {"name": "get_weather", "arguments": '{"city":"Paris"}'}


def _export(attributes):
    events = FakeEvents()
    exporter = LucidicSpanExporter()
    exporter.register_client(SimpleNamespace(_client_id="c1", _resources={"events": events}))
    exporter._process_span(FakeSpan({**ROUTING, **attributes}))
    return events.calls


def _new_shape(*output_parts):
    return {
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.input.messages": PROMPT_NEW,
        "gen_ai.output.messages": json.dumps([{"role": "assistant", "parts": list(output_parts)}]),
    }


def _text_and_tool_call_new_shape():
    return _new_shape(
        {"type": "text", "content": "Let me look that up."},
        {"type": "tool_call", "id": "call_1", **CALL_NEW},
    )


def test_text_and_tool_calls_new_shape_both_kept():
    [kw] = _export(_text_and_tool_call_new_shape())
    assert kw["tool_calls"] == [CALL_NEW]
    assert kw["output"].startswith("Let me look that up.")
    assert "Tool Call:" in kw["output"]
    assert "get_weather" in kw["output"]


def test_text_and_tool_calls_legacy_shape_both_kept():
    [kw] = _export({
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.prompt.0.role": "user",
        "gen_ai.prompt.0.content": "Weather in Paris?",
        "gen_ai.completion.0.role": "assistant",
        "gen_ai.completion.0.content": "Let me look that up.",
        "gen_ai.completion.0.tool_calls.0.id": "call_1",
        "gen_ai.completion.0.tool_calls.0.name": "get_weather",
        "gen_ai.completion.0.tool_calls.0.arguments": '{"city":"Paris"}',
    })
    assert kw["tool_calls"] == [CALL_LEGACY]
    assert kw["output"].startswith("Let me look that up.")
    assert "Tool Call:" in kw["output"]
    assert "get_weather" in kw["output"]


def test_tool_calls_only_reply():
    [kw] = _export(_new_shape({"type": "tool_call", "id": "call_1", **CALL_NEW}))
    assert kw["output"] == _format_tool_calls([CALL_NEW])
    assert kw["tool_calls"] == [CALL_NEW]


def test_text_only_reply_has_no_tool_calls_key():
    [kw] = _export(_new_shape({"type": "text", "content": "hello"}))
    assert kw["output"] == "hello"
    assert "tool_calls" not in kw


def test_span_with_no_messages_or_tool_calls_is_dropped():
    calls = _export({
        "gen_ai.system": "openai",
        "gen_ai.completion.0.role": "assistant",
        "gen_ai.completion.0.content": "hello",
    })
    assert calls == []


def test_tool_calls_reach_payload_response():
    [kw] = _export(_text_and_tool_call_new_shape())
    built = EventBuilder.build({**kw, "type": "llm_generation"})
    assert built["payload"]["response"]["tool_calls"] == [CALL_NEW]
