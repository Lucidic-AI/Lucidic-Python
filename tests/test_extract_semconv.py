"""LUC-667: extraction across OTel GenAI semconv shapes.

fixtures are byte-faithful to spans captured live (docs/telemetry/validation/out)
from openllmetry 0.53.4 (legacy flat) and 0.61.0 (new gen_ai.input/output.messages).
"""
import json

import pytest

from lucidicai.telemetry.extract import (
    detect_is_llm_span,
    extract_prompts,
    extract_completions,
    extract_tool_calls,
    extract_model,
)
from lucidicai.telemetry.utils.provider import detect_provider


class FakeStatus:
    def __init__(self, code=None, description=None):
        from opentelemetry.trace import StatusCode
        self.status_code = code or StatusCode.UNSET
        self.description = description


class FakeEvent:
    def __init__(self, name, attributes=None):
        self.name = name
        self.attributes = attributes or {}


class FakeSpan:
    """minimal ReadableSpan stand-in for the extractors."""

    def __init__(self, name, attributes=None, events=None, status=None):
        self.name = name
        self.attributes = attributes or {}
        self.events = events or []
        self.status = status or FakeStatus()


# --------------------------------------------------------------------------- #
# fixtures — exact shapes observed on the wire
# --------------------------------------------------------------------------- #

def legacy_chat_span():
    """openllmetry 0.53.4 openai.chat — legacy flat-indexed shape."""
    return FakeSpan("openai.chat", {
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.prompt.0.role": "user",
        "gen_ai.prompt.0.content": "Say the single word: hello.",
        "gen_ai.completion.0.role": "assistant",
        "gen_ai.completion.0.content": "Hello.",
        "gen_ai.completion.0.finish_reason": "stop",
    })


def legacy_responses_toolcall_span():
    """openllmetry 0.53.4 openai.response — tool call in legacy flat shape, NO finish_reason."""
    return FakeSpan("openai.response", {
        "gen_ai.system": "Azure",
        "gen_ai.request.model": "gpt-5.4",
        "gen_ai.prompt.0.role": "system",
        "gen_ai.prompt.0.content": "You are a helpful assistant. Use tools when needed.",
        "gen_ai.prompt.1.role": "user",
        "gen_ai.prompt.1.content": "What's the weather in Paris? Call the get_weather tool.",
        "gen_ai.completion.0.role": "assistant",
        "gen_ai.completion.0.tool_calls.0.id": "fc_08ea",
        "gen_ai.completion.0.tool_calls.0.name": "get_weather",
        "gen_ai.completion.0.tool_calls.0.arguments": '{"city":"Paris"}',
        # NOTE: deliberately no gen_ai.completion.0.finish_reason
    })


def new_chat_span():
    """openllmetry 0.61.0 openai.chat — new JSON-attr shape, no flat content."""
    return FakeSpan("openai.chat", {
        "gen_ai.provider.name": "azure.ai.openai",
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "gpt-5.4",
        "gen_ai.response.model": "gpt-5.4-2026-03-05",
        "gen_ai.response.finish_reasons": ["stop"],
        "gen_ai.prompt.prompt_filter_results": "[{...azure noise...}]",
        "gen_ai.input.messages": json.dumps(
            [{"role": "user", "parts": [{"content": "Say the single word: hello.", "type": "text"}]}]
        ),
        "gen_ai.output.messages": json.dumps(
            [{"role": "assistant", "parts": [{"content": "hello", "type": "text"}], "finish_reason": "stop"}]
        ),
    })


def new_responses_toolcall_span():
    """openllmetry 0.61.0 openai.response — tool call in new JSON-attr shape."""
    return FakeSpan("openai.response", {
        "gen_ai.provider.name": "azure.ai.openai",
        "gen_ai.request.model": "gpt-5.4",
        "gen_ai.input.messages": json.dumps([
            {"role": "system", "parts": [{"type": "text", "content": "You are a helpful assistant."}]},
            {"role": "user", "parts": [{"type": "text", "content": "What's the weather in Paris?"}]},
        ]),
        "gen_ai.output.messages": json.dumps([
            {"role": "assistant",
             "parts": [{"type": "tool_call", "name": "get_weather",
                        "id": "fc_0830", "arguments": {"city": "Paris"}}],
             "finish_reason": "tool_call"}
        ]),
    })


def new_anthropic_span():
    """openllmetry 0.61.0 anthropic.chat — new JSON-attr shape (parts {type,content})."""
    return FakeSpan("anthropic.chat", {
        "gen_ai.provider.name": "anthropic",
        "gen_ai.request.model": "claude-haiku-4-5",
        "gen_ai.input.messages": json.dumps(
            [{"role": "user", "parts": [{"type": "text", "content": "Say the single word: hello."}]}]
        ),
        "gen_ai.output.messages": json.dumps(
            [{"role": "assistant", "parts": [{"type": "text", "content": "hello"}], "finish_reason": "stop"}]
        ),
    })


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #

def test_detect_is_llm_span_new_shape():
    assert detect_is_llm_span(new_chat_span()) is True
    assert detect_is_llm_span(legacy_chat_span()) is True


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #

def test_prompts_legacy_flat():
    span = legacy_chat_span()
    msgs = extract_prompts(span, span.attributes)
    assert msgs == [{"role": "user", "content": "Say the single word: hello."}]


def test_prompts_new_json_openai():
    span = new_chat_span()
    msgs = extract_prompts(span, span.attributes)
    assert msgs == [{"role": "user", "content": "Say the single word: hello."}]


def test_prompts_new_json_multimessage():
    span = new_responses_toolcall_span()
    msgs = extract_prompts(span, span.attributes)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "helpful assistant" in msgs[0]["content"]
    assert "weather in Paris" in msgs[1]["content"]


def test_prompts_new_json_anthropic_part_order():
    span = new_anthropic_span()
    msgs = extract_prompts(span, span.attributes)
    assert msgs == [{"role": "user", "content": "Say the single word: hello."}]


def test_prompts_captures_tool_result_in_history():
    """tool RESULTS (role=tool, parts=[{type:tool_call_response, response}]) must not be empty."""
    span = FakeSpan("openai.response", {
        "gen_ai.provider.name": "azure.ai.openai",
        "gen_ai.input.messages": json.dumps([
            {"role": "user", "parts": [{"type": "text", "content": "Weather in Paris?"}]},
            {"role": "assistant", "parts": [{"type": "tool_call", "name": "get_weather",
                                             "id": "call_1", "arguments": {"city": "Paris"}}]},
            {"role": "tool", "parts": [{"type": "tool_call_response", "id": "call_1",
                                        "response": '{"temp_c": 18, "sky": "clear"}'}]},
        ]),
    })
    msgs = extract_prompts(span, span.attributes)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert "get_weather" in msgs[1]["content"]            # assistant tool call rendered
    tool_msg = msgs[2]["content"]
    assert tool_msg and "18" in tool_msg and "clear" in tool_msg  # tool RESULT captured, not empty


def test_prompts_ignores_azure_prompt_filter_noise():
    """the only gen_ai.prompt.* key on a new-shape span is azure noise; must not be read as a message."""
    span = FakeSpan("openai.chat", {
        "gen_ai.provider.name": "azure.ai.openai",
        "gen_ai.prompt.prompt_filter_results": "[{...}]",
        "gen_ai.input.messages": json.dumps(
            [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]
        ),
    })
    assert extract_prompts(span, span.attributes) == [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- #
# completions
# --------------------------------------------------------------------------- #

def test_completions_legacy_flat():
    span = legacy_chat_span()
    assert extract_completions(span, span.attributes) == "Hello."


def test_completions_new_json():
    span = new_chat_span()
    assert extract_completions(span, span.attributes) == "hello"


# --------------------------------------------------------------------------- #
# tool calls
# --------------------------------------------------------------------------- #

def test_tool_calls_legacy_without_finish_reason():
    """regression: legacy tool_calls must be detected by PRESENCE, not finish_reason gate."""
    span = legacy_responses_toolcall_span()
    out = extract_tool_calls(span, span.attributes)
    assert out is not None
    assert "get_weather" in out
    assert "Paris" in out


def test_tool_calls_new_json():
    span = new_responses_toolcall_span()
    out = extract_tool_calls(span, span.attributes)
    assert out is not None
    assert "get_weather" in out
    assert "Paris" in out


def test_tool_calls_none_when_absent():
    span = legacy_chat_span()
    assert extract_tool_calls(span, span.attributes) is None


# --------------------------------------------------------------------------- #
# model + provider
# --------------------------------------------------------------------------- #

def test_extract_model_both_shapes():
    assert extract_model(new_chat_span().attributes) == "gpt-5.4-2026-03-05"
    assert extract_model(legacy_chat_span().attributes) == "gpt-4o-mini"


def test_provider_new_provider_name_normalized():
    assert detect_provider(model="gpt-5.4", attributes={"gen_ai.provider.name": "azure.ai.openai"}) == "openai"
    assert detect_provider(model="claude-haiku-4-5", attributes={"gen_ai.provider.name": "anthropic"}) == "anthropic"


def test_provider_legacy_azure_falls_through_to_model():
    """gen_ai.system='Azure' is ambiguous; should resolve via model name, not return 'azure'."""
    assert detect_provider(model="gpt-5.4", attributes={"gen_ai.system": "Azure"}) == "openai"


def test_provider_legacy_system_openai():
    assert detect_provider(model="gpt-4o-mini", attributes={"gen_ai.system": "openai"}) == "openai"
