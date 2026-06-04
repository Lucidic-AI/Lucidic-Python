"""LUC-677: offline tests for the LiteLLM bridge (no live API keys).

fixtures are faithful to REAL litellm 1.87.0 standard_logging_object payloads captured via
Azure OpenAI + Anthropic (docs/telemetry/validation/capture_litellm.py).
"""
import json
from datetime import datetime, timedelta
from unittest import mock

import pytest

import lucidicai.telemetry.litellm_bridge as bridge
from lucidicai.telemetry.litellm_bridge import LucidicLiteLLMCallback


# --------------------------------------------------------------------------- #
# real-shaped fixtures
# --------------------------------------------------------------------------- #

def _slo_chat():
    return {
        "call_type": "completion",
        "custom_llm_provider": "azure",
        "model": "gpt-5.4",
        "response_cost": 0.00010750000000000001,
        "prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18,
        "messages": [{"role": "user", "content": "Say the single word: hello."}],
        "metadata": {},
        "response": {
            "choices": [{"message": {"role": "assistant", "content": "hello", "tool_calls": None}}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18,
                      "prompt_tokens_details": {"cached_tokens": 0},
                      "completion_tokens_details": {"reasoning_tokens": 0}},
        },
    }


def _slo_toolcall(stream=False):
    return {
        "call_type": "completion",
        "custom_llm_provider": "azure",
        "model": "gpt-5.4",
        "response_cost": 0.000615,
        "stream": stream,
        "prompt_tokens": 138, "completion_tokens": 18, "total_tokens": 156,
        "messages": [{"role": "user", "content": "What's the weather in Paris? Call the get_weather tool."}],
        "metadata": {},
        "response": {
            "choices": [{"message": {
                "role": "assistant", "content": None, "function_call": None,
                "tool_calls": [{
                    "function": {"arguments": '{"city":"Paris"}', "name": "get_weather"},
                    "id": "call_81TOT15WmWAGFEle9BS16GW7", "type": "function",
                }],
            }}],
            "usage": {"prompt_tokens": 138, "completion_tokens": 18, "total_tokens": 156,
                      "prompt_tokens_details": {"cached_tokens": 0},
                      "completion_tokens_details": {"reasoning_tokens": 0}},
        },
    }


def _slo_anthropic_cache_reasoning():
    """anthropic-style: cache read+creation + reasoning tokens present."""
    return {
        "call_type": "completion",
        "custom_llm_provider": "anthropic",
        "model": "claude-haiku-4-5",
        "response_cost": 3.4e-05,
        "prompt_tokens": 14, "completion_tokens": 40, "total_tokens": 54,
        "messages": [{"role": "user", "content": "Think, then say hello."}],
        # cache-creation is a PrivateAttr excluded from response.usage model_dump, so litellm
        # only exposes it in the raw provider usage under metadata.usage_object.
        "metadata": {"usage_object": {"cache_creation_input_tokens": 7, "cache_read_input_tokens": 10}},
        "response": {
            "choices": [{"message": {
                "role": "assistant", "content": "hello",
                "reasoning_content": "the user wants a greeting",
            }}],
            "usage": {"prompt_tokens": 14, "completion_tokens": 40, "total_tokens": 54,
                      "prompt_tokens_details": {"cached_tokens": 10},  # cache_creation NOT here
                      "completion_tokens_details": {"reasoning_tokens": 25}},
        },
    }


def _slo_embedding():
    return {
        "call_type": "embedding",
        "custom_llm_provider": "openai",
        "model": "text-embedding-3-small",
        "response_cost": 1e-06,
        "prompt_tokens": 8, "completion_tokens": 0, "total_tokens": 8,
        "messages": [], "metadata": {},
        "response": {"data": [{"embedding": [0.1, 0.2], "index": 0}],
                     "usage": {"prompt_tokens": 8, "total_tokens": 8}},
    }


def _times():
    start = datetime(2026, 6, 3, 12, 0, 0)
    return start, start + timedelta(seconds=1, milliseconds=500)


def _run_success(slo, kwargs_extra=None):
    """invoke log_success_event with a session active and capture the emitted event kwargs."""
    cb = LucidicLiteLLMCallback()
    kwargs = {"litellm_call_id": "c1", "model": slo["model"],
              "standard_logging_object": slo, "messages": slo.get("messages")}
    if kwargs_extra:
        kwargs.update(kwargs_extra)
    start, end = _times()
    with mock.patch.object(bridge, "get_session_id", return_value="sess-123"), \
         mock.patch.object(bridge, "emit_event") as emit:
        cb.log_success_event(kwargs, None, start, end)
    assert emit.called, "emit_event was not called"
    return emit.call_args.kwargs


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

def test_plain_chat_extracts_output_usage_cost_provider():
    ev = _run_success(_slo_chat())
    assert ev["type"] == "llm_generation"
    assert ev["output"] == "hello"
    assert ev["provider"] == "openai"          # custom_llm_provider 'azure' -> normalized via model
    assert ev["input_tokens"] == 13 and ev["output_tokens"] == 5
    assert ev["cost"] == pytest.approx(0.00010750000000000001)
    assert "tool_calls" not in ev


def test_tool_call_captured_not_response_received():
    ev = _run_success(_slo_toolcall())
    assert ev["tool_calls"] == [{"name": "get_weather", "arguments": '{"city":"Paris"}'}]
    # tool calls folded into output (content was None) — NOT "Response received"/"No content"
    assert "get_weather" in ev["output"]
    assert ev["output"] not in ("No content", "Response received", "")
    assert ev["cost"] == pytest.approx(0.000615)


def test_streaming_tool_call_uses_assembled_response():
    # streaming success carries the assembled response in standard_logging_object
    ev = _run_success(_slo_toolcall(stream=True))
    assert ev["tool_calls"] == [{"name": "get_weather", "arguments": '{"city":"Paris"}'}]
    assert "get_weather" in ev["output"]


def test_cache_and_reasoning_captured():
    ev = _run_success(_slo_anthropic_cache_reasoning())
    assert ev["provider"] == "anthropic"
    assert ev["cache"] == {"read_input_tokens": 10, "creation_input_tokens": 7}
    assert ev["thinking"] == "the user wants a greeting"
    assert ev["metadata"]["reasoning_tokens"] == 25


def test_cost_prefers_litellm_then_falls_back():
    slo = _slo_chat()
    slo["response_cost"] = None  # litellm couldn't price it
    with mock.patch.object(bridge, "calculate_cost", return_value=0.999) as calc:
        ev = _run_success(slo)
    assert calc.called
    assert ev["cost"] == 0.999


def test_provider_from_custom_llm_provider_kwarg():
    slo = _slo_chat()
    slo.pop("custom_llm_provider")
    ev = _run_success(slo, kwargs_extra={"custom_llm_provider": "anthropic", "model": "some-model"})
    assert ev["provider"] == "anthropic"


def test_non_chat_call_type_does_not_crash():
    ev = _run_success(_slo_embedding())
    assert ev["type"] == "llm_generation"
    assert "embedding" in ev["output"]
    assert "tool_calls" not in ev


def test_no_session_does_not_emit_and_no_leak():
    cb = LucidicLiteLLMCallback()
    kwargs = {"litellm_call_id": "c9", "standard_logging_object": _slo_chat()}
    cb.log_pre_api_call("gpt-5.4", [], kwargs)
    assert "c9" in cb._active_events
    start, end = _times()
    with mock.patch.object(bridge, "get_session_id", return_value=None), \
         mock.patch.object(bridge, "emit_event") as emit:
        cb.log_success_event(kwargs, None, start, end)
    assert not emit.called
    assert "c9" not in cb._active_events  # popped even with no session (leak fix)


def test_response_obj_fallback_when_slo_absent():
    """older litellm / build failure: no standard_logging_object -> parse response_obj."""
    class FakeMsg:
        content = "fallback hi"
        tool_calls = None
        function_call = None
    class FakeChoice:
        message = FakeMsg()
    class FakeResp:
        choices = [FakeChoice()]
        def model_dump(self):
            return {"choices": [{"message": {"role": "assistant", "content": "fallback hi"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    cb = LucidicLiteLLMCallback()
    kwargs = {"litellm_call_id": "c2", "model": "gpt-4o", "custom_llm_provider": "openai"}
    start, end = _times()
    with mock.patch.object(bridge, "get_session_id", return_value="s"), \
         mock.patch.object(bridge, "emit_event") as emit:
        cb.log_success_event(kwargs, FakeResp(), start, end)
    ev = emit.call_args.kwargs
    assert ev["output"] == "fallback hi"
    assert ev["provider"] == "openai"


def test_failure_emits_error_event():
    cb = LucidicLiteLLMCallback()
    kwargs = {"litellm_call_id": "c3", "model": "gpt-4o",
              "standard_logging_object": {"model": "gpt-4o", "error_str": "boom", "custom_llm_provider": "openai"}}
    start, end = _times()
    with mock.patch.object(bridge, "get_session_id", return_value="s"), \
         mock.patch.object(bridge, "emit_event") as emit:
        cb.log_failure_event(kwargs, Exception("boom"), start, end)
    ev = emit.call_args.kwargs
    assert ev["type"] == "error_traceback"
    assert "boom" in ev["error"]
