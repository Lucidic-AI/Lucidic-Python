"""
Tests for LiveKit exporter verification.

This test suite verifies the implementation against the LiveKit agents tracing system
as documented in the verification analysis.

Test scenarios covered:
1. Hybrid span processing - llm_request cached, llm_node creates events
2. Messages from lk.chat_ctx (restored familiar format)
3. Model/provider/tokens from cached llm_request data
4. Provider hostname normalization (e.g., "api.openai.com" -> "openai")
5. Function execution - verify function_call events with arguments/return_value
6. Edge cases - llm_node without llm_request, unknown provider hostnames
"""

import json
import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode
from opentelemetry.trace.span import TraceState

from lucidicai.integrations.livekit import (
    LucidicLiveKitExporter,
    _MetadataSpanProcessor,
    setup_livekit,
)
from lucidicai.telemetry.utils.provider import normalize_provider


class MockEvent:
    """Mock OpenTelemetry span event."""

    def __init__(self, name: str, attributes: Dict[str, Any] = None):
        self.name = name
        self.attributes = attributes or {}
        self.timestamp = int(datetime.now(timezone.utc).timestamp() * 1e9)


class MockSpanContext:
    """Mock SpanContext for parent reference."""

    def __init__(self, span_id: int):
        self.span_id = span_id


class MockSpan:
    """Mock OpenTelemetry ReadableSpan for testing."""

    def __init__(
        self,
        name: str,
        attributes: Dict[str, Any] = None,
        events: List[MockEvent] = None,
        start_time: int = None,
        end_time: int = None,
        trace_id: int = 0x000000000000000000000000deadbeef,
        span_id: int = 0x00000000cafebabe,
        parent_span_id: int = None,
    ):
        self.name = name
        self._attributes = attributes or {}
        self._events = events or []

        # default timestamps in nanoseconds
        now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
        self._start_time = start_time or now_ns - int(1e9)  # 1 second ago
        self._end_time = end_time or now_ns

        # required span attributes
        self._context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_state=TraceState(),
        )
        self._status = Status(StatusCode.OK)
        self._kind = SpanKind.INTERNAL
        self._parent = MockSpanContext(parent_span_id) if parent_span_id else None
        self._resource = MagicMock()
        self._instrumentation_scope = MagicMock()

    @property
    def attributes(self) -> Dict[str, Any]:
        return self._attributes

    @property
    def events(self) -> List[MockEvent]:
        return self._events

    @property
    def start_time(self) -> int:
        return self._start_time

    @property
    def end_time(self) -> int:
        return self._end_time

    @property
    def context(self):
        return self._context

    @property
    def status(self):
        return self._status

    @property
    def kind(self):
        return self._kind

    @property
    def parent(self):
        return self._parent

    @property
    def resource(self):
        return self._resource

    @property
    def instrumentation_scope(self):
        return self._instrumentation_scope


class MockLucidicClient:
    """Mock LucidicAI client for testing."""

    def __init__(self):
        self.events = MagicMock()
        self.sessions = MagicMock()
        self.created_events = []

    def _capture_event(self, **kwargs):
        self.created_events.append(kwargs)
        return "event-123"


@pytest.fixture
def mock_client():
    """Create a mock Lucidic client."""
    client = MockLucidicClient()
    client.events.create = client._capture_event
    return client


@pytest.fixture
def exporter(mock_client):
    """Create a LucidicLiveKitExporter with mock client."""
    return LucidicLiveKitExporter(mock_client, "test-session-id")


class TestHybridSpanProcessing:
    """Test hybrid llm_request -> llm_node span processing."""

    def test_llm_request_caches_data_no_event(self, exporter, mock_client):
        """Test that llm_request spans cache data but don't create events."""
        llm_metrics = json.dumps({
            "ttft": 0.245,
            "duration": 2.456,
            "tokens_per_second": 65.2,
            "metadata": {"model_provider": "openai", "model_name": "gpt-4o"}
        })

        # llm_request span (child of llm_node)
        llm_request_span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 500,
                "gen_ai.usage.output_tokens": 150,
                "lk.llm_metrics": llm_metrics,
            },
            events=[
                MockEvent("gen_ai.choice", {"content": "Response from LLM"}),
            ],
            span_id=0x000000000000c001,
            parent_span_id=0x000000000000a001,
        )

        result = exporter.export([llm_request_span])

        assert result == SpanExportResult.SUCCESS
        # no event should be created for llm_request
        assert len(mock_client.created_events) == 0
        # data should be cached
        assert len(exporter._llm_request_cache) == 1

    def test_llm_node_creates_event_with_cached_data(self, exporter, mock_client):
        """Test that llm_node spans create events using cached llm_request data."""
        llm_metrics = json.dumps({
            "ttft": 0.245,
            "duration": 2.456,
            "tokens_per_second": 65.2,
            "metadata": {"model_provider": "openai", "model_name": "gpt-4o"}
        })

        chat_ctx = json.dumps({
            "items": [
                {"type": "message", "role": "system", "text_content": "You are a helpful assistant."},
                {"type": "message", "role": "user", "text_content": "Hello, how are you?"},
            ]
        })

        # use consistent IDs
        trace_id = 0x000000000000000000000000deadbeef
        llm_node_span_id = 0x000000000000a001
        llm_request_span_id = 0x000000000000c001

        # first, export llm_request (child) - should cache data
        llm_request_span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 500,
                "gen_ai.usage.output_tokens": 150,
                "lk.llm_metrics": llm_metrics,
            },
            events=[
                MockEvent("gen_ai.choice", {"content": "I'm doing well, thank you!"}),
            ],
            trace_id=trace_id,
            span_id=llm_request_span_id,
            parent_span_id=llm_node_span_id,
        )

        # then, export llm_node (parent) - should create event with merged data
        llm_node_span = MockSpan(
            name="llm_node",
            attributes={
                "lk.chat_ctx": chat_ctx,
                "lk.response.text": "I'm doing well, thank you!",
            },
            trace_id=trace_id,
            span_id=llm_node_span_id,
        )

        # export both spans
        result = exporter.export([llm_request_span, llm_node_span])

        assert result == SpanExportResult.SUCCESS
        # only one event should be created (from llm_node)
        assert len(mock_client.created_events) == 1

        event = mock_client.created_events[0]

        # verify event type
        assert event["type"] == "llm_generation"

        # verify model from cached llm_request
        assert event["model"] == "gpt-4o"

        # verify provider from cached llm_request (normalized)
        assert event["provider"] == "openai"

        # verify tokens from cached llm_request
        assert event["input_tokens"] == 500
        assert event["output_tokens"] == 150

        # verify cost is calculated
        assert event["cost"] is not None
        assert event["cost"] > 0

        # verify messages from lk.chat_ctx (familiar format!)
        assert len(event["messages"]) == 2
        assert event["messages"][0]["role"] == "system"
        assert event["messages"][0]["content"] == "You are a helpful assistant."
        assert event["messages"][1]["role"] == "user"
        assert event["messages"][1]["content"] == "Hello, how are you?"

        # verify output from lk.response.text
        assert event["output"] == "I'm doing well, thank you!"

        # verify metadata has timing from cached llm_request
        assert event["metadata"]["ttft"] == 0.245
        assert event["metadata"]["tokens_per_second"] == 65.2

    def test_llm_node_without_cached_data_uses_fallbacks(self, exporter, mock_client):
        """Test llm_node without prior llm_request uses fallback values."""
        chat_ctx = json.dumps({
            "items": [
                {"type": "message", "role": "user", "text_content": "Hello"},
            ]
        })

        # llm_node span without any cached llm_request data
        llm_node_span = MockSpan(
            name="llm_node",
            attributes={
                "lk.chat_ctx": chat_ctx,
                "lk.response.text": "Hi there!",
            },
        )

        result = exporter.export([llm_node_span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 1

        event = mock_client.created_events[0]

        # model/provider should be unknown without cached data
        assert event["model"] == "unknown"
        assert event["provider"] == "unknown"

        # tokens should be None
        assert event["input_tokens"] is None
        assert event["output_tokens"] is None

        # messages still come from lk.chat_ctx
        assert len(event["messages"]) == 1
        assert event["messages"][0]["content"] == "Hello"

        # output still comes from lk.response.text
        assert event["output"] == "Hi there!"


class TestProviderHostnameNormalization:
    """Test provider hostname to standard name normalization."""

    def test_normalize_openai_hostname(self):
        """Test normalizing api.openai.com -> openai."""
        assert normalize_provider("api.openai.com") == "openai"

    def test_normalize_anthropic_hostname(self):
        """Test normalizing api.anthropic.com -> anthropic."""
        assert normalize_provider("api.anthropic.com") == "anthropic"

    def test_normalize_google_hostname(self):
        """Test normalizing generativelanguage.googleapis.com -> google."""
        assert normalize_provider("generativelanguage.googleapis.com") == "google"

    def test_normalize_groq_hostname(self):
        """Test normalizing api.groq.com -> groq."""
        assert normalize_provider("api.groq.com") == "groq"

    def test_normalize_already_valid_provider(self):
        """Test that already valid provider names pass through."""
        assert normalize_provider("openai") == "openai"
        assert normalize_provider("anthropic") == "anthropic"

    def test_normalize_unknown_hostname(self):
        """Test that unknown hostnames pass through."""
        assert normalize_provider("api.unknown-provider.com") == "api.unknown-provider.com"

    def test_normalize_none(self):
        """Test normalizing None returns unknown."""
        assert normalize_provider(None) == "unknown"

    def test_normalize_empty_string(self):
        """Test normalizing empty string returns unknown."""
        assert normalize_provider("") == "unknown"

    def test_llm_request_normalizes_hostname_provider(self, exporter, mock_client):
        """Test that llm_request caches normalized provider from hostname."""
        # lk.llm_metrics with hostname as provider (LiveKit's actual format)
        llm_metrics = json.dumps({
            "ttft": 0.2,
            "metadata": {"model_provider": "api.openai.com", "model_name": "gpt-4o"}
        })

        chat_ctx = json.dumps({
            "items": [{"type": "message", "role": "user", "text_content": "Test"}]
        })

        trace_id = 0x000000000000000000000000deadbeef
        llm_node_span_id = 0x000000000000a001
        llm_request_span_id = 0x000000000000c001

        llm_request_span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 50,
                "lk.llm_metrics": llm_metrics,
            },
            trace_id=trace_id,
            span_id=llm_request_span_id,
            parent_span_id=llm_node_span_id,
        )

        llm_node_span = MockSpan(
            name="llm_node",
            attributes={
                "lk.chat_ctx": chat_ctx,
                "lk.response.text": "Response",
            },
            trace_id=trace_id,
            span_id=llm_node_span_id,
        )

        exporter.export([llm_request_span, llm_node_span])

        event = mock_client.created_events[0]
        # provider should be normalized from "api.openai.com" to "openai"
        assert event["provider"] == "openai"


class TestMessagesFromChatContext:
    """Test messages extraction from lk.chat_ctx (restored familiar format)."""

    def test_parses_text_content_format(self, exporter, mock_client):
        """Test parsing lk.chat_ctx with text_content field."""
        chat_ctx = json.dumps({
            "items": [
                {"type": "message", "role": "system", "text_content": "System prompt here"},
                {"type": "message", "role": "user", "text_content": "User question here"},
                {"type": "message", "role": "assistant", "text_content": "Previous response"},
            ]
        })

        llm_node_span = MockSpan(
            name="llm_node",
            attributes={
                "lk.chat_ctx": chat_ctx,
                "lk.response.text": "New response",
            },
        )

        exporter.export([llm_node_span])

        event = mock_client.created_events[0]
        assert len(event["messages"]) == 3
        assert event["messages"][0] == {"role": "system", "content": "System prompt here"}
        assert event["messages"][1] == {"role": "user", "content": "User question here"}
        assert event["messages"][2] == {"role": "assistant", "content": "Previous response"}

    def test_parses_content_array_format(self, exporter, mock_client):
        """Test parsing lk.chat_ctx with content array format."""
        chat_ctx = json.dumps({
            "items": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Part one"},
                        {"type": "text", "text": "Part two"},
                    ]
                },
            ]
        })

        llm_node_span = MockSpan(
            name="llm_node",
            attributes={
                "lk.chat_ctx": chat_ctx,
                "lk.response.text": "Response",
            },
        )

        exporter.export([llm_node_span])

        event = mock_client.created_events[0]
        assert event["messages"][0]["content"] == "Part one Part two"

    def test_handles_invalid_chat_ctx_json(self, exporter, mock_client):
        """Test handling of invalid JSON in lk.chat_ctx."""
        llm_node_span = MockSpan(
            name="llm_node",
            attributes={
                "lk.chat_ctx": "not valid json",
                "lk.response.text": "Response",
            },
        )

        result = exporter.export([llm_node_span])

        assert result == SpanExportResult.SUCCESS
        event = mock_client.created_events[0]
        assert event["messages"] == []


class TestToolCallsFromCache:
    """Test tool_calls extraction from cached llm_request data."""

    def test_tool_calls_from_cached_gen_ai_choice(self, exporter, mock_client):
        """Test tool_calls are extracted from gen_ai.choice and included in event."""
        llm_metrics = json.dumps({
            "ttft": 0.3,
            "metadata": {"model_provider": "openai"}
        })

        chat_ctx = json.dumps({
            "items": [{"type": "message", "role": "user", "text_content": "What's the weather?"}]
        })

        tool_calls = [
            json.dumps({"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'}}),
        ]

        trace_id = 0x000000000000000000000000deadbeef
        llm_node_span_id = 0x000000000000a001
        llm_request_span_id = 0x000000000000c001

        llm_request_span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 50,
                "lk.llm_metrics": llm_metrics,
            },
            events=[
                MockEvent("gen_ai.choice", {"content": "", "tool_calls": tool_calls}),
            ],
            trace_id=trace_id,
            span_id=llm_request_span_id,
            parent_span_id=llm_node_span_id,
        )

        llm_node_span = MockSpan(
            name="llm_node",
            attributes={
                "lk.chat_ctx": chat_ctx,
                "lk.response.text": "",
            },
            trace_id=trace_id,
            span_id=llm_node_span_id,
        )

        exporter.export([llm_request_span, llm_node_span])

        event = mock_client.created_events[0]
        assert "tool_calls" in event
        assert len(event["tool_calls"]) == 1
        assert event["tool_calls"][0]["function"]["name"] == "get_weather"


class TestFunctionToolSpanProcessing:
    """Test function_tool span to function_call event conversion."""

    def test_function_tool_basic(self, exporter, mock_client):
        """Test basic function_tool span conversion."""
        span = MockSpan(
            name="function_tool",
            attributes={
                "lk.function_tool.id": "call_123",
                "lk.function_tool.name": "get_weather",
                "lk.function_tool.arguments": '{"location": "NYC", "unit": "celsius"}',
                "lk.function_tool.output": '{"temperature": 22, "condition": "sunny"}',
                "lk.function_tool.is_error": False,
            }
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 1

        event = mock_client.created_events[0]

        assert event["type"] == "function_call"
        assert event["function_name"] == "get_weather"
        assert event["arguments"] == '{"location": "NYC", "unit": "celsius"}'
        assert event["return_value"] == '{"temperature": 22, "condition": "sunny"}'
        assert event["metadata"]["tool_call_id"] == "call_123"
        assert "is_error" not in event
        assert event["session_id"] == "test-session-id"

    def test_function_tool_with_error(self, exporter, mock_client):
        """Test function_tool span with error flag."""
        span = MockSpan(
            name="function_tool",
            attributes={
                "lk.function_tool.id": "call_456",
                "lk.function_tool.name": "database_query",
                "lk.function_tool.arguments": '{"query": "SELECT * FROM users"}',
                "lk.function_tool.output": "Connection timeout",
                "lk.function_tool.is_error": True,
            }
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        event = mock_client.created_events[0]

        assert event["is_error"] == True
        assert event["return_value"] == "Connection timeout"

    def test_function_tool_duration_calculation(self, exporter, mock_client):
        """Test duration is calculated from span timing."""
        start_ns = int(datetime.now(timezone.utc).timestamp() * 1e9) - int(1.5e9)
        end_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)

        span = MockSpan(
            name="function_tool",
            attributes={
                "lk.function_tool.id": "call_789",
                "lk.function_tool.name": "slow_function",
                "lk.function_tool.arguments": "{}",
                "lk.function_tool.output": "done",
            },
            start_time=start_ns,
            end_time=end_ns,
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        event = mock_client.created_events[0]

        assert event["duration"] is not None
        assert 1.4 < event["duration"] < 1.6


class TestSpanFiltering:
    """Test that only relevant spans are processed."""

    def test_llm_node_creates_event(self, exporter, mock_client):
        """Test that llm_node spans create events."""
        chat_ctx = json.dumps({
            "items": [{"type": "message", "role": "user", "text_content": "Hello"}]
        })

        span = MockSpan(
            name="llm_node",
            attributes={
                "lk.chat_ctx": chat_ctx,
                "lk.response.text": "Hi!",
            }
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 1
        assert mock_client.created_events[0]["type"] == "llm_generation"

    def test_llm_request_does_not_create_event(self, exporter, mock_client):
        """Test that llm_request spans cache but don't create events."""
        llm_metrics = json.dumps({"ttft": 0.1, "metadata": {"model_provider": "openai"}})

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "lk.llm_metrics": llm_metrics,
            },
            events=[MockEvent("gen_ai.choice", {"content": "Response"})],
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 0

    def test_ignores_agent_session_span(self, exporter, mock_client):
        """Test that agent_session spans are ignored."""
        span = MockSpan(
            name="agent_session",
            attributes={"lk.job_id": "job-123"}
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 0

    def test_ignores_agent_turn_span(self, exporter, mock_client):
        """Test that agent_turn spans are ignored."""
        span = MockSpan(
            name="agent_turn",
            attributes={"lk.generation_id": "gen-123"}
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 0

    def test_ignores_tts_spans(self, exporter, mock_client):
        """Test that TTS spans are ignored."""
        spans = [
            MockSpan(name="tts_node", attributes={}),
            MockSpan(name="tts_request", attributes={}),
            MockSpan(name="tts_request_run", attributes={}),
        ]

        result = exporter.export(spans)

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 0

    def test_processes_llm_node_llm_request_and_function_tool(self, exporter, mock_client):
        """Test that llm_node, llm_request, and function_tool spans are processed."""
        llm_metrics = json.dumps({"ttft": 0.1, "metadata": {"model_provider": "openai"}})
        chat_ctx = json.dumps({"items": [{"type": "message", "role": "user", "text_content": "Test"}]})

        trace_id = 0x000000000000000000000000deadbeef
        llm_node_span_id = 0x000000000000a001
        llm_request_span_id = 0x000000000000c001

        spans = [
            MockSpan(name="agent_session", attributes={}),
            MockSpan(name="agent_turn", attributes={}),
            MockSpan(
                name="llm_request",
                attributes={
                    "gen_ai.request.model": "gpt-4o",
                    "lk.llm_metrics": llm_metrics,
                },
                events=[MockEvent("gen_ai.choice", {"content": "Response"})],
                trace_id=trace_id,
                span_id=llm_request_span_id,
                parent_span_id=llm_node_span_id,
            ),
            MockSpan(
                name="llm_node",
                attributes={
                    "lk.chat_ctx": chat_ctx,
                    "lk.response.text": "Response",
                },
                trace_id=trace_id,
                span_id=llm_node_span_id,
            ),
            MockSpan(
                name="function_tool",
                attributes={
                    "lk.function_tool.name": "test_func",
                    "lk.function_tool.arguments": "{}",
                }
            ),
            MockSpan(name="tts_request", attributes={}),
        ]

        result = exporter.export(spans)

        assert result == SpanExportResult.SUCCESS
        # 2 events: one llm_generation from llm_node, one function_call
        assert len(mock_client.created_events) == 2
        assert mock_client.created_events[0]["type"] == "llm_generation"
        assert mock_client.created_events[1]["type"] == "function_call"


class TestMetadataSpanProcessor:
    """Test the MetadataSpanProcessor."""

    def test_attaches_metadata_on_start(self):
        """Test that metadata is attached to spans on start."""
        metadata = {
            "customer_id": "cust-123",
            "environment": "production",
        }

        processor = _MetadataSpanProcessor(metadata)

        mock_span = MagicMock()
        processor.on_start(mock_span)

        mock_span.set_attributes.assert_called_once_with(metadata)

    def test_on_end_does_nothing(self):
        """Test that on_end doesn't do anything."""
        processor = _MetadataSpanProcessor({})
        mock_span = MagicMock()

        processor.on_end(mock_span)

    def test_force_flush_returns_true(self):
        """Test that force_flush returns True."""
        processor = _MetadataSpanProcessor({})
        assert processor.force_flush() == True


class TestSetupLiveKit:
    """Test the setup_livekit function."""

    @patch("opentelemetry.sdk.trace.export.BatchSpanProcessor")
    @patch("opentelemetry.sdk.trace.TracerProvider")
    def test_creates_session_and_tracer_provider(self, mock_provider_cls, mock_batch):
        """Test that setup creates session and returns TracerProvider."""
        mock_client = MockLucidicClient()
        mock_provider = MagicMock()
        mock_provider_cls.return_value = mock_provider

        result = setup_livekit(
            client=mock_client,
            session_id="room-123",
            session_name="Test Voice Session",
        )

        mock_client.sessions.create.assert_called_once_with(
            session_id="room-123",
            session_name="Test Voice Session",
        )

        assert result == mock_provider
        mock_provider.add_span_processor.assert_called()

    @patch("opentelemetry.sdk.trace.export.BatchSpanProcessor")
    @patch("opentelemetry.sdk.trace.TracerProvider")
    def test_adds_metadata_processor_when_metadata_provided(self, mock_provider_cls, mock_batch):
        """Test that MetadataSpanProcessor is added when metadata provided."""
        mock_client = MockLucidicClient()
        mock_provider = MagicMock()
        mock_provider_cls.return_value = mock_provider

        metadata = {"customer_id": "cust-123"}

        setup_livekit(
            client=mock_client,
            session_id="room-123",
            metadata=metadata,
        )

        assert mock_provider.add_span_processor.call_count == 2

    @patch("opentelemetry.sdk.trace.export.BatchSpanProcessor")
    @patch("opentelemetry.sdk.trace.TracerProvider")
    def test_default_session_name(self, mock_provider_cls, mock_batch):
        """Test default session name when not provided."""
        mock_client = MockLucidicClient()
        mock_provider = MagicMock()
        mock_provider_cls.return_value = mock_provider

        setup_livekit(
            client=mock_client,
            session_id="room-456",
        )

        mock_client.sessions.create.assert_called_once_with(
            session_id="room-456",
            session_name="LiveKit Voice Session - room-456",
        )


class TestExporterLifecycle:
    """Test exporter lifecycle methods."""

    def test_shutdown_prevents_export(self, exporter, mock_client):
        """Test that shutdown prevents further exports."""
        chat_ctx = json.dumps({
            "items": [{"type": "message", "role": "user", "text_content": "Test"}]
        })

        span = MockSpan(
            name="llm_node",
            attributes={
                "lk.chat_ctx": chat_ctx,
                "lk.response.text": "Response",
            }
        )

        exporter.shutdown()
        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 0

    def test_force_flush_returns_true(self, exporter):
        """Test that force_flush returns True."""
        assert exporter.force_flush() == True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_empty_span_list(self, exporter, mock_client):
        """Test handling of empty span list."""
        result = exporter.export([])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 0

    def test_handles_llm_node_with_no_attributes(self, exporter, mock_client):
        """Test handling of llm_node span with no attributes."""
        span = MockSpan(name="llm_node", attributes={}, events=[])

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 1

        event = mock_client.created_events[0]
        assert event["model"] == "unknown"
        assert event["provider"] == "unknown"
        assert event["messages"] == []

    def test_handles_malformed_llm_metrics(self, exporter, mock_client):
        """Test handling of malformed lk.llm_metrics in llm_request."""
        chat_ctx = json.dumps({
            "items": [{"type": "message", "role": "user", "text_content": "Test"}]
        })

        trace_id = 0x000000000000000000000000deadbeef
        llm_node_span_id = 0x000000000000a001
        llm_request_span_id = 0x000000000000c001

        llm_request_span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "lk.llm_metrics": "not valid json",
            },
            trace_id=trace_id,
            span_id=llm_request_span_id,
            parent_span_id=llm_node_span_id,
        )

        llm_node_span = MockSpan(
            name="llm_node",
            attributes={
                "lk.chat_ctx": chat_ctx,
                "lk.response.text": "Response",
            },
            trace_id=trace_id,
            span_id=llm_node_span_id,
        )

        result = exporter.export([llm_request_span, llm_node_span])

        assert result == SpanExportResult.SUCCESS
        event = mock_client.created_events[0]

        # should fallback to detect_provider from model
        assert event["provider"] == "openai"
        assert event["model"] == "gpt-4o"

    def test_occurred_at_calculated_from_start_time(self, exporter, mock_client):
        """Test that occurred_at is calculated from span start time."""
        start_ns = int(datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp() * 1e9)
        end_ns = start_ns + int(1e9)

        chat_ctx = json.dumps({
            "items": [{"type": "message", "role": "user", "text_content": "Test"}]
        })

        span = MockSpan(
            name="llm_node",
            attributes={
                "lk.chat_ctx": chat_ctx,
                "lk.response.text": "Response",
            },
            start_time=start_ns,
            end_time=end_ns,
        )

        exporter.export([span])

        event = mock_client.created_events[0]
        assert event["occurred_at"] is not None
        assert "2024-01-15" in event["occurred_at"]

    def test_cache_cleanup_after_retrieval(self, exporter, mock_client):
        """Test that cache is cleaned up after llm_node retrieves data."""
        llm_metrics = json.dumps({"ttft": 0.1, "metadata": {"model_provider": "openai"}})
        chat_ctx = json.dumps({"items": [{"type": "message", "role": "user", "text_content": "Test"}]})

        trace_id = 0x000000000000000000000000deadbeef
        llm_node_span_id = 0x000000000000a001
        llm_request_span_id = 0x000000000000c001

        llm_request_span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "lk.llm_metrics": llm_metrics,
            },
            trace_id=trace_id,
            span_id=llm_request_span_id,
            parent_span_id=llm_node_span_id,
        )

        # export llm_request - cache should be populated
        exporter.export([llm_request_span])
        assert len(exporter._llm_request_cache) == 1

        llm_node_span = MockSpan(
            name="llm_node",
            attributes={
                "lk.chat_ctx": chat_ctx,
                "lk.response.text": "Response",
            },
            trace_id=trace_id,
            span_id=llm_node_span_id,
        )

        # export llm_node - cache should be cleaned up
        exporter.export([llm_node_span])
        assert len(exporter._llm_request_cache) == 0

    def test_output_fallback_to_cached_when_lk_response_text_empty(self, exporter, mock_client):
        """Test output falls back to cached GenAI choice when lk.response.text is empty."""
        llm_metrics = json.dumps({"ttft": 0.1, "metadata": {"model_provider": "openai"}})
        chat_ctx = json.dumps({"items": [{"type": "message", "role": "user", "text_content": "Test"}]})

        trace_id = 0x000000000000000000000000deadbeef
        llm_node_span_id = 0x000000000000a001
        llm_request_span_id = 0x000000000000c001

        llm_request_span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "lk.llm_metrics": llm_metrics,
            },
            events=[
                MockEvent("gen_ai.choice", {"content": "Fallback output from events"}),
            ],
            trace_id=trace_id,
            span_id=llm_request_span_id,
            parent_span_id=llm_node_span_id,
        )

        llm_node_span = MockSpan(
            name="llm_node",
            attributes={
                "lk.chat_ctx": chat_ctx,
                # no lk.response.text
            },
            trace_id=trace_id,
            span_id=llm_node_span_id,
        )

        exporter.export([llm_request_span, llm_node_span])

        event = mock_client.created_events[0]
        assert event["output"] == "Fallback output from events"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
