"""
Tests for LiveKit exporter verification.

This test suite verifies the implementation against the LiveKit agents tracing system
as documented in the verification analysis.

Test scenarios covered:
1. Basic LLM call - verify model, provider, tokens, cost appear
2. With tool calls - verify tool_calls array populated from gen_ai.choice
3. Function execution - verify function_call events with arguments/return_value
4. Missing metrics - verify fallback to detect_provider() works
5. Legacy format - verify fallback to lk.chat_ctx and lk.response.text
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


class MockEvent:
    """Mock OpenTelemetry span event."""

    def __init__(self, name: str, attributes: Dict[str, Any] = None):
        self.name = name
        self.attributes = attributes or {}
        self.timestamp = int(datetime.now(timezone.utc).timestamp() * 1e9)


class MockSpan:
    """Mock OpenTelemetry ReadableSpan for testing."""

    def __init__(
        self,
        name: str,
        attributes: Dict[str, Any] = None,
        events: List[MockEvent] = None,
        start_time: int = None,
        end_time: int = None,
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
            trace_id=0x000000000000000000000000deadbeef,
            span_id=0x00000000cafebabe,
            is_remote=False,
            trace_state=TraceState(),
        )
        self._status = Status(StatusCode.OK)
        self._kind = SpanKind.INTERNAL
        self._parent = None
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


class TestLLMRequestSpanProcessing:
    """Test llm_request span to llm_generation event conversion."""

    def test_basic_llm_call_with_genai_events(self, exporter, mock_client):
        """Test basic LLM call - verify model, provider, tokens, cost appear."""
        # create lk.llm_metrics JSON as LiveKit does
        llm_metrics = json.dumps({
            "timestamp": 1234567890.123,
            "request_id": "req-123",
            "ttft": 0.245,
            "duration": 2.456,
            "cancelled": False,
            "completion_tokens": 150,
            "prompt_tokens": 500,
            "total_tokens": 650,
            "tokens_per_second": 65.2,
            "metadata": {
                "model_name": "gpt-4o",
                "model_provider": "openai"
            }
        })

        # create span with GenAI events (as llm_request spans have)
        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 500,
                "gen_ai.usage.output_tokens": 150,
                "lk.llm_metrics": llm_metrics,
            },
            events=[
                MockEvent("gen_ai.system.message", {"content": "You are a helpful assistant."}),
                MockEvent("gen_ai.user.message", {"content": "Hello, how are you?"}),
                MockEvent("gen_ai.choice", {"content": "I'm doing well, thank you!"}),
            ]
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 1

        event = mock_client.created_events[0]

        # verify type
        assert event["type"] == "llm_generation"

        # verify model extraction (primary: gen_ai.request.model)
        assert event["model"] == "gpt-4o"

        # verify provider extraction (from lk.llm_metrics.metadata.model_provider)
        assert event["provider"] == "openai"

        # verify token extraction
        assert event["input_tokens"] == 500
        assert event["output_tokens"] == 150

        # verify cost calculation
        assert event["cost"] is not None
        assert event["cost"] > 0

        # verify messages from GenAI events
        assert len(event["messages"]) == 2  # system + user (choice is output)
        assert event["messages"][0]["role"] == "system"
        assert event["messages"][0]["content"] == "You are a helpful assistant."
        assert event["messages"][1]["role"] == "user"
        assert event["messages"][1]["content"] == "Hello, how are you?"

        # verify output from gen_ai.choice
        assert event["output"] == "I'm doing well, thank you!"

        # verify metadata from lk.llm_metrics
        assert event["metadata"]["ttft"] == 0.245
        assert event["metadata"]["tokens_per_second"] == 65.2

        # verify session_id
        assert event["session_id"] == "test-session-id"

    def test_llm_call_with_tool_calls_in_choice(self, exporter, mock_client):
        """Test LLM call with tool_calls array populated from gen_ai.choice."""
        llm_metrics = json.dumps({
            "ttft": 0.3,
            "duration": 1.5,
            "metadata": {"model_provider": "openai", "model_name": "gpt-4o"}
        })

        # tool_calls in gen_ai.choice are stored as a list of JSON strings
        tool_calls = [
            json.dumps({"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'}}),
            json.dumps({"id": "call_456", "type": "function", "function": {"name": "get_time", "arguments": '{"timezone": "EST"}'}}),
        ]

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 200,
                "gen_ai.usage.output_tokens": 50,
                "lk.llm_metrics": llm_metrics,
            },
            events=[
                MockEvent("gen_ai.user.message", {"content": "What's the weather in NYC?"}),
                MockEvent("gen_ai.choice", {"content": "", "tool_calls": tool_calls}),
            ]
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 1

        event = mock_client.created_events[0]

        # verify tool_calls are extracted
        assert "tool_calls" in event
        assert len(event["tool_calls"]) == 2
        assert event["tool_calls"][0]["function"]["name"] == "get_weather"
        assert event["tool_calls"][1]["function"]["name"] == "get_time"

    def test_llm_call_with_assistant_tool_calls_in_history(self, exporter, mock_client):
        """Test tool_calls in assistant messages (input history)."""
        llm_metrics = json.dumps({
            "ttft": 0.2,
            "duration": 1.0,
            "metadata": {"model_provider": "openai"}
        })

        # assistant message with tool_calls (from previous turn)
        assistant_tool_calls = [
            json.dumps({"id": "call_prev", "type": "function", "function": {"name": "lookup_user", "arguments": '{"id": 123}'}}),
        ]

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 300,
                "gen_ai.usage.output_tokens": 100,
                "lk.llm_metrics": llm_metrics,
            },
            events=[
                MockEvent("gen_ai.user.message", {"content": "Who is user 123?"}),
                MockEvent("gen_ai.assistant.message", {"content": "", "tool_calls": assistant_tool_calls}),
                MockEvent("gen_ai.tool.message", {"content": '{"name": "John Doe"}', "name": "lookup_user", "id": "call_prev"}),
                MockEvent("gen_ai.choice", {"content": "User 123 is John Doe."}),
            ]
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        event = mock_client.created_events[0]

        # verify messages include tool message
        assert len(event["messages"]) == 3

        # user message
        assert event["messages"][0]["role"] == "user"

        # assistant with tool_calls
        assert event["messages"][1]["role"] == "assistant"
        assert "tool_calls" in event["messages"][1]

        # tool response
        assert event["messages"][2]["role"] == "tool"
        assert event["messages"][2]["name"] == "lookup_user"
        assert event["messages"][2]["tool_call_id"] == "call_prev"

        # verify output
        assert event["output"] == "User 123 is John Doe."

    def test_provider_fallback_to_detect_provider(self, exporter, mock_client):
        """Test fallback to detect_provider() when lk.llm_metrics missing provider."""
        # llm_metrics without model_provider
        llm_metrics = json.dumps({
            "ttft": 0.5,
            "duration": 2.0,
            "metadata": {}  # no model_provider
        })

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "claude-3-5-sonnet-20241022",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 200,
                "lk.llm_metrics": llm_metrics,
            },
            events=[
                MockEvent("gen_ai.user.message", {"content": "Test"}),
                MockEvent("gen_ai.choice", {"content": "Response"}),
            ]
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        event = mock_client.created_events[0]

        # provider should be detected from model name
        assert event["provider"] == "anthropic"

    def test_model_fallback_to_llm_metrics(self, exporter, mock_client):
        """Test fallback to lk.llm_metrics.metadata.model_name when gen_ai.request.model missing."""
        llm_metrics = json.dumps({
            "ttft": 0.3,
            "duration": 1.0,
            "metadata": {
                "model_provider": "google",
                "model_name": "gemini-1.5-pro"
            }
        })

        span = MockSpan(
            name="llm_request",
            attributes={
                # no gen_ai.request.model
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 50,
                "lk.llm_metrics": llm_metrics,
            },
            events=[
                MockEvent("gen_ai.user.message", {"content": "Test"}),
                MockEvent("gen_ai.choice", {"content": "Response"}),
            ]
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        event = mock_client.created_events[0]

        # model should fallback to llm_metrics
        assert event["model"] == "gemini-1.5-pro"
        assert event["provider"] == "google"

    def test_legacy_chat_context_fallback(self, exporter, mock_client):
        """Test fallback to lk.chat_ctx when no GenAI events."""
        llm_metrics = json.dumps({
            "ttft": 0.2,
            "metadata": {"model_provider": "openai"}
        })

        chat_ctx = json.dumps({
            "items": [
                {"type": "message", "role": "system", "text_content": "You are helpful."},
                {"type": "message", "role": "user", "text_content": "Hello!"},
            ]
        })

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 50,
                "gen_ai.usage.output_tokens": 20,
                "lk.llm_metrics": llm_metrics,
                "lk.chat_ctx": chat_ctx,
                "lk.response.text": "Hi there!",
            },
            events=[]  # no GenAI events
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        event = mock_client.created_events[0]

        # messages should come from lk.chat_ctx
        assert len(event["messages"]) == 2
        assert event["messages"][0]["role"] == "system"
        assert event["messages"][0]["content"] == "You are helpful."
        assert event["messages"][1]["role"] == "user"
        assert event["messages"][1]["content"] == "Hello!"

        # output should come from lk.response.text
        assert event["output"] == "Hi there!"

    def test_legacy_response_text_fallback(self, exporter, mock_client):
        """Test fallback to lk.response.text when gen_ai.choice has no content."""
        llm_metrics = json.dumps({
            "ttft": 0.1,
            "metadata": {"model_provider": "openai"}
        })

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 30,
                "gen_ai.usage.output_tokens": 10,
                "lk.llm_metrics": llm_metrics,
                "lk.response.text": "Fallback response",
            },
            events=[
                MockEvent("gen_ai.user.message", {"content": "Test"}),
                # no gen_ai.choice event
            ]
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        event = mock_client.created_events[0]

        # output should fallback to lk.response.text
        assert event["output"] == "Fallback response"

    def test_metadata_includes_diagnostics(self, exporter, mock_client):
        """Test that metadata includes timing and diagnostic info."""
        llm_metrics = json.dumps({
            "ttft": 0.245,
            "duration": 2.456,
            "cancelled": True,
            "tokens_per_second": 65.2,
            "metadata": {"model_provider": "openai"}
        })

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 50,
                "lk.llm_metrics": llm_metrics,
                "lk.retry_count": 2,
                "lk.job_id": "job-123",
                "lk.room_name": "room-abc",
            },
            events=[
                MockEvent("gen_ai.user.message", {"content": "Test"}),
                MockEvent("gen_ai.choice", {"content": "Response"}),
            ]
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        event = mock_client.created_events[0]

        # verify diagnostics in metadata
        assert event["metadata"]["ttft"] == 0.245
        assert event["metadata"]["tokens_per_second"] == 65.2
        assert event["metadata"]["cancelled"] == True
        assert event["metadata"]["retry_count"] == 2
        assert event["metadata"]["job_id"] == "job-123"
        assert event["metadata"]["room_name"] == "room-abc"


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

        # verify type
        assert event["type"] == "function_call"

        # verify function details
        assert event["function_name"] == "get_weather"
        assert event["arguments"] == '{"location": "NYC", "unit": "celsius"}'
        assert event["return_value"] == '{"temperature": 22, "condition": "sunny"}'

        # verify tool_call_id in metadata
        assert event["metadata"]["tool_call_id"] == "call_123"

        # verify no is_error flag when False
        assert "is_error" not in event

        # verify session_id
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

        # verify is_error flag is included
        assert event["is_error"] == True
        assert event["return_value"] == "Connection timeout"

    def test_function_tool_duration_calculation(self, exporter, mock_client):
        """Test duration is calculated from span timing."""
        start_ns = int(datetime.now(timezone.utc).timestamp() * 1e9) - int(1.5e9)  # 1.5 seconds ago
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

        # duration should be approximately 1.5 seconds
        assert event["duration"] is not None
        assert 1.4 < event["duration"] < 1.6


class TestSpanFiltering:
    """Test that only relevant spans are processed."""

    def test_ignores_llm_node_span(self, exporter, mock_client):
        """Test that llm_node spans are ignored (lacks model/tokens)."""
        span = MockSpan(
            name="llm_node",
            attributes={
                "lk.chat_ctx": "{}",
                "lk.response.text": "Response",
            }
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 0

    def test_ignores_agent_session_span(self, exporter, mock_client):
        """Test that agent_session spans are ignored (lifecycle span)."""
        span = MockSpan(
            name="agent_session",
            attributes={"lk.job_id": "job-123"}
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 0

    def test_ignores_agent_turn_span(self, exporter, mock_client):
        """Test that agent_turn spans are ignored (parent context)."""
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

    def test_processes_only_llm_request_and_function_tool(self, exporter, mock_client):
        """Test that only llm_request and function_tool spans are processed."""
        llm_metrics = json.dumps({
            "ttft": 0.1,
            "metadata": {"model_provider": "openai"}
        })

        spans = [
            MockSpan(name="agent_session", attributes={}),
            MockSpan(name="agent_turn", attributes={}),
            MockSpan(name="llm_node", attributes={}),
            MockSpan(
                name="llm_request",
                attributes={
                    "gen_ai.request.model": "gpt-4o",
                    "lk.llm_metrics": llm_metrics,
                },
                events=[MockEvent("gen_ai.choice", {"content": "Response"})]
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

        # should not raise
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

        # verify session created
        mock_client.sessions.create.assert_called_once_with(
            session_id="room-123",
            session_name="Test Voice Session",
        )

        # verify provider returned
        assert result == mock_provider

        # verify batch processor added
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

        # verify two processors added (metadata + batch)
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

        # verify default session name
        mock_client.sessions.create.assert_called_once_with(
            session_id="room-456",
            session_name="LiveKit Voice Session - room-456",
        )


class TestExporterLifecycle:
    """Test exporter lifecycle methods."""

    def test_shutdown_prevents_export(self, exporter, mock_client):
        """Test that shutdown prevents further exports."""
        llm_metrics = json.dumps({
            "ttft": 0.1,
            "metadata": {"model_provider": "openai"}
        })

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "lk.llm_metrics": llm_metrics,
            },
            events=[MockEvent("gen_ai.choice", {"content": "Response"})]
        )

        exporter.shutdown()
        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 0

    def test_force_flush_returns_true(self, exporter):
        """Test that force_flush returns True."""
        assert exporter.force_flush() == True


class TestChatContextParsing:
    """Test parsing of lk.chat_ctx JSON."""

    def test_parses_items_with_text_content(self, exporter, mock_client):
        """Test parsing chat context with text_content field."""
        chat_ctx = json.dumps({
            "items": [
                {"type": "message", "role": "system", "text_content": "System prompt"},
                {"type": "message", "role": "user", "text_content": "User message"},
            ]
        })

        llm_metrics = json.dumps({"ttft": 0.1, "metadata": {"model_provider": "openai"}})

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "lk.llm_metrics": llm_metrics,
                "lk.chat_ctx": chat_ctx,
                "lk.response.text": "Response",
            },
            events=[]
        )

        exporter.export([span])

        event = mock_client.created_events[0]
        assert len(event["messages"]) == 2
        assert event["messages"][0]["content"] == "System prompt"

    def test_parses_items_with_content_array(self, exporter, mock_client):
        """Test parsing chat context with content array."""
        chat_ctx = json.dumps({
            "items": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Part 1"},
                        {"type": "text", "text": "Part 2"},
                    ]
                },
            ]
        })

        llm_metrics = json.dumps({"ttft": 0.1, "metadata": {"model_provider": "openai"}})

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "lk.llm_metrics": llm_metrics,
                "lk.chat_ctx": chat_ctx,
                "lk.response.text": "Response",
            },
            events=[]
        )

        exporter.export([span])

        event = mock_client.created_events[0]
        assert event["messages"][0]["content"] == "Part 1 Part 2"

    def test_handles_invalid_chat_ctx_json(self, exporter, mock_client):
        """Test handling of invalid JSON in chat_ctx."""
        llm_metrics = json.dumps({"ttft": 0.1, "metadata": {"model_provider": "openai"}})

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "lk.llm_metrics": llm_metrics,
                "lk.chat_ctx": "not valid json",
                "lk.response.text": "Response",
            },
            events=[]
        )

        # should not raise
        result = exporter.export([span])
        assert result == SpanExportResult.SUCCESS

        event = mock_client.created_events[0]
        assert event["messages"] == []


class TestToolCallParsing:
    """Test parsing of tool_calls in various formats."""

    def test_parses_list_of_json_strings(self, exporter, mock_client):
        """Test parsing tool_calls as list of JSON strings."""
        tool_calls = [
            json.dumps({"id": "1", "function": {"name": "func1"}}),
            json.dumps({"id": "2", "function": {"name": "func2"}}),
        ]

        llm_metrics = json.dumps({"ttft": 0.1, "metadata": {"model_provider": "openai"}})

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "lk.llm_metrics": llm_metrics,
            },
            events=[
                MockEvent("gen_ai.user.message", {"content": "Test"}),
                MockEvent("gen_ai.choice", {"content": "", "tool_calls": tool_calls}),
            ]
        )

        exporter.export([span])

        event = mock_client.created_events[0]
        assert len(event["tool_calls"]) == 2

    def test_parses_list_of_dicts(self, exporter, mock_client):
        """Test parsing tool_calls as list of dicts (already parsed)."""
        tool_calls = [
            {"id": "1", "function": {"name": "func1"}},
            {"id": "2", "function": {"name": "func2"}},
        ]

        llm_metrics = json.dumps({"ttft": 0.1, "metadata": {"model_provider": "openai"}})

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "lk.llm_metrics": llm_metrics,
            },
            events=[
                MockEvent("gen_ai.user.message", {"content": "Test"}),
                MockEvent("gen_ai.choice", {"content": "", "tool_calls": tool_calls}),
            ]
        )

        exporter.export([span])

        event = mock_client.created_events[0]
        assert len(event["tool_calls"]) == 2

    def test_handles_invalid_tool_calls_json(self, exporter, mock_client):
        """Test handling of invalid JSON in tool_calls."""
        tool_calls = ["not valid json", "also not valid"]

        llm_metrics = json.dumps({"ttft": 0.1, "metadata": {"model_provider": "openai"}})

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "lk.llm_metrics": llm_metrics,
            },
            events=[
                MockEvent("gen_ai.user.message", {"content": "Test"}),
                MockEvent("gen_ai.choice", {"content": "Response", "tool_calls": tool_calls}),
            ]
        )

        # should not raise
        result = exporter.export([span])
        assert result == SpanExportResult.SUCCESS


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_empty_span_list(self, exporter, mock_client):
        """Test handling of empty span list."""
        result = exporter.export([])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 0

    def test_handles_span_with_no_attributes(self, exporter, mock_client):
        """Test handling of llm_request span with no attributes."""
        span = MockSpan(name="llm_request", attributes={}, events=[])

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_client.created_events) == 1

        event = mock_client.created_events[0]
        assert event["model"] == "unknown"
        assert event["provider"] == "unknown"

    def test_handles_malformed_llm_metrics(self, exporter, mock_client):
        """Test handling of malformed lk.llm_metrics."""
        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "lk.llm_metrics": "not valid json",
            },
            events=[MockEvent("gen_ai.choice", {"content": "Response"})]
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        event = mock_client.created_events[0]

        # should fallback to detect_provider
        assert event["provider"] == "openai"

    def test_occurred_at_calculated_from_start_time(self, exporter, mock_client):
        """Test that occurred_at is calculated from span start time."""
        start_ns = int(datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp() * 1e9)
        end_ns = start_ns + int(1e9)

        llm_metrics = json.dumps({"ttft": 0.1, "metadata": {"model_provider": "openai"}})

        span = MockSpan(
            name="llm_request",
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "lk.llm_metrics": llm_metrics,
            },
            events=[MockEvent("gen_ai.choice", {"content": "Response"})],
            start_time=start_ns,
            end_time=end_ns,
        )

        exporter.export([span])

        event = mock_client.created_events[0]
        assert event["occurred_at"] is not None
        assert "2024-01-15" in event["occurred_at"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
