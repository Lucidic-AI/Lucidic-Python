"""LiveKit voice agent integration for Lucidic AI SDK.

This module provides OpenTelemetry span export for LiveKit voice agents,
converting LiveKit's internal spans into Lucidic events with full metadata
support including latency diagnostics, EOU detection data, and tool context.

Example:
    from lucidicai import LucidicAI
    from lucidicai.integrations.livekit import setup_livekit
    from livekit.agents import AgentServer, JobContext, AgentSession, cli
    from livekit.agents.telemetry import set_tracer_provider

    client = LucidicAI(api_key="...", agent_id="...")
    server = AgentServer()

    @server.rtc_session()
    async def entrypoint(ctx: JobContext):
        trace_provider = setup_livekit(
            client=client,
            session_id=ctx.room.name,
        )
        set_tracer_provider(trace_provider)
        # ... rest of agent setup
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from ..client import LucidicAI

logger = logging.getLogger("lucidicai.integrations.livekit")


class LucidicLiveKitExporter(SpanExporter):
    """Custom OpenTelemetry exporter for LiveKit voice agent spans.

    Converts LiveKit spans (llm_node, function_tool) into Lucidic events
    with full metadata including latency diagnostics, EOU detection,
    and tool context.
    """

    # livekit span names we care about
    LIVEKIT_LLM_SPANS = {"llm_node", "function_tool"}

    def __init__(self, client: "LucidicAI", session_id: str):
        """Initialize the exporter.

        Args:
            client: Initialized LucidicAI client instance
            session_id: Session ID for all events created by this exporter
        """
        self._client = client
        self._session_id = session_id
        self._shutdown = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to Lucidic as events.

        Args:
            spans: Sequence of completed OpenTelemetry spans

        Returns:
            SpanExportResult indicating success or failure
        """
        if self._shutdown:
            return SpanExportResult.SUCCESS

        try:
            for span in spans:
                if self._is_livekit_llm_span(span):
                    self._process_span(span)
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"[LiveKit] Failed to export spans: {e}")
            return SpanExportResult.FAILURE

    def _is_livekit_llm_span(self, span: ReadableSpan) -> bool:
        """Check if span is a LiveKit LLM-related span we should process."""
        return span.name in self.LIVEKIT_LLM_SPANS

    def _process_span(self, span: ReadableSpan) -> None:
        """Process a single LiveKit span and create corresponding Lucidic event."""
        try:
            if span.name == "llm_node":
                event_data = self._convert_llm_span(span)
                self._client.events.create(**event_data)
                logger.debug(f"[LiveKit] Created llm_generation event for span {span.name}")
            elif span.name == "function_tool":
                event_data = self._convert_function_span(span)
                self._client.events.create(**event_data)
                logger.debug(f"[LiveKit] Created function_call event for span {span.name}")
        except Exception as e:
            logger.error(f"[LiveKit] Failed to process span {span.name}: {e}")

    def _convert_llm_span(self, span: ReadableSpan) -> Dict[str, Any]:
        """Convert an llm_node span to llm_generation event data."""
        attrs = dict(span.attributes or {})

        # extract messages from chat context
        messages = self._parse_chat_context(attrs.get("lk.chat_ctx"))

        # extract output text
        output = attrs.get("lk.response.text", "")

        # build metadata with diagnostics
        metadata = self._build_metadata(attrs)

        # calculate duration
        duration = None
        if span.start_time and span.end_time:
            duration = (span.end_time - span.start_time) / 1e9

        # extract timing for occurred_at
        occurred_at = None
        if span.start_time:
            occurred_at = datetime.fromtimestamp(
                span.start_time / 1e9, tz=timezone.utc
            ).isoformat()

        return {
            "type": "llm_generation",
            "session_id": self._session_id,
            "model": attrs.get("gen_ai.request.model", "unknown"),
            "messages": messages,
            "output": output,
            "input_tokens": attrs.get("gen_ai.usage.input_tokens"),
            "output_tokens": attrs.get("gen_ai.usage.output_tokens"),
            "duration": duration,
            "occurred_at": occurred_at,
            "metadata": metadata,
        }

    def _convert_function_span(self, span: ReadableSpan) -> Dict[str, Any]:
        """Convert a function_tool span to function_call event data."""
        attrs = dict(span.attributes or {})

        # calculate duration
        duration = None
        if span.start_time and span.end_time:
            duration = (span.end_time - span.start_time) / 1e9

        # extract timing for occurred_at
        occurred_at = None
        if span.start_time:
            occurred_at = datetime.fromtimestamp(
                span.start_time / 1e9, tz=timezone.utc
            ).isoformat()

        # build metadata (subset for function calls)
        metadata = {
            "job_id": attrs.get("lk.job_id"),
            "room_name": attrs.get("lk.room_name") or attrs.get("room_id"),
            "agent_name": attrs.get("lk.agent_name"),
            "generation_id": attrs.get("lk.generation_id"),
            "tool_call_id": attrs.get("lk.function_tool.id"),
        }
        metadata = self._clean_none_values(metadata)

        return {
            "type": "function_call",
            "session_id": self._session_id,
            "function_name": attrs.get("lk.function_tool.name", "unknown"),
            "arguments": attrs.get("lk.function_tool.arguments"),
            "return_value": attrs.get("lk.function_tool.output"),
            "duration": duration,
            "occurred_at": occurred_at,
            "metadata": metadata,
        }

    def _parse_chat_context(self, chat_ctx_json: Optional[str]) -> List[Dict[str, str]]:
        """Parse LiveKit's lk.chat_ctx JSON into Lucidic messages format.

        Args:
            chat_ctx_json: JSON string of LiveKit chat context

        Returns:
            List of message dicts with role and content keys
        """
        if not chat_ctx_json:
            return []

        try:
            chat_ctx = json.loads(chat_ctx_json)
            messages = []

            # livekit chat context has 'items' list
            items = chat_ctx.get("items", [])
            for item in items:
                if item.get("type") == "message":
                    role = item.get("role", "user")
                    # livekit stores content in various ways
                    content = item.get("text_content", "")
                    if not content:
                        # try content array
                        content_list = item.get("content", [])
                        if isinstance(content_list, list):
                            text_parts = []
                            for c in content_list:
                                if isinstance(c, str):
                                    text_parts.append(c)
                                elif isinstance(c, dict) and c.get("type") == "text":
                                    text_parts.append(c.get("text", ""))
                            content = " ".join(text_parts)
                        elif isinstance(content_list, str):
                            content = content_list

                    messages.append({"role": role, "content": content})

            return messages
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"[LiveKit] Failed to parse chat context: {e}")
            return []

    def _build_metadata(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Build metadata dict with diagnostics from span attributes.

        Args:
            attrs: Span attributes dictionary

        Returns:
            Cleaned metadata dict with nested diagnostics
        """
        metadata = {
            # identity & tracking
            "job_id": attrs.get("lk.job_id"),
            "room_name": attrs.get("lk.room_name") or attrs.get("room_id"),
            "agent_name": attrs.get("lk.agent_name"),
            "participant_id": attrs.get("lk.participant_id"),
            "generation_id": attrs.get("lk.generation_id"),
            "parent_generation_id": attrs.get("lk.parent_generation_id"),
            "speech_id": attrs.get("lk.speech_id"),
            "interrupted": attrs.get("lk.interrupted"),
            # diagnostics (nested)
            "diagnostics": {
                "latency": {
                    "llm_ttft": attrs.get("llm_node_ttft"),
                    "tts_ttfb": attrs.get("tts_node_ttfb"),
                    "e2e_latency": attrs.get("e2e_latency"),
                    "transcription_delay": attrs.get("lk.transcription_delay"),
                    "end_of_turn_delay": attrs.get("lk.end_of_turn_delay"),
                },
                "eou": {
                    "probability": attrs.get("lk.eou.probability"),
                    "threshold": attrs.get("lk.eou.unlikely_threshold"),
                    "delay": attrs.get("lk.eou.endpointing_delay"),
                    "language": attrs.get("lk.eou.language"),
                },
                "tools": {
                    "function_tools": attrs.get("lk.function_tools"),
                    "provider_tools": attrs.get("lk.provider_tools"),
                    "tool_sets": attrs.get("lk.tool_sets"),
                },
                "session_options": attrs.get("lk.session_options"),
            },
        }
        return self._clean_none_values(metadata)

    def _clean_none_values(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively remove None values and empty dicts.

        Args:
            d: Dictionary to clean

        Returns:
            Cleaned dictionary with no None values or empty nested dicts
        """
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, dict):
                nested = self._clean_none_values(v)
                if nested:  # only include non-empty dicts
                    cleaned[k] = nested
            elif v is not None:
                cleaned[k] = v
        return cleaned

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._shutdown = True
        logger.debug("[LiveKit] Exporter shutdown")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending exports.

        Returns:
            True (events are created synchronously)
        """
        return True


class _MetadataSpanProcessor(SpanProcessor):
    """Span processor that adds metadata to all spans.

    This allows users to attach custom metadata (e.g., customer_id, environment)
    that will be included on every span exported.
    """

    def __init__(self, metadata: Dict[str, AttributeValue]):
        """Initialize with metadata to attach.

        Args:
            metadata: Dictionary of metadata key-value pairs
        """
        self._metadata = metadata

    def on_start(
        self, span: Span, parent_context: Optional[otel_context.Context] = None
    ) -> None:
        """Called when a span is started - attach metadata."""
        span.set_attributes(self._metadata)

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span ends - no action needed."""
        pass

    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush - no buffering in this processor."""
        return True


def setup_livekit(
    client: "LucidicAI",
    session_id: str,
    session_name: Optional[str] = None,
    metadata: Optional[Dict[str, AttributeValue]] = None,
) -> "TracerProvider":
    """Set up Lucidic tracing for LiveKit voice agents.

    Automatically creates a Lucidic session and configures OpenTelemetry
    to export LiveKit spans as Lucidic events.

    Args:
        client: Initialized LucidicAI client instance
        session_id: Session ID for all events (typically ctx.room.name)
        session_name: Optional human-readable session name
        metadata: Optional metadata to attach to all spans (e.g., customer_id)

    Returns:
        TracerProvider to pass to livekit's set_tracer_provider()

    Example:
        from lucidicai import LucidicAI
        from lucidicai.integrations.livekit import setup_livekit
        from livekit.agents import AgentServer, JobContext, AgentSession, cli
        from livekit.agents.telemetry import set_tracer_provider

        client = LucidicAI(api_key="...", agent_id="...")
        server = AgentServer()

        @server.rtc_session()
        async def entrypoint(ctx: JobContext):
            trace_provider = setup_livekit(
                client=client,
                session_id=ctx.room.name,
                session_name=f"Voice Call - {ctx.room.name}",
            )
            set_tracer_provider(trace_provider)

            async def cleanup():
                trace_provider.force_flush()
            ctx.add_shutdown_callback(cleanup)

            session = AgentSession(...)
            await session.start(agent=MyAgent(), room=ctx.room)

        if __name__ == "__main__":
            cli.run_app(server)
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # auto-create Lucidic session
    client.sessions.create(
        session_id=session_id,
        session_name=session_name or f"LiveKit Voice Session - {session_id}",
    )
    logger.info(f"[LiveKit] Created Lucidic session: {session_id}")

    # create exporter
    exporter = LucidicLiveKitExporter(client, session_id)

    # create tracer provider
    trace_provider = TracerProvider()

    # add metadata processor if metadata provided
    if metadata:
        trace_provider.add_span_processor(_MetadataSpanProcessor(metadata))

    # add exporter via batch processor
    trace_provider.add_span_processor(BatchSpanProcessor(exporter))

    logger.info("[LiveKit] Lucidic tracing configured")
    return trace_provider
