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

from ..telemetry.utils.model_pricing import calculate_cost
from ..telemetry.utils.provider import detect_provider

logger = logging.getLogger("lucidicai.integrations.livekit")


class LucidicLiveKitExporter(SpanExporter):
    """Custom OpenTelemetry exporter for LiveKit voice agent spans.

    Converts LiveKit spans into Lucidic events:
    - llm_request spans -> llm_generation events (with model, tokens, messages, output)
    - function_tool spans -> function_call events (with name, arguments, return value)

    The llm_request span contains:
    - gen_ai.request.model: Model name
    - gen_ai.usage.input_tokens / gen_ai.usage.output_tokens: Token counts
    - lk.llm_metrics: JSON with ttft, duration, tokens_per_second, metadata
    - GenAI events: gen_ai.system.message, gen_ai.user.message, gen_ai.assistant.message,
      gen_ai.tool.message (inputs), gen_ai.choice (output with optional tool_calls)

    The function_tool span contains:
    - lk.function_tool.id: Tool call ID
    - lk.function_tool.name: Function name
    - lk.function_tool.arguments: JSON arguments
    - lk.function_tool.output: Return value
    - lk.function_tool.is_error: Error flag
    """

    # livekit span names we care about
    # note: llm_request has model/provider/tokens, llm_node is parent without these
    LIVEKIT_LLM_SPANS = {"llm_request", "function_tool"}

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

    def _parse_llm_metrics(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Parse lk.llm_metrics JSON to extract provider, model, and timing info.

        Args:
            attrs: Span attributes dictionary

        Returns:
            Dict with 'provider', 'model', 'ttft', 'tokens_per_second' keys if found
        """
        llm_metrics_json = attrs.get("lk.llm_metrics")
        if not llm_metrics_json:
            return {}

        try:
            if isinstance(llm_metrics_json, str):
                metrics = json.loads(llm_metrics_json)
            else:
                metrics = llm_metrics_json

            result = {}
            metadata = metrics.get("metadata", {})

            if metadata.get("model_provider"):
                result["provider"] = metadata["model_provider"]
            if metadata.get("model_name"):
                result["model"] = metadata["model_name"]

            # extract timing and performance metrics
            if metrics.get("ttft") is not None:
                result["ttft"] = metrics["ttft"]
            if metrics.get("duration") is not None:
                result["duration"] = metrics["duration"]
            if metrics.get("tokens_per_second") is not None:
                result["tokens_per_second"] = metrics["tokens_per_second"]
            if metrics.get("cancelled") is not None:
                result["cancelled"] = metrics["cancelled"]

            return result
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"[LiveKit] Failed to parse llm_metrics: {e}")
            return {}

    def _parse_span_events(self, span: ReadableSpan) -> tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
        """Parse span events to extract messages, output, and tool calls.

        llm_request spans have GenAI events:
        - gen_ai.system.message, gen_ai.user.message, gen_ai.assistant.message (input)
        - gen_ai.tool.message (tool output)
        - gen_ai.choice (output/completion with optional tool_calls)

        Args:
            span: The OpenTelemetry span

        Returns:
            Tuple of (messages list, output string, tool_calls list)
        """
        messages: List[Dict[str, Any]] = []
        output = ""
        tool_calls: List[Dict[str, Any]] = []

        # map event names to roles
        event_to_role = {
            "gen_ai.system.message": "system",
            "gen_ai.user.message": "user",
            "gen_ai.assistant.message": "assistant",
            "gen_ai.tool.message": "tool",
        }

        if not span.events:
            return messages, output, tool_calls

        for event in span.events:
            event_name = event.name
            event_attrs = dict(event.attributes or {})

            if event_name in event_to_role:
                # message event
                role = event_to_role[event_name]
                msg: Dict[str, Any] = {"role": role}

                content = event_attrs.get("content", "")
                if content:
                    msg["content"] = content

                # handle tool_calls in assistant messages (input tool calls)
                if event_name == "gen_ai.assistant.message" and "tool_calls" in event_attrs:
                    msg["tool_calls"] = self._parse_tool_calls(event_attrs["tool_calls"])

                # handle tool message metadata
                if event_name == "gen_ai.tool.message":
                    if "name" in event_attrs:
                        msg["name"] = event_attrs["name"]
                    if "id" in event_attrs:
                        msg["tool_call_id"] = event_attrs["id"]

                messages.append(msg)

            elif event_name == "gen_ai.choice":
                # completion/output event
                content = event_attrs.get("content", "")
                if content:
                    output = content

                # extract tool_calls from completion if present
                if "tool_calls" in event_attrs:
                    tool_calls = self._parse_tool_calls(event_attrs["tool_calls"])

        return messages, output, tool_calls

    def _parse_tool_calls(self, tool_calls_attr: Any) -> List[Dict[str, Any]]:
        """Parse tool_calls attribute from GenAI events.

        Tool calls are stored as a list of JSON strings.

        Args:
            tool_calls_attr: The tool_calls attribute value (list of JSON strings)

        Returns:
            List of parsed tool call dicts
        """
        if not tool_calls_attr:
            return []

        parsed = []
        try:
            # tool_calls is a list of JSON strings
            if isinstance(tool_calls_attr, (list, tuple)):
                for tc in tool_calls_attr:
                    if isinstance(tc, str):
                        parsed.append(json.loads(tc))
                    elif isinstance(tc, dict):
                        parsed.append(tc)
            elif isinstance(tool_calls_attr, str):
                # single JSON string
                parsed.append(json.loads(tool_calls_attr))
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"[LiveKit] Failed to parse tool_calls: {e}")

        return parsed

    def _process_span(self, span: ReadableSpan) -> None:
        """Process a single LiveKit span and create corresponding Lucidic event."""
        try:
            if span.name == "llm_request":
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
        """Convert an llm_request span to llm_generation event data."""
        attrs = dict(span.attributes or {})

        # parse lk.llm_metrics for provider/model/timing
        llm_info = self._parse_llm_metrics(attrs)

        # extract model (gen_ai attribute takes precedence, fallback to metrics)
        model = attrs.get("gen_ai.request.model") or llm_info.get("model") or "unknown"

        # extract provider (from metrics first, then detect from model)
        provider = llm_info.get("provider") or detect_provider(model=model, attributes=attrs)

        # extract messages, output, and tool_calls from span events (llm_request uses GenAI events)
        messages, output, tool_calls = self._parse_span_events(span)

        # fallback to lk.chat_ctx if no events (backwards compatibility)
        if not messages:
            messages = self._parse_chat_context(attrs.get("lk.chat_ctx"))

        # fallback to lk.response.text if no output from events
        if not output:
            output = attrs.get("lk.response.text", "")

        # extract token counts
        input_tokens = attrs.get("gen_ai.usage.input_tokens")
        output_tokens = attrs.get("gen_ai.usage.output_tokens")

        # calculate cost using existing pricing utility
        cost = None
        if input_tokens is not None and output_tokens is not None:
            token_usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            cost = calculate_cost(model, token_usage)

        # build metadata with diagnostics from llm_metrics
        metadata = self._build_llm_metadata(attrs, llm_info)

        # calculate duration (prefer from metrics, fallback to span timing)
        duration = llm_info.get("duration")
        if duration is None and span.start_time and span.end_time:
            duration = (span.end_time - span.start_time) / 1e9

        # extract timing for occurred_at
        occurred_at = None
        if span.start_time:
            occurred_at = datetime.fromtimestamp(
                span.start_time / 1e9, tz=timezone.utc
            ).isoformat()

        result: Dict[str, Any] = {
            "type": "llm_generation",
            "session_id": self._session_id,
            "provider": provider,
            "model": model,
            "messages": messages,
            "output": output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "duration": duration,
            "occurred_at": occurred_at,
            "metadata": metadata,
        }

        # include tool_calls if present (LLM requested function calls)
        if tool_calls:
            result["tool_calls"] = tool_calls

        return result

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

        # extract function call details (these are the attributes actually on function_tool span)
        tool_call_id = attrs.get("lk.function_tool.id")
        is_error = attrs.get("lk.function_tool.is_error", False)

        # build metadata with tool call id
        metadata: Dict[str, Any] = {}
        if tool_call_id:
            metadata["tool_call_id"] = tool_call_id

        result: Dict[str, Any] = {
            "type": "function_call",
            "session_id": self._session_id,
            "function_name": attrs.get("lk.function_tool.name", "unknown"),
            "arguments": attrs.get("lk.function_tool.arguments"),
            "return_value": attrs.get("lk.function_tool.output"),
            "duration": duration,
            "occurred_at": occurred_at,
        }

        # include is_error flag if the tool execution failed
        if is_error:
            result["is_error"] = True

        if metadata:
            result["metadata"] = metadata

        return result

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

    def _build_llm_metadata(self, attrs: Dict[str, Any], llm_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build metadata dict with diagnostics from llm_request span attributes.

        Note: llm_request spans have limited attributes. Most metadata (job_id, room_name,
        generation_id, etc.) are on parent spans (agent_session, agent_turn) and are only
        available if a MetadataSpanProcessor propagates them.

        Args:
            attrs: Span attributes dictionary
            llm_info: Parsed lk.llm_metrics data

        Returns:
            Cleaned metadata dict with available diagnostics
        """
        metadata: Dict[str, Any] = {}

        # timing metrics from lk.llm_metrics (actually available on llm_request)
        if llm_info.get("ttft") is not None:
            metadata["ttft"] = llm_info["ttft"]
        if llm_info.get("tokens_per_second") is not None:
            metadata["tokens_per_second"] = llm_info["tokens_per_second"]
        if llm_info.get("cancelled"):
            metadata["cancelled"] = llm_info["cancelled"]

        # retry count if available (set on llm_request_run, may be propagated)
        retry_count = attrs.get("lk.retry_count")
        if retry_count is not None and retry_count > 0:
            metadata["retry_count"] = retry_count

        # attributes that may be available via MetadataSpanProcessor
        # (set by user or propagated from parent spans)
        optional_attrs = {
            "job_id": "lk.job_id",
            "room_name": "lk.room_name",
            "room_id": "room_id",
            "agent_name": "lk.agent_name",
            "participant_id": "lk.participant_id",
            "generation_id": "lk.generation_id",
            "parent_generation_id": "lk.parent_generation_id",
            "speech_id": "lk.speech_id",
            "interrupted": "lk.interrupted",
        }

        for key, attr_name in optional_attrs.items():
            value = attrs.get(attr_name)
            if value is not None:
                metadata[key] = value

        # prefer room_name over room_id
        if "room_id" in metadata and "room_name" not in metadata:
            metadata["room_name"] = metadata.pop("room_id")
        elif "room_id" in metadata:
            del metadata["room_id"]

        return metadata

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
