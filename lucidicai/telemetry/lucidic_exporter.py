"""Custom OpenTelemetry exporter for Lucidic (Exporter-only mode).

Converts completed spans into immutable typed LLM events via emit_event(),
which fires events in the background without blocking the exporter.
"""
from typing import Sequence, Optional, Dict, Any, List, TYPE_CHECKING
from datetime import datetime, timezone
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.semconv_ai import SpanAttributes
import threading

from ..sdk.event import emit_event
from ..sdk.init import get_session_id
from ..sdk.context import current_session_id, current_parent_event_id
from ..telemetry.utils.model_pricing import calculate_cost
from .extract import detect_is_llm_span, extract_prompts, extract_completions, extract_model, extract_tool_calls
from .utils.provider import detect_provider
from ..utils.logger import debug, info, warning, error, verbose, truncate_id

if TYPE_CHECKING:
    from ..client import LucidicAI


class LucidicSpanExporter(SpanExporter):
    """Exporter that creates immutable LLM events for completed spans.

    Uses emit_event() for fire-and-forget event creation without blocking.
    Supports multi-client routing via client registry.
    """

    # warn loudly (but only a few times per process) when an LLM span yields no
    # extractable content - this is the silent-failure mode behind LUC-667.
    _MAX_DROP_WARNINGS = 5

    def __init__(self):
        """Initialize the exporter."""
        self._shutdown = False
        # Client registry for multi-client support
        self._client_registry: Dict[str, "LucidicAI"] = {}
        self._registry_lock = threading.Lock()
        self._dropped_span_warnings = 0

    def _warn_dropped_span(self, span: ReadableSpan, attributes: Dict[str, Any]) -> None:
        """emit a loud, actionable warning the first few times a span is dropped."""
        self._dropped_span_warnings += 1
        if self._dropped_span_warnings > self._MAX_DROP_WARNINGS:
            verbose(f"[Telemetry] Skipping span {span.name} with no meaningful content")
            return
        genai_keys = sorted(
            k for k in attributes if isinstance(k, str) and (k.startswith("gen_ai.") or k.startswith("llm."))
        )
        scope = getattr(span, "instrumentation_scope", None)
        scope_desc = ""
        if scope is not None:
            scope_desc = f" (instrumentation: {getattr(scope, 'name', '?')} {getattr(scope, 'version', '') or ''})"
        warning(
            f"[Telemetry] Dropped LLM span '{span.name}'{scope_desc}: extracted 0 messages, 0 tool calls. "
            f"This usually means the instrumentation emits a span shape this SDK version does not read yet "
            f"(e.g. a newer OTel GenAI semantic-convention). Span gen_ai/llm keys: {genai_keys}. "
            f"Upgrade lucidicai or pin the instrumentation package."
        )
        if self._dropped_span_warnings == self._MAX_DROP_WARNINGS:
            warning("[Telemetry] Further dropped-span warnings will be suppressed this process.")

    def register_client(self, client: "LucidicAI") -> None:
        """Register a client for span routing.

        Args:
            client: The LucidicAI client to register
        """
        with self._registry_lock:
            self._client_registry[client._client_id] = client
            debug(f"[Exporter] Registered client {client._client_id[:8]}...")

    def unregister_client(self, client_id: str) -> None:
        """Unregister a client.

        Args:
            client_id: The client ID to unregister
        """
        with self._registry_lock:
            self._client_registry.pop(client_id, None)
            debug(f"[Exporter] Unregistered client {client_id[:8]}...")

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            if spans:
                debug(f"[Telemetry] Processing {len(spans)} OpenTelemetry spans")
            for span in spans:
                self._process_span(span)
            if spans:
                debug(f"[Telemetry] Successfully exported {len(spans)} spans")
            return SpanExportResult.SUCCESS
        except Exception as e:
            error(f"[Telemetry] Failed to export spans: {e}")
            return SpanExportResult.FAILURE

    def _process_span(self, span: ReadableSpan) -> None:
        """Convert a single LLM span into a typed, immutable event."""
        try:
            if not detect_is_llm_span(span):
                verbose(f"[Telemetry] Skipping non-LLM span: {span.name}")
                return

            debug(f"[Telemetry] Processing LLM span: {span.name}")
            verbose(f"[Telemetry] Span: {span.attributes}")
            verbose(f"[Telemetry] Span name: {span.name}")

            attributes = dict(span.attributes or {})

            # Resolve session id
            target_session_id = attributes.get('lucidic.session_id')
            if not target_session_id:
                try:
                    target_session_id = current_session_id.get(None)
                except Exception as e:
                    debug(f"[Telemetry] Failed to get session_id from contextvar: {e}")
                    target_session_id = None
            if not target_session_id:
                target_session_id = get_session_id()
            if not target_session_id:
                debug(f"[Telemetry] No session ID for span {span.name}, skipping")
                return

            # Parent nesting - get from span attributes (captured at span creation)
            parent_id = attributes.get('lucidic.parent_event_id')
            debug(f"[Telemetry] Span {span.name} has parent_id from attributes: {truncate_id(parent_id)}")
            if not parent_id:
                # Fallback to trying context (may work if same thread)
                try:
                    parent_id = current_parent_event_id.get(None)
                    if parent_id:
                        debug(f"[Telemetry] Got parent_id from context for span {span.name}: {truncate_id(parent_id)}")
                except Exception as e:
                    debug(f"[Telemetry] Failed to get parent_event_id from contextvar: {e}")
                    parent_id = None
            
            if not parent_id:
                debug(f"[Telemetry] No parent_id available for span {span.name}")

            # Timing
            occurred_at_dt = datetime.fromtimestamp(span.start_time / 1_000_000_000, tz=timezone.utc) if span.start_time else datetime.now(tz=timezone.utc)
            occurred_at = occurred_at_dt.isoformat()  # Convert to ISO string for JSON serialization
            duration_seconds = ((span.end_time - span.start_time) / 1_000_000_000) if (span.start_time and span.end_time) else None

            # Typed fields using extract utilities
            model = extract_model(attributes) or 'unknown'
            provider = detect_provider(model=model, attributes=attributes)
            messages = extract_prompts(span, attributes) or []
            params = self._extract_params(attributes)
            output_text = extract_completions(span, attributes)
            tool_calls = extract_tool_calls(span, attributes)
            debug(f"[Telemetry] Extracted tool calls: {tool_calls}")

            # see if tool calls need to be used instead of output_text
            if not output_text or output_text == "Response received" or not tool_calls:

                if tool_calls:
                    debug(f"[Telemetry] Using tool calls for span {span.name}")
                    output_text = tool_calls

                # Only use "Response received" if we have other meaningful data
                if not messages and not tool_calls and not attributes.get("lucidic.instrumented"):
                    self._warn_dropped_span(span, attributes)
                    return
                # Use a more descriptive default if we must
                if not output_text:
                    debug(f"[Telemetry] No output text for span {span.name}. Using default 'Response received'")
                    output_text = "Response received"

            input_tokens = self._extract_prompt_tokens(attributes)
            output_tokens = self._extract_completion_tokens(attributes)
            cost = self._calculate_cost(attributes)

            # Prepare event data for async creation
            event_data = {
                'type': 'llm_generation',
                'session_id': target_session_id,
                'occurred_at': occurred_at,
                'duration': duration_seconds,
                'provider': provider,
                'model': model,
                'messages': messages,
                'params': params,
                'output': output_text,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost': cost,
                'raw': None,
                'parent_event_id': parent_id,
            }
            
            # Get client_id for routing
            client_id = attributes.get("lucidic.client_id")

            if not self._shutdown:
                self._send_event_async(event_data, span.name, parent_id, client_id)

            debug(
                f"[Telemetry] Queued LLM event creation for span {span.name} "
                f"(session: {truncate_id(target_session_id)}, client: {truncate_id(client_id)})"
            )

        except Exception as e:
            error(f"[Telemetry] Failed to process span {span.name}: {e}")
    
    def _send_event_async(
        self,
        event_data: Dict[str, Any],
        span_name: str,
        parent_id: Optional[str],
        client_id: Optional[str] = None,
    ) -> None:
        """Send event asynchronously in a background thread.

        Args:
            event_data: Event data to send
            span_name: Name of the span (for logging)
            parent_id: Parent event ID (for context)
            client_id: Client ID for routing (if available)
        """
        try:
            # Set context for parent if needed
            from ..sdk.context import current_parent_event_id as parent_context

            if parent_id:
                token = parent_context.set(parent_id)
            else:
                token = None

            try:
                # Try to route to specific client if client_id is available
                if client_id:
                    with self._registry_lock:
                        client = self._client_registry.get(client_id)
                    if client:
                        # Use client's event resource directly
                        try:
                            response = client._resources["events"].create(**event_data)
                            event_id = response if response else None
                            debug(
                                f"[Telemetry] Routed LLM event {truncate_id(event_id)} to client {client_id[:8]}..."
                            )
                            return
                        except Exception as e:
                            debug(f"[Telemetry] Failed to route event to client: {e}")
                            # Fall through to emit_event

                # Fallback to emit_event (uses global state)
                event_id = emit_event(**event_data)
                debug(
                    f"[Telemetry] Emitted LLM event {truncate_id(event_id)} from span {span_name}"
                )
            finally:
                # Reset parent context
                if token:
                    parent_context.reset(token)

        except Exception as e:
            error(f"[Telemetry] Failed to send event for span {span_name}: {e}")


    def _extract_params(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "temperature": attributes.get('gen_ai.request.temperature'),
            "max_tokens": attributes.get('gen_ai.request.max_tokens'),
            "top_p": attributes.get('gen_ai.request.top_p'),
        }

    def _extract_prompt_tokens(self, attributes: Dict[str, Any]) -> int:
        # Check each attribute and return the first non-None value
        value = attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)
        if value is not None:
            return value
        value = attributes.get('gen_ai.usage.prompt_tokens')
        if value is not None:
            return value
        value = attributes.get('gen_ai.usage.input_tokens')
        if value is not None:
            return value
        return 0

    def _extract_completion_tokens(self, attributes: Dict[str, Any]) -> int:
        # Check each attribute and return the first non-None value
        value = attributes.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS)
        if value is not None:
            return value
        value = attributes.get('gen_ai.usage.completion_tokens')
        if value is not None:
            return value
        value = attributes.get('gen_ai.usage.output_tokens')
        if value is not None:
            return value
        return 0
    
    def _calculate_cost(self, attributes: Dict[str, Any]) -> Optional[float]:
        prompt_tokens = self._extract_prompt_tokens(attributes)
        completion_tokens = self._extract_completion_tokens(attributes)
        total_tokens = prompt_tokens + completion_tokens
        if total_tokens > 0:
            model = (
                attributes.get(SpanAttributes.LLM_RESPONSE_MODEL) or
                attributes.get(SpanAttributes.LLM_REQUEST_MODEL) or
                attributes.get('gen_ai.response.model') or
                attributes.get('gen_ai.request.model')
            )
            if model:
                usage = {"prompt_tokens": prompt_tokens or 0, "completion_tokens": completion_tokens or 0, "total_tokens": total_tokens}
                return calculate_cost(model, usage)
        return None
    
    def shutdown(self) -> None:
        """Shutdown the exporter and flush pending events."""
        from ..sdk.event import flush
        self._shutdown = True
        # Flush any pending background events
        flush(timeout=5.0)
        debug("[Telemetry] LucidicSpanExporter shutdown complete")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush is a no-op since events are sent immediately in background threads."""
        return True
