"""
Context Capture Processor for OpenTelemetry spans.

This processor captures Lucidic context (session_id, parent_event_id) at span creation time
and stores it in span attributes. This ensures context is preserved even when spans are
processed asynchronously in different threads/contexts.

This fixes the nesting issue for ALL providers (OpenAI, Anthropic, LangChain, etc.)
"""

from typing import Optional
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.trace import Span
from opentelemetry import context as otel_context
from ..utils.logger import debug, verbose, truncate_id


class ContextCaptureProcessor(SpanProcessor):
    """Captures Lucidic context at span creation and stores in attributes."""
    
    def on_start(self, span: Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """Called when a span is started - capture context here."""
        try:
            # Import here to avoid circular imports
            from lucidicai.sdk.context import current_session_id, current_parent_event_id
            from .context_bridge import extract_lucidic_context
            
            # Try to get from contextvars first
            session_id = None
            parent_event_id = None
            
            try:
                session_id = current_session_id.get(None)
            except Exception as e:
                debug(f"[ContextCapture] Failed to get session_id from contextvar: {e}")

            try:
                parent_event_id = current_parent_event_id.get(None)
            except Exception as e:
                debug(f"[ContextCapture] Failed to get parent_event_id from contextvar: {e}")
            
            # If not found in contextvars, try OpenTelemetry baggage
            # This handles cases where spans are created in different threads
            if not session_id or not parent_event_id:
                baggage_session, baggage_parent = extract_lucidic_context(parent_context)
                if not session_id and baggage_session:
                    session_id = baggage_session
                    debug(f"[ContextCapture] Got session_id from OTel baggage for span {span.name}")
                if not parent_event_id and baggage_parent:
                    parent_event_id = baggage_parent
                    debug(f"[ContextCapture] Got parent_event_id from OTel baggage for span {span.name}")
            
            # Add debug logging to understand context propagation
            debug(f"[ContextCapture] Processing span '{span.name}' - session: {truncate_id(session_id)}, parent: {truncate_id(parent_event_id)}")
            
            # Store in span attributes for later retrieval
            if session_id:
                span.set_attribute("lucidic.session_id", session_id)
                debug(f"[ContextCapture] Set session_id attribute for span {span.name}")

            if parent_event_id:
                span.set_attribute("lucidic.parent_event_id", parent_event_id)
                debug(f"[ContextCapture] Captured parent_event_id {truncate_id(parent_event_id)} for span {span.name}")
            else:
                debug(f"[ContextCapture] No parent_event_id available for span {span.name}")

            # Capture client_id for multi-client routing
            from lucidicai.sdk.context import get_active_client
            client = get_active_client()
            if client:
                span.set_attribute("lucidic.client_id", client._client_id)
                debug(f"[ContextCapture] Set client_id attribute for span {span.name}")
            
        except Exception as e:
            # Never fail span creation due to context capture
            verbose(f"[ContextCapture] Failed to capture context: {e}")
    
    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span ends - no action needed."""
        pass
    
    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush - no buffering in this processor."""
        return True