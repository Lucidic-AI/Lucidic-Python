"""
Context Capture Processor for OpenTelemetry spans.

This processor captures Lucidic context (session_id, parent_event_id) at span creation time
and stores it in span attributes. This ensures context is preserved even when spans are
processed asynchronously in different threads/contexts.

This fixes the nesting issue for ALL providers (OpenAI, Anthropic, LangChain, etc.)
"""

import logging
from typing import Optional
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.trace import Span
from opentelemetry import context as otel_context

logger = logging.getLogger("Lucidic")


class ContextCaptureProcessor(SpanProcessor):
    """Captures Lucidic context at span creation and stores in attributes."""
    
    def on_start(self, span: Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """Called when a span is started - capture context here."""
        try:
            # Import here to avoid circular imports
            from lucidicai.context import current_session_id, current_parent_event_id
            
            # Capture session ID from context
            session_id = None
            try:
                session_id = current_session_id.get(None)
            except Exception:
                pass
            
            # Capture parent event ID from context
            parent_event_id = None
            try:
                parent_event_id = current_parent_event_id.get(None)
            except Exception:
                pass
            
            # Store in span attributes for later retrieval
            if session_id:
                span.set_attribute("lucidic.session_id", session_id)
            
            if parent_event_id:
                span.set_attribute("lucidic.parent_event_id", parent_event_id)
                logger.debug(f"[ContextCapture] Captured parent_event_id {parent_event_id[:8]}... for span {span.name}")
            
        except Exception as e:
            # Never fail span creation due to context capture
            logger.debug(f"[ContextCapture] Failed to capture context: {e}")
    
    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span ends - no action needed."""
        pass
    
    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush - no buffering in this processor."""
        return True