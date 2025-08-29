"""Session stamp processor for OpenTelemetry spans.

This processor stamps spans with the current session ID from context at span creation time,
ensuring proper session association even in complex async scenarios.
"""
import logging
from typing import Optional
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.context import Context
from lucidicai.context import current_session_id

logger = logging.getLogger("Lucidic")


class SessionStampProcessor(SpanProcessor):
    """Stamps spans with current session ID from context.
    
    This ensures that spans are properly associated with sessions even when
    multiple sessions are running concurrently or in async contexts.
    """
    
    def on_start(self, span: "Span", parent_context: Optional[Context] = None) -> None:
        """Stamp span with session ID at creation time.
        
        Args:
            span: The span being started
            parent_context: Optional parent context
        """
        try:
            # Get session ID from context
            session_id = current_session_id.get(None)
            if session_id:
                span.set_attribute('lucidic.session_id', session_id)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[SessionStamp] Stamped span '{span.name}' with session_id: {session_id}")
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[SessionStamp] No session_id in context for span '{span.name}'")
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[SessionStamp] Failed to stamp span: {e}")
    
    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span ends. No-op for this processor."""
        pass
    
    def shutdown(self) -> None:
        """Shutdown the processor. No-op for this processor."""
        pass
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending data. No-op for this processor.
        
        Args:
            timeout_millis: Maximum time to wait for flush
            
        Returns:
            True always as this processor doesn't buffer data
        """
        return True