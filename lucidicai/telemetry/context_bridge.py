"""Bridge between Lucidic context and OpenTelemetry context.

This module ensures that Lucidic's contextvars (session_id, parent_event_id)
are properly propagated through OpenTelemetry's context system, which is
necessary for instrumentors that create spans in different execution contexts.
"""

from typing import Optional
from opentelemetry import baggage, context as otel_context
from opentelemetry.trace import set_span_in_context
from ..utils.logger import debug, verbose, truncate_id


def inject_lucidic_context() -> otel_context.Context:
    """Inject Lucidic context into OpenTelemetry baggage.
    
    This ensures that our context variables are available to any spans
    created by OpenTelemetry instrumentors, even if they run in different
    threads or async contexts.
    
    Returns:
        OpenTelemetry Context with Lucidic values in baggage
    """
    try:
        from ..sdk.context import current_session_id, current_parent_event_id
        
        ctx = otel_context.get_current()
        
        # Get Lucidic context values
        session_id = None
        parent_event_id = None
        
        try:
            session_id = current_session_id.get(None)
        except Exception:
            pass
            
        try:
            parent_event_id = current_parent_event_id.get(None)
        except Exception:
            pass
        
        # Inject into OpenTelemetry baggage
        if session_id:
            ctx = baggage.set_baggage("lucidic.session_id", session_id, context=ctx)
            debug(f"[ContextBridge] Injected session_id {truncate_id(session_id)} into OTel baggage")
            
        if parent_event_id:
            ctx = baggage.set_baggage("lucidic.parent_event_id", parent_event_id, context=ctx)
            debug(f"[ContextBridge] Injected parent_event_id {truncate_id(parent_event_id)} into OTel baggage")
        
        return ctx
        
    except Exception as e:
        verbose(f"[ContextBridge] Failed to inject context: {e}")
        return otel_context.get_current()


def extract_lucidic_context(ctx: Optional[otel_context.Context] = None) -> tuple[Optional[str], Optional[str]]:
    """Extract Lucidic context from OpenTelemetry baggage.
    
    Args:
        ctx: OpenTelemetry context (uses current if not provided)
        
    Returns:
        Tuple of (session_id, parent_event_id)
    """
    if ctx is None:
        ctx = otel_context.get_current()
    
    try:
        session_id = baggage.get_baggage("lucidic.session_id", context=ctx)
        parent_event_id = baggage.get_baggage("lucidic.parent_event_id", context=ctx)
        
        if session_id or parent_event_id:
            debug(f"[ContextBridge] Extracted from OTel baggage - session: {truncate_id(session_id)}, parent: {truncate_id(parent_event_id)}")
        
        return session_id, parent_event_id
        
    except Exception as e:
        verbose(f"[ContextBridge] Failed to extract context: {e}")
        return None, None