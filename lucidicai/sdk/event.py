"""SDK event creation and management."""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from .context import current_parent_event_id
from ..core.config import get_config
from .event_builder import EventBuilder
from ..utils.logger import debug, truncate_id


def create_event(
    type: str = "generic",
    event_id: Optional[str] = None,
    **kwargs
) -> str:
    """Create a new event.
    
    Args:
        type: Event type (llm_generation, function_call, error_traceback, generic)
        event_id: Optional client event ID (will generate if not provided)
        **kwargs: Event-specific fields
        
    Returns:
        Event ID (client-generated or provided UUID)
    """
    # Import here to avoid circular dependency
    from ..sdk.init import get_session_id, get_event_queue
    
    # Get current session
    session_id = get_session_id()
    if not session_id:
        # No active session, return dummy ID
        debug("[Event] No active session, returning dummy event ID")
        return str(uuid.uuid4())
    
    # Get parent event ID from context
    parent_event_id = None
    try:
        parent_event_id = current_parent_event_id.get()
    except Exception:
        pass
    
    # Use provided event ID or generate new one
    client_event_id = event_id or str(uuid.uuid4())
    
    # Build parameters for EventBuilder
    params = {
        'type': type,
        'event_id': client_event_id,
        'parent_event_id': parent_event_id,
        'session_id': session_id,
        'occurred_at': kwargs.get('occurred_at') or datetime.now(timezone.utc).isoformat(),
        **kwargs  # Include all other kwargs
    }
    
    # Use EventBuilder to create normalized event request
    event_request = EventBuilder.build(params)
    
    debug(f"[Event] Creating {type} event {truncate_id(client_event_id)} (parent: {truncate_id(parent_event_id)}, session: {truncate_id(session_id)})")
    
    # Queue event for async sending
    event_queue = get_event_queue()
    if event_queue:
        event_queue.queue_event(event_request)
    
    return client_event_id



def create_error_event(
    error: Union[str, Exception],
    parent_event_id: Optional[str] = None,
    **kwargs
) -> str:
    """Create an error traceback event.
    
    This is a convenience function for creating error events with proper
    traceback information.
    
    Args:
        error: The error message or exception object
        parent_event_id: Optional parent event ID for nesting
        **kwargs: Additional event parameters
        
    Returns:
        Event ID of the created error event
    """
    import traceback
    
    if isinstance(error, Exception):
        error_str = str(error)
        traceback_str = traceback.format_exc()
    else:
        error_str = str(error)
        traceback_str = kwargs.pop('traceback', '')
    
    return create_event(
        type="error_traceback",
        error=error_str,
        traceback=traceback_str,
        parent_event_id=parent_event_id,
        **kwargs
    )


def flush(timeout_seconds: float = 2.0) -> bool:
    """Flush pending events.
    
    Args:
        timeout_seconds: Maximum time to wait for flush
        
    Returns:
        True if flush completed, False if timeout
    """
    from ..sdk.init import get_event_queue
    event_queue = get_event_queue()
    if event_queue:
        debug(f"[Event] Forcing flush with {timeout_seconds}s timeout")
        event_queue.force_flush(timeout_seconds)
        return True
    return False