"""Event management operations"""

from typing import Optional, List, Dict, Any
from lucidicai.sdk.state import SDKState
from lucidicai.sdk.context import current_session_id
from lucidicai.sdk.step import _active_steps
from lucidicai.client.resources import EventResource, UploadResource
from lucidicai.util.logger import logger
from lucidicai.util.errors import InvalidOperationError


# Track active events
_active_events: Dict[str, str] = {}


def create_event(
    step_id: Optional[str] = None,
    description: Optional[str] = None,
    result: Optional[str] = None,
    cost_added: Optional[float] = None,
    model: Optional[str] = None,
    screenshots: Optional[List[str]] = None,
    function_name: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Create a new event
    
    Args:
        step_id: Optional step ID to link event to
        description: Event description
        result: Event result
        cost_added: Cost added by this event
        model: Model used
        screenshots: List of base64 encoded screenshots
        function_name: Function that triggered event
        arguments: Function arguments
        
    Returns:
        Event ID if created, None otherwise
    """
    sdk_state = SDKState()
    if not sdk_state.is_initialized():
        logger.warning("SDK not initialized, cannot create event")
        return None
    
    # Get session ID
    session_id = current_session_id.get() or sdk_state.session_id
    if not session_id:
        logger.warning("No active session, cannot create event")
        return None
    
    # Use active step if not provided
    if not step_id and session_id in _active_steps:
        step_id = _active_steps[session_id]
    
    # Apply masking
    if description:
        description = sdk_state.mask(description)
    if result:
        result = sdk_state.mask(result)
    
    # Create event
    event_resource = EventResource(sdk_state.http_client)
    response = event_resource.init_event(
        session_id=session_id,
        step_id=step_id,
        description=description,
        result=result,
        cost_added=cost_added,
        model=model,
        nscreenshots=len(screenshots) if screenshots else None,
        function_name=function_name,
        arguments=arguments
    )
    
    event_id = response.get("event_id")
    if not event_id:
        logger.error("Failed to create event")
        return None
    
    # Track active event
    _active_events[session_id] = event_id
    
    # Upload screenshots if provided
    if screenshots:
        upload_resource = UploadResource(sdk_state.http_client)
        uploaded = upload_resource.upload_screenshots(
            agent_id=sdk_state.agent_id,
            screenshots=screenshots,
            event_id=event_id,
            session_id=session_id
        )
        logger.info(f"Uploaded {uploaded}/{len(screenshots)} screenshots for event {event_id}")
    
    logger.info(f"Created event: {event_id}")
    return event_id


def update_event(
    event_id: Optional[str] = None,
    description: Optional[str] = None,
    result: Optional[str] = None,
    cost_added: Optional[float] = None,
    model: Optional[str] = None,
    screenshots: Optional[List[str]] = None,
    function_name: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> None:
    """Update an existing event
    
    Args:
        event_id: Event ID to update (uses active event if not provided)
        description: Event description
        result: Event result
        cost_added: Cost added by this event
        model: Model used
        screenshots: Additional screenshots (not supported for updates)
        function_name: Function name
        arguments: Function arguments
    """
    sdk_state = SDKState()
    if not sdk_state.is_initialized():
        logger.warning("SDK not initialized, cannot update event")
        return
    
    # Get event ID
    if not event_id:
        session_id = current_session_id.get() or sdk_state.session_id
        if session_id and session_id in _active_events:
            event_id = _active_events[session_id]
    
    if not event_id:
        raise InvalidOperationError("No event ID provided and no active event found")
    
    # Apply masking
    if description:
        description = sdk_state.mask(description)
    if result:
        result = sdk_state.mask(result)
    
    # Update event
    event_resource = EventResource(sdk_state.http_client)
    event_resource.update_event(
        event_id=event_id,
        description=description,
        result=result,
        cost_added=cost_added,
        model=model,
        function_name=function_name,
        arguments=arguments
    )
    
    if screenshots:
        logger.warning("Screenshot updates not supported, ignoring screenshots parameter")


def end_event(
    event_id: Optional[str] = None,
    description: Optional[str] = None,
    result: Optional[str] = None,
    cost_added: Optional[float] = None,
    model: Optional[str] = None,
    screenshots: Optional[List[str]] = None,
    function_name: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> None:
    """End an event
    
    Args:
        event_id: Event ID to end (uses active event if not provided)
        description: Final description
        result: Final result
        cost_added: Final cost
        model: Model used
        screenshots: Final screenshots (not supported for updates)
        function_name: Function name
        arguments: Function arguments
    """
    sdk_state = SDKState()
    if not sdk_state.is_initialized():
        logger.warning("SDK not initialized, cannot end event")
        return
    
    # Get event ID
    if not event_id:
        session_id = current_session_id.get() or sdk_state.session_id
        if session_id and session_id in _active_events:
            event_id = _active_events[session_id]
    
    if not event_id:
        raise InvalidOperationError("No event ID provided and no active event found")
    
    # Apply masking
    if description:
        description = sdk_state.mask(description)
    if result:
        result = sdk_state.mask(result)
    
    # End event
    event_resource = EventResource(sdk_state.http_client)
    event_resource.update_event(
        event_id=event_id,
        description=description,
        result=result or "Response received",
        cost_added=cost_added,
        model=model,
        is_finished=True,
        function_name=function_name,
        arguments=arguments
    )
    
    # Clear from active events
    for session_id, active_event_id in list(_active_events.items()):
        if active_event_id == event_id:
            del _active_events[session_id]
            break
    
    if screenshots:
        logger.warning("Screenshot updates not supported, ignoring screenshots parameter")
    
    logger.info(f"Ended event: {event_id}")