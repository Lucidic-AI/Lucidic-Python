"""Session management operations"""

from typing import Optional

from lucidicai.sdk.state import SDKState
from lucidicai.sdk.context import current_session_id
from lucidicai.client.resources import SessionResource
from lucidicai.util.logger import logger


def update_session(
    task: Optional[str] = None,
    session_eval: Optional[float] = None,
    session_eval_reason: Optional[str] = None,
    is_successful: Optional[bool] = None,
    is_successful_reason: Optional[str] = None,
    tags: Optional[list] = None
) -> None:
    """Update the current session
    
    Args:
        task: Task description
        session_eval: Session evaluation score
        session_eval_reason: Evaluation reason
        is_successful: Whether session was successful
        is_successful_reason: Reason for success/failure
        tags: Tags to add to session
    """
    state = SDKState()
    if not state.is_initialized():
        logger.warning("SDK not initialized, cannot update session")
        return
    
    # Get session ID from context or state
    session_id = current_session_id.get()
    if not session_id:
        session_id = state.session_id
    
    if not session_id:
        logger.warning("No active session to update")
        return
    
    # Apply masking
    if is_successful_reason:
        is_successful_reason = state.mask(is_successful_reason)
    if session_eval_reason:
        session_eval_reason = state.mask(session_eval_reason)
    
    # Update via API
    session_resource = SessionResource(state.http_client)
    session_resource.update_session(
        session_id=session_id,
        task=task,
        session_eval=session_eval,
        session_eval_reason=session_eval_reason,
        is_successful=is_successful,
        is_successful_reason=is_successful_reason,
        tags=tags
    )


def end_session(
    session_eval: Optional[float] = None,
    session_eval_reason: Optional[str] = None,
    is_successful: Optional[bool] = None,
    is_successful_reason: Optional[str] = None
) -> None:
    """End the current session
    
    Args:
        session_eval: Session evaluation score
        session_eval_reason: Evaluation reason
        is_successful: Whether session was successful
        is_successful_reason: Reason for success/failure
    """
    state = SDKState()
    if not state.is_initialized():
        logger.warning("SDK not initialized, cannot end session")
        return
    
    # Get session ID from context or state
    session_id = current_session_id.get()
    if not session_id:
        session_id = state.session_id
    
    if not session_id:
        logger.warning("No active session to end")
        return
    
    # Wait for any pending telemetry
    for provider in state.providers:
        if hasattr(provider, '_callback') and hasattr(provider._callback, 'wait_for_pending_callbacks'):
            logger.info("Waiting for pending callbacks before ending session...")
            provider._callback.wait_for_pending_callbacks(timeout=5.0)
    
    # Apply masking
    if is_successful_reason:
        is_successful_reason = state.mask(is_successful_reason)
    if session_eval_reason:
        session_eval_reason = state.mask(session_eval_reason)
    
    # End via API
    session_resource = SessionResource(state.http_client)
    session_resource.update_session(
        session_id=session_id,
        is_finished=True,
        session_eval=session_eval,
        session_eval_reason=session_eval_reason,
        is_successful=is_successful,
        is_successful_reason=is_successful_reason
    )
    
    # Clear session from state if it matches
    if session_id == state.session_id:
        state.session_id = None
        logger.info(f"Session {session_id} ended")