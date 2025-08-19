"""Step management operations"""

from typing import Optional, Dict, Any
from lucidicai.sdk.state import SDKState
from lucidicai.sdk.context import current_session_id
from lucidicai.client.resources import StepResource, UploadResource
from lucidicai.util.logger import logger
from lucidicai.util.errors import InvalidOperationError


# Track active steps per session
_active_steps: Dict[str, str] = {}


def create_step(
    state: Optional[str] = None,
    action: Optional[str] = None,
    goal: Optional[str] = None,
    eval_score: Optional[float] = None,
    eval_description: Optional[str] = None,
    screenshot: Optional[str] = None,
    screenshot_path: Optional[str] = None
) -> Optional[str]:
    """Create a new step in the current session
    
    Args:
        state: State description
        action: Action description
        goal: Goal description
        eval_score: Evaluation score
        eval_description: Evaluation description
        screenshot: Base64 encoded screenshot
        screenshot_path: Path to screenshot file
        
    Returns:
        Step ID if created, None otherwise
    """
    sdk_state = SDKState()
    if not sdk_state.is_initialized():
        logger.warning("SDK not initialized, cannot create step")
        return None
    
    # Get session ID
    session_id = current_session_id.get() or sdk_state.session_id
    if not session_id:
        logger.warning("No active session, cannot create step")
        return None
    
    # Check for active step in this session
    if session_id in _active_steps:
        logger.warning(f"Step {_active_steps[session_id]} still active, ending it first")
        end_step(step_id=_active_steps[session_id])
    
    # Create step
    step_resource = StepResource(sdk_state.http_client)
    response = step_resource.init_step(session_id)
    step_id = response.get("step_id")
    
    if not step_id:
        logger.error("Failed to create step")
        return None
    
    # Track active step
    _active_steps[session_id] = step_id
    
    # Update with initial data if provided
    update_params = {}
    if state:
        update_params["state"] = sdk_state.mask(state)
    if action:
        update_params["action"] = sdk_state.mask(action)
    if goal:
        update_params["goal"] = sdk_state.mask(goal)
    if eval_score is not None:
        update_params["eval_score"] = eval_score
    if eval_description:
        update_params["eval_description"] = sdk_state.mask(eval_description)
    
    # Handle screenshot upload if provided
    if screenshot or screenshot_path:
        upload_resource = UploadResource(sdk_state.http_client)
        img_data = upload_resource.process_screenshot(screenshot, screenshot_path)
        if img_data:
            presigned_url = upload_resource.get_presigned_url(
                agent_id=sdk_state.agent_id,
                step_id=step_id
            )
            if upload_resource.upload_image_to_s3(presigned_url, img_data):
                update_params["has_screenshot"] = True
    
    if update_params:
        step_resource.update_step(step_id, **update_params)
    
    logger.info(f"Created step: {step_id}")
    return step_id


def update_step(
    step_id: Optional[str] = None,
    state: Optional[str] = None,
    action: Optional[str] = None,
    goal: Optional[str] = None,
    eval_score: Optional[float] = None,
    eval_description: Optional[str] = None,
    screenshot: Optional[str] = None,
    screenshot_path: Optional[str] = None
) -> None:
    """Update an existing step
    
    Args:
        step_id: Step ID to update (uses active step if not provided)
        state: State description
        action: Action description
        goal: Goal description
        eval_score: Evaluation score
        eval_description: Evaluation description
        screenshot: Base64 encoded screenshot
        screenshot_path: Path to screenshot file
    """
    sdk_state = SDKState()
    if not sdk_state.is_initialized():
        logger.warning("SDK not initialized, cannot update step")
        return
    
    # Get step ID
    if not step_id:
        session_id = current_session_id.get() or sdk_state.session_id
        if session_id and session_id in _active_steps:
            step_id = _active_steps[session_id]
    
    if not step_id:
        raise InvalidOperationError("No step ID provided and no active step found")
    
    # Build update params
    update_params = {}
    if state:
        update_params["state"] = sdk_state.mask(state)
    if action:
        update_params["action"] = sdk_state.mask(action)
    if goal:
        update_params["goal"] = sdk_state.mask(goal)
    if eval_score is not None:
        update_params["eval_score"] = eval_score
    if eval_description:
        update_params["eval_description"] = sdk_state.mask(eval_description)
    
    # Handle screenshot upload
    if screenshot or screenshot_path:
        upload_resource = UploadResource(sdk_state.http_client)
        img_data = upload_resource.process_screenshot(screenshot, screenshot_path)
        if img_data:
            presigned_url = upload_resource.get_presigned_url(
                agent_id=sdk_state.agent_id,
                step_id=step_id
            )
            if upload_resource.upload_image_to_s3(presigned_url, img_data):
                update_params["has_screenshot"] = True
    
    # Update step
    step_resource = StepResource(sdk_state.http_client)
    step_resource.update_step(step_id, **update_params)


def end_step(
    step_id: Optional[str] = None,
    state: Optional[str] = None,
    action: Optional[str] = None,
    goal: Optional[str] = None,
    eval_score: Optional[float] = None,
    eval_description: Optional[str] = None,
    screenshot: Optional[str] = None,
    screenshot_path: Optional[str] = None
) -> None:
    """End a step
    
    Args:
        step_id: Step ID to end (uses active step if not provided)
        state: Final state description
        action: Final action description
        goal: Final goal description
        eval_score: Final evaluation score
        eval_description: Final evaluation description
        screenshot: Final screenshot (base64)
        screenshot_path: Path to final screenshot
    """
    sdk_state = SDKState()
    if not sdk_state.is_initialized():
        logger.warning("SDK not initialized, cannot end step")
        return
    
    # Get step ID
    if not step_id:
        session_id = current_session_id.get() or sdk_state.session_id
        if session_id and session_id in _active_steps:
            step_id = _active_steps[session_id]
    
    if not step_id:
        raise InvalidOperationError("No step ID provided and no active step found")
    
    # Update with final data and mark as finished
    update_params = {"is_finished": True}
    if state:
        update_params["state"] = sdk_state.mask(state)
    if action:
        update_params["action"] = sdk_state.mask(action)
    if goal:
        update_params["goal"] = sdk_state.mask(goal)
    if eval_score is not None:
        update_params["eval_score"] = eval_score
    if eval_description:
        update_params["eval_description"] = sdk_state.mask(eval_description)
    
    # Handle screenshot
    if screenshot or screenshot_path:
        upload_resource = UploadResource(sdk_state.http_client)
        img_data = upload_resource.process_screenshot(screenshot, screenshot_path)
        if img_data:
            presigned_url = upload_resource.get_presigned_url(
                agent_id=sdk_state.agent_id,
                step_id=step_id
            )
            if upload_resource.upload_image_to_s3(presigned_url, img_data):
                update_params["has_screenshot"] = True
    
    # Update step
    step_resource = StepResource(sdk_state.http_client)
    step_resource.update_step(step_id, **update_params)
    
    # Clear from active steps
    for session_id, active_step_id in list(_active_steps.items()):
        if active_step_id == step_id:
            del _active_steps[session_id]
            break
    
    logger.info(f"Ended step: {step_id}")