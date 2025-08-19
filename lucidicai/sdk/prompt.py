"""Prompt management operations"""

from typing import Optional, Dict, Any
from lucidicai.sdk.state import SDKState
from lucidicai.client.resources import PromptResource
from lucidicai.util.logger import logger
from lucidicai.util.errors import PromptError


def get_prompt(
    prompt_name: str,
    variables: Optional[Dict[str, Any]] = None,
    cache_ttl: int = 300,
    label: str = "production"
) -> str:
    """Get a prompt from the backend with variable substitution
    
    Args:
        prompt_name: Name of the prompt
        variables: Variables to substitute in the prompt
        cache_ttl: Cache time-to-live in seconds (-1 for forever, 0 to disable)
        label: Prompt label/version
        
    Returns:
        Prompt with variables substituted
        
    Raises:
        PromptError: If prompt not found or variables missing
    """
    sdk_state = SDKState()
    if not sdk_state.is_initialized():
        logger.warning("SDK not initialized, cannot get prompt")
        return ""
    
    # Get prompt from API
    prompt_resource = PromptResource(sdk_state.http_client)
    prompt = prompt_resource.get_prompt(
        agent_id=sdk_state.agent_id,
        prompt_name=prompt_name,
        label=label,
        cache_ttl=cache_ttl
    )
    
    # Substitute variables if provided
    if variables:
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            if placeholder not in prompt:
                raise PromptError(f"Variable '{key}' not found in prompt")
            prompt = prompt.replace(placeholder, str(value))
    
    # Check for unreplaced variables
    if "{{" in prompt and "}}" in prompt:
        start = prompt.find("{{")
        end = prompt.find("}}", start)
        if start < end:
            logger.warning("Unreplaced variable(s) left in prompt. Please check your prompt.")
    
    return prompt