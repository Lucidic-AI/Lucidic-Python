"""SDK initialization module"""

import os
from typing import Optional, List, Dict, Any

from lucidicai.sdk.state import SDKState
from lucidicai.client.http_client import HttpClient
from lucidicai.client.resources import (
    SessionResource,
    StepResource,
    EventResource,
    UploadResource,
    PromptResource
)
from lucidicai.util.errors import APIKeyVerificationError
from lucidicai.util.logger import logger


def initialize_sdk(
    session_name: Optional[str] = None,
    session_id: Optional[str] = None,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    task: Optional[str] = None,
    providers: Optional[List[str]] = None,
    production_monitoring: bool = False,
    mass_sim_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    rubrics: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    masking_function: Optional[Any] = None,
    auto_end: bool = True,
) -> str:
    """Initialize the Lucidic SDK
    
    Args:
        session_name: Display name for the session
        session_id: Custom session ID (optional)
        api_key: API key for authentication
        agent_id: Agent ID
        task: Task description
        providers: List of provider names to enable
        production_monitoring: Enable production monitoring
        mass_sim_id: Mass simulation ID
        experiment_id: Experiment ID
        rubrics: Evaluation rubrics
        tags: Session tags
        masking_function: Function to mask sensitive data
        auto_end: Automatically end session on exit
        
    Returns:
        Session ID
        
    Raises:
        APIKeyVerificationError: If API key or agent ID is invalid
    """
    # Get or validate API credentials
    if api_key is None:
        api_key = os.getenv("LUCIDIC_API_KEY")
        if not api_key:
            raise APIKeyVerificationError(
                "API key not provided. Pass it to init() or set LUCIDIC_API_KEY environment variable."
            )
    
    if agent_id is None:
        agent_id = os.getenv("LUCIDIC_AGENT_ID")
        if not agent_id:
            raise APIKeyVerificationError(
                "Agent ID not provided. Pass it to init() or set LUCIDIC_AGENT_ID environment variable."
            )
    
    # Get SDK state
    state = SDKState()
    
    # Create HTTP client if needed
    if not state.http_client or state.api_key != api_key:
        state.http_client = HttpClient(api_key)
        state.api_key = api_key
        
        # Verify API key
        try:
            state.http_client.verify_api_key()
        except Exception as e:
            raise APIKeyVerificationError(f"Invalid API key: {e}")
    
    # Store configuration
    state.agent_id = agent_id
    state.masking_function = masking_function
    state.auto_end = auto_end
    
    # Initialize session
    session_resource = SessionResource(state.http_client)
    
    # Check for existing session with this ID
    if session_id and session_id in state.custom_session_id_translations:
        session_id = state.custom_session_id_translations[session_id]
    
    # Initialize or continue session
    response = session_resource.init_session(
        agent_id=agent_id,
        session_name=session_name,
        session_id=session_id,
        task=task,
        mass_sim_id=mass_sim_id,
        experiment_id=experiment_id,
        rubrics=rubrics,
        tags=tags,
        production_monitoring=production_monitoring
    )
    
    real_session_id = response["session_id"]
    
    # Track custom session ID mapping
    if session_id and session_id != real_session_id:
        state.custom_session_id_translations[session_id] = real_session_id
    
    # Store session ID
    state.session_id = real_session_id
    state.initialized = True
    
    # Set up providers if specified
    if providers:
        _setup_providers(providers)
    
    logger.info(f"Session initialized: {real_session_id}")
    return real_session_id


def _setup_providers(providers: List[str]):
    """Set up telemetry providers
    
    Args:
        providers: List of provider names to enable
    """
    from lucidicai.telemetry.otel_init import LucidicTelemetry
    from lucidicai.telemetry.otel_handlers import (
        OTelOpenAIHandler,
        OTelAnthropicHandler,
        OTelLangChainHandler,
        OTelPydanticAIHandler,
        OTelOpenAIAgentsHandler,
        OTelLiteLLMHandler
    )
    
    state = SDKState()
    
    # Initialize telemetry
    telemetry = LucidicTelemetry()
    if not telemetry.is_initialized():
        telemetry.initialize(agent_id=state.agent_id)
    
    # Map provider names to handlers
    handler_map = {
        "openai": OTelOpenAIHandler,
        "anthropic": OTelAnthropicHandler,
        "langchain": OTelLangChainHandler,
        "pydantic_ai": OTelPydanticAIHandler,
        "openai_agents": OTelOpenAIAgentsHandler,
        "litellm": OTelLiteLLMHandler,
    }
    
    # Set up each provider
    for provider_name in providers:
        if provider_name in handler_map:
            # Check if already set up
            already_setup = any(
                type(p).__name__ == handler_map[provider_name].__name__ 
                for p in state.providers
            )
            if not already_setup:
                handler = handler_map[provider_name]()
                handler.override()
                state.providers.append(handler)
                logger.info(f"Provider {provider_name} initialized")


def get_sdk_state() -> SDKState:
    """Get the current SDK state
    
    Returns:
        Current SDKState instance
    """
    return SDKState()