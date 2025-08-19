"""Lucidic AI SDK for Python

A comprehensive observability SDK for AI agents providing session, step, and event tracking.
"""

import atexit
import os
import signal
from typing import List, Optional, Dict, Any

# Import SDK functionality
from lucidicai.sdk import (
    initialize_sdk,
    get_sdk_state,
    update_session,
    end_session,
    create_step,
    update_step,
    end_step,
    create_event,
    update_event,
    end_event,
    get_prompt,
    step,
    event,
    session,
    session_async,
    bind_session,
    bind_session_async,
    set_active_session,
    clear_active_session,
)

# Import errors
from lucidicai.util.errors import (
    APIKeyVerificationError,
    LucidicNotInitializedError,
    PromptError,
    InvalidOperationError,
)

# Import logger for module-level use
from lucidicai.util.logger import logger


# Public API - Only expose essential functions
__all__ = [
    # Core initialization
    'init',
    
    # Session management
    'update_session',
    'end_session',
    
    # Step management
    'create_step',
    'update_step',
    'end_step',
    
    # Event management
    'create_event',
    'update_event',
    'end_event',
    
    # Prompt management
    'get_prompt',
    
    # Mass simulation
    'create_mass_sim',
    
    # Decorators
    'step',
    'event',
    
    # Context managers
    'session',
    'session_async',
    'bind_session',
    'bind_session_async',
    
    # Essential errors
    'APIKeyVerificationError',
    'LucidicNotInitializedError',
    'PromptError',
    'InvalidOperationError',
]


def init(
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
    auto_end: Optional[bool] = None,
) -> str:
    """Initialize the Lucidic SDK and start a new session
    
    Args:
        session_name: Display name for the session
        session_id: Custom session ID (optional)
        api_key: API key for authentication (uses LUCIDIC_API_KEY env var if not provided)
        agent_id: Agent ID (uses LUCIDIC_AGENT_ID env var if not provided)
        task: Task description
        providers: List of provider names to enable ("openai", "anthropic", "langchain", etc.)
        production_monitoring: Enable production monitoring
        mass_sim_id: Mass simulation ID if part of a simulation
        experiment_id: Experiment ID if part of an experiment
        rubrics: Evaluation rubrics
        tags: Session tags
        masking_function: Function to mask sensitive data
        auto_end: Automatically end session on exit (default: True)
        
    Returns:
        Session ID
        
    Raises:
        APIKeyVerificationError: If API key or agent ID is invalid or missing
        
    Example:
        >>> import lucidicai as lai
        >>> session_id = lai.init(
        ...     session_name="My Agent Session",
        ...     providers=["openai", "anthropic"]
        ... )
    """
    # Handle auto_end default with environment variable support
    if auto_end is None:
        auto_end = os.getenv("LUCIDIC_AUTO_END", "true").lower() == "true"
    
    # Initialize SDK
    session_id = initialize_sdk(
        session_name=session_name,
        session_id=session_id,
        api_key=api_key,
        agent_id=agent_id,
        task=task,
        providers=providers or [],
        production_monitoring=production_monitoring,
        mass_sim_id=mass_sim_id,
        experiment_id=experiment_id,
        rubrics=rubrics,
        tags=tags,
        masking_function=masking_function,
        auto_end=auto_end,
    )
    
    # Set active session in context
    try:
        set_active_session(session_id)
    except Exception as e:
        logger.debug(f"Could not set active session in context: {e}")
    
    logger.info("Session initialized successfully")
    return session_id


def create_mass_sim(
    mass_sim_name: str,
    total_num_sessions: int,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    task: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> str:
    """Create a new mass simulation
    
    Args:
        mass_sim_name: Name of the mass simulation
        total_num_sessions: Total intended number of sessions
        api_key: API key (uses LUCIDIC_API_KEY env var if not provided)
        agent_id: Agent ID (uses LUCIDIC_AGENT_ID env var if not provided)
        task: Task description
        tags: Tags for the mass simulation
        
    Returns:
        Mass simulation ID to pass to init()
        
    Raises:
        APIKeyVerificationError: If API key or agent ID is invalid
    """
    from lucidicai.client.http_client import HttpClient
    from lucidicai.util.errors import APIKeyVerificationError
    
    # Get credentials
    if not api_key:
        api_key = os.getenv("LUCIDIC_API_KEY")
        if not api_key:
            raise APIKeyVerificationError(
                "API key not provided. Pass it to create_mass_sim() or set LUCIDIC_API_KEY environment variable."
            )
    
    if not agent_id:
        agent_id = os.getenv("LUCIDIC_AGENT_ID")
        if not agent_id:
            raise APIKeyVerificationError(
                "Agent ID not provided. Pass it to create_mass_sim() or set LUCIDIC_AGENT_ID environment variable."
            )
    
    # Create mass sim via API
    http_client = HttpClient(api_key)
    response = http_client.post("initmasssim", {
        "agent_id": agent_id,
        "mass_sim_name": mass_sim_name,
        "total_num_sims": total_num_sessions,
        "task": task,
        "tags": tags,
    })
    
    mass_sim_id = response.get("mass_sim_id")
    logger.info(f"Created mass simulation with ID: {mass_sim_id}")
    return mass_sim_id


# Cleanup functions
def _cleanup_telemetry():
    """Cleanup telemetry on exit"""
    try:
        from lucidicai.telemetry.otel_init import LucidicTelemetry
        telemetry = LucidicTelemetry()
        if telemetry.is_initialized():
            telemetry.uninstrument_all()
            logger.info("OpenTelemetry instrumentation cleaned up")
    except Exception as e:
        logger.debug(f"Error during telemetry cleanup: {e}")


def _auto_end_session():
    """Automatically end session on exit if auto_end is enabled"""
    try:
        state = get_sdk_state()
        if state.auto_end and state.session_id:
            logger.info("Auto-ending active session on exit")
            state.auto_end = False  # Prevent double-ending
            end_session()
    except Exception as e:
        logger.debug(f"Error during auto-end session: {e}")


def _signal_handler(signum, frame):
    """Handle interruption signals gracefully"""
    _auto_end_session()
    _cleanup_telemetry()
    # Re-raise signal for default handling
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Register cleanup functions (LIFO order - auto-end runs first)
atexit.register(_cleanup_telemetry)
atexit.register(_auto_end_session)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)