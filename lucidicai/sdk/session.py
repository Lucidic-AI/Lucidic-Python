"""SDK session creation and management."""
import asyncio
import threading
import uuid
from typing import List, Optional, Set
from weakref import WeakSet

from ..core.config import SDKConfig, set_config
from ..utils.logger import debug, info, warning, error, truncate_id
from .context import set_active_session, clear_active_session
from .shutdown_manager import get_shutdown_manager, SessionState


# Track background threads for flush()
_background_threads: Set[threading.Thread] = WeakSet()


def _prepare_session_config(
    api_key: Optional[str],
    agent_id: Optional[str],
    providers: Optional[List[str]],
    production_monitoring: bool,
    auto_end: bool,
    capture_uncaught: bool,
) -> SDKConfig:
    """Prepare and validate SDK configuration.
    
    Returns:
        Validated SDKConfig instance
    """
    config = SDKConfig.from_env(
        api_key=api_key,
        agent_id=agent_id,
        auto_end=auto_end,
        production_monitoring=production_monitoring
    )
    
    if providers:
        config.telemetry.providers = providers
    
    config.error_handling.capture_uncaught = capture_uncaught
    
    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {', '.join(errors)}")
    
    return config


def _ensure_http_and_resources_initialized(config: SDKConfig) -> None:
    """Ensure HTTP client and resources are initialized."""
    from .init import get_http, get_resources, set_http, set_resources
    from ..api.client import HttpClient
    from ..api.resources.event import EventResource
    from ..api.resources.session import SessionResource
    from ..api.resources.dataset import DatasetResource
    
    # Initialize HTTP client
    if not get_http():
        debug("[SDK] Initializing HTTP client")
        set_http(HttpClient(config))
    
    # Initialize resources
    resources = get_resources()
    if not resources:
        http = get_http()
        set_resources({
            'events': EventResource(http),
            'sessions': SessionResource(http),
            'datasets': DatasetResource(http)
        })


def _build_session_params(
    session_id: Optional[str],
    session_name: Optional[str],
    agent_id: str,
    task: Optional[str],
    tags: Optional[List],
    experiment_id: Optional[str],
    datasetitem_id: Optional[str],
    evaluators: Optional[List],
    production_monitoring: bool,
) -> tuple[str, dict]:
    """Build session parameters for API call.
    
    Returns:
        Tuple of (real_session_id, session_params)
    """
    # Create or retrieve session
    if session_id:
        real_session_id = session_id
    else:
        real_session_id = str(uuid.uuid4())
    
    # Create session via API - only send non-None values
    session_params = {
        'session_id': real_session_id,
        'session_name': session_name or 'Unnamed Session',
        'agent_id': agent_id,
    }
    
    # Only add optional fields if they have values
    if task:
        session_params['task'] = task
    if tags:
        session_params['tags'] = tags
    if experiment_id:
        session_params['experiment_id'] = experiment_id
    if datasetitem_id:
        session_params['datasetitem_id'] = datasetitem_id
    if evaluators:
        session_params['evaluators'] = evaluators
    if production_monitoring:
        session_params['production_monitoring'] = production_monitoring
    
    return real_session_id, session_params


def _finalize_session(
    real_session_id: str,
    session_name: Optional[str],
    auto_end: bool,
    providers: Optional[List[str]],
) -> str:
    """Finalize session setup after API call."""
    from .init import _sdk_state, _initialize_telemetry, get_resources
    
    _sdk_state.session_id = real_session_id
    
    info(f"[SDK] Session created: {truncate_id(real_session_id)} (name: {session_name or 'Unnamed Session'})")
    
    # Set active session in context
    set_active_session(real_session_id)
    
    # Register session with shutdown manager
    debug(f"[SDK] Registering session with shutdown manager (auto_end={auto_end})")
    shutdown_manager = get_shutdown_manager()
    session_state = SessionState(
        session_id=real_session_id,
        http_client=get_resources(),
        auto_end=auto_end
    )
    shutdown_manager.register_session(real_session_id, session_state)
    
    # Initialize telemetry if providers specified
    if providers:
        debug(f"[SDK] Initializing telemetry for providers: {providers}")
        _initialize_telemetry(providers)
    
    return real_session_id


def create_session(
    session_name: Optional[str] = None,
    session_id: Optional[str] = None,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    task: Optional[str] = None,
    providers: Optional[List[str]] = None,
    production_monitoring: bool = False,
    experiment_id: Optional[str] = None,
    evaluators: Optional[List] = None,
    tags: Optional[List] = None,
    datasetitem_id: Optional[str] = None,
    masking_function: Optional[callable] = None,
    auto_end: bool = True,
    capture_uncaught: bool = True,
) -> str:
    """Create a new Lucidic session (synchronous).
    
    Args:
        session_name: Name for the session
        session_id: Custom session ID (optional)
        api_key: API key (uses env if not provided)
        agent_id: Agent ID (uses env if not provided)
        task: Task description
        providers: List of telemetry providers to instrument
        production_monitoring: Enable production monitoring
        experiment_id: Experiment ID to associate with session
        evaluators: Evaluators to use
        tags: Session tags
        datasetitem_id: Dataset item ID
        masking_function: Function to mask sensitive data
        auto_end: Automatically end session on exit
        capture_uncaught: Capture uncaught exceptions
        
    Returns:
        Session ID
        
    Raises:
        APIKeyVerificationError: If API credentials are invalid
    """
    from .init import get_resources
    
    # Prepare configuration
    config = _prepare_session_config(
        api_key, agent_id, providers, production_monitoring, auto_end, capture_uncaught
    )
    set_config(config)
    
    # Ensure HTTP client and resources are initialized
    _ensure_http_and_resources_initialized(config)
    
    # Build session parameters
    real_session_id, session_params = _build_session_params(
        session_id, session_name, config.agent_id, task, tags,
        experiment_id, datasetitem_id, evaluators, production_monitoring
    )
    
    # Create session via API (synchronous)
    debug(f"[SDK] Creating session with params: {session_params}")
    session_resource = get_resources()['sessions']
    session_data = session_resource.create_session(session_params)
    
    # Use the session_id returned by the backend
    real_session_id = session_data.get('session_id', real_session_id)
    
    return _finalize_session(real_session_id, session_name, auto_end, providers)


async def acreate_session(
    session_name: Optional[str] = None,
    session_id: Optional[str] = None,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    task: Optional[str] = None,
    providers: Optional[List[str]] = None,
    production_monitoring: bool = False,
    experiment_id: Optional[str] = None,
    evaluators: Optional[List] = None,
    tags: Optional[List] = None,
    datasetitem_id: Optional[str] = None,
    masking_function: Optional[callable] = None,
    auto_end: bool = True,
    capture_uncaught: bool = True,
) -> str:
    """Create a new Lucidic session (asynchronous).
    
    Args:
        session_name: Name for the session
        session_id: Custom session ID (optional)
        api_key: API key (uses env if not provided)
        agent_id: Agent ID (uses env if not provided)
        task: Task description
        providers: List of telemetry providers to instrument
        production_monitoring: Enable production monitoring
        experiment_id: Experiment ID to associate with session
        evaluators: Evaluators to use
        tags: Session tags
        datasetitem_id: Dataset item ID
        masking_function: Function to mask sensitive data
        auto_end: Automatically end session on exit
        capture_uncaught: Capture uncaught exceptions
        
    Returns:
        Session ID
        
    Raises:
        APIKeyVerificationError: If API credentials are invalid
    """
    from .init import get_resources
    
    # Prepare configuration
    config = _prepare_session_config(
        api_key, agent_id, providers, production_monitoring, auto_end, capture_uncaught
    )
    set_config(config)
    
    # Ensure HTTP client and resources are initialized
    _ensure_http_and_resources_initialized(config)
    
    # Build session parameters
    real_session_id, session_params = _build_session_params(
        session_id, session_name, config.agent_id, task, tags,
        experiment_id, datasetitem_id, evaluators, production_monitoring
    )
    
    # Create session via API (asynchronous)
    debug(f"[SDK] Creating session with params: {session_params}")
    session_resource = get_resources()['sessions']
    session_data = await session_resource.acreate_session(session_params)
    
    # Use the session_id returned by the backend
    real_session_id = session_data.get('session_id', real_session_id)
    
    return _finalize_session(real_session_id, session_name, auto_end, providers)


def emit_session(
    session_name: Optional[str] = None,
    session_id: Optional[str] = None,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    task: Optional[str] = None,
    providers: Optional[List[str]] = None,
    production_monitoring: bool = False,
    experiment_id: Optional[str] = None,
    evaluators: Optional[List] = None,
    tags: Optional[List] = None,
    datasetitem_id: Optional[str] = None,
    masking_function: Optional[callable] = None,
    auto_end: bool = True,
    capture_uncaught: bool = True,
) -> str:
    """Fire-and-forget session creation that returns instantly.
    
    This function returns immediately with a session ID, while the actual
    session creation happens in a background thread. Perfect for reducing
    initialization latency.
    
    Args:
        session_name: Name for the session
        session_id: Custom session ID (optional)
        api_key: API key (uses env if not provided)
        agent_id: Agent ID (uses env if not provided)
        task: Task description
        providers: List of telemetry providers to instrument
        production_monitoring: Enable production monitoring
        experiment_id: Experiment ID to associate with session
        evaluators: Evaluators to use
        tags: Session tags
        datasetitem_id: Dataset item ID
        masking_function: Function to mask sensitive data
        auto_end: Automatically end session on exit
        capture_uncaught: Capture uncaught exceptions
        
    Returns:
        Session ID - returned immediately
    """
    from .init import _sdk_state
    
    # Pre-generate session ID for instant return
    real_session_id = session_id or str(uuid.uuid4())
    
    # Immediately set session state for subsequent operations
    _sdk_state.session_id = real_session_id
    set_active_session(real_session_id)
    
    # Run async session creation in background thread
    def _run():
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    acreate_session(
                        session_name=session_name,
                        session_id=real_session_id,  # Use pre-generated ID
                        api_key=api_key,
                        agent_id=agent_id,
                        task=task,
                        providers=providers,
                        production_monitoring=production_monitoring,
                        experiment_id=experiment_id,
                        evaluators=evaluators,
                        tags=tags,
                        datasetitem_id=datasetitem_id,
                        masking_function=masking_function,
                        auto_end=auto_end,
                        capture_uncaught=capture_uncaught,
                    )
                )
            finally:
                loop.close()
        except Exception as e:
            error(f"[Session] Background emit failed for {truncate_id(real_session_id)}: {e}")
    
    thread = threading.Thread(
        target=_run, 
        daemon=True, 
        name=f"emit-session-{truncate_id(real_session_id)}"
    )
    _background_threads.add(thread)
    thread.start()
    
    info(f"[Session] Emitted session {truncate_id(real_session_id)} (name: {session_name or 'Unnamed Session'}, fire-and-forget)")
    return real_session_id


def flush_sessions(timeout: float = 5.0) -> None:
    """Wait for all background session creations to complete.
    
    Args:
        timeout: Maximum time to wait in seconds (default: 5.0)
    """
    import time
    
    start_time = time.time()
    
    # Wait for background threads
    threads = list(_background_threads)
    for thread in threads:
        if thread.is_alive():
            remaining = timeout - (time.time() - start_time)
            if remaining > 0:
                thread.join(timeout=remaining)
                if thread.is_alive():
                    warning(f"[Session] Thread {thread.name} did not complete within timeout")
    
    debug(f"[Session] Flush completed in {time.time() - start_time:.2f}s")


def end_session(
    session_id: Optional[str] = None,
    is_successful: Optional[bool] = None,
    is_successful_reason: Optional[str] = None,
    session_eval: Optional[float] = None,
    session_eval_reason: Optional[str] = None,
) -> None:
    """End the current or specified session.
    
    Args:
        session_id: Session ID to end (uses current if not provided)
        is_successful: Whether session was successful
        is_successful_reason: Reason for success or failure
        session_eval: Session evaluation score
        session_eval_reason: Reason for evaluation
    """
    from .init import get_session_id, get_resources
    
    # Get session ID
    if not session_id:
        session_id = get_session_id()
    
    if not session_id:
        warning("[Session] No active session to end")
        return
    
    # End session via API
    resources = get_resources()
    if resources and 'sessions' in resources:
        try:
            resources['sessions'].end_session(
                session_id=session_id,
                is_successful=is_successful,
                is_successful_reason=is_successful_reason,
                session_eval=session_eval,
                session_eval_reason=session_eval_reason
            )
            info(f"[Session] Ended session {truncate_id(session_id)}")
        except Exception as e:
            error(f"[Session] Failed to end session {truncate_id(session_id)}: {e}")
        
        # Unregister session from shutdown manager
        shutdown_manager = get_shutdown_manager()
        shutdown_manager.unregister_session(session_id)
        
        # Clear active session
        clear_active_session()



async def aend_session(
    session_id: Optional[str] = None,
    is_successful: Optional[bool] = None,
    is_successful_reason: Optional[str] = None,
    session_eval: Optional[float] = None,
    session_eval_reason: Optional[str] = None,
) -> None:
    """End the current or specified session (asynchronous).
    
    Args:
        session_id: Session ID to end (uses current if not provided)
        is_successful: Whether session was successful
        is_successful_reason: Reason for success or failure
        session_eval: Session evaluation score
        session_eval_reason: Reason for evaluation
    """
    from .init import get_session_id, get_resources
    
    # Get session ID
    if not session_id:
        session_id = get_session_id()
    
    if not session_id:
        warning("[Session] No active session to end")
        return
    
    # End session via API
    resources = get_resources()
    if resources and 'sessions' in resources:
        try:
            await resources['sessions'].aend_session(
                session_id=session_id,
                is_successful=is_successful,
                is_successful_reason=is_successful_reason,
                session_eval=session_eval,
                session_eval_reason=session_eval_reason
            )
            info(f"[Session] Ended session {truncate_id(session_id)}")
        except Exception as e:
            error(f"[Session] Failed to end session {truncate_id(session_id)}: {e}")
        
        # Unregister session from shutdown manager
        shutdown_manager = get_shutdown_manager()
        shutdown_manager.unregister_session(session_id)
        
        # Clear active session
        clear_active_session()