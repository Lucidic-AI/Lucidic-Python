"""SDK initialization module.

This module handles SDK initialization, separating concerns from the main __init__.py
"""
import uuid
from typing import List, Optional
import asyncio
import threading
from weakref import WeakKeyDictionary

from ..api.client import HttpClient
from ..api.resources.event import EventResource
from ..api.resources.session import SessionResource
from ..api.resources.dataset import DatasetResource
from ..core.config import SDKConfig, get_config, set_config
from ..utils.logger import debug, info, warning, error, truncate_id
from .context import set_active_session, current_session_id
from .shutdown_manager import get_shutdown_manager, SessionState
from ..telemetry.telemetry_init import instrument_providers
from opentelemetry.sdk.trace import TracerProvider


class SDKState:
    """Container for SDK runtime state."""

    def __init__(self):
        self.http: Optional[HttpClient] = None
        self.session_id: Optional[str] = None
        self.tracer_provider: Optional[TracerProvider] = None
        self.resources = {}
        # Task-local storage for async task isolation
        self.task_sessions: WeakKeyDictionary = WeakKeyDictionary()
        # Thread-local storage for thread isolation
        self.thread_local = threading.local()

    def reset(self):
        """Reset SDK state."""
        # Shutdown telemetry first to ensure all spans are exported
        if self.tracer_provider:
            try:
                # Force flush all pending spans with 5 second timeout
                debug("[SDK] Flushing OpenTelemetry spans...")
                self.tracer_provider.force_flush(timeout_millis=5000)
                # Shutdown the tracer provider and all processors
                self.tracer_provider.shutdown()
                debug("[SDK] TracerProvider shutdown complete")
            except Exception as e:
                error(f"[SDK] Error shutting down TracerProvider: {e}")

        if self.http:
            self.http.close()

        self.http = None
        self.session_id = None
        self.tracer_provider = None
        self.resources = {}
        self.task_sessions.clear()
        # Clear thread-local storage for current thread
        if hasattr(self.thread_local, 'session_id'):
            delattr(self.thread_local, 'session_id')


# Global SDK state
_sdk_state = SDKState()


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
    global _sdk_state
    
    # Initialize HTTP client
    if not _sdk_state.http:
        debug("[SDK] Initializing HTTP client")
        _sdk_state.http = HttpClient(config)
    
    # Initialize resources
    if not _sdk_state.resources:
        _sdk_state.resources = {
            'events': EventResource(_sdk_state.http),
            'sessions': SessionResource(_sdk_state.http),
            'datasets': DatasetResource(_sdk_state.http)
        }


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
    global _sdk_state
    
    _sdk_state.session_id = real_session_id
    
    info(f"[SDK] Session created: {truncate_id(real_session_id)} (name: {session_name or 'Unnamed Session'})")
    
    # Set active session in context
    set_active_session(real_session_id)
    
    # Register session with shutdown manager
    debug(f"[SDK] Registering session with shutdown manager (auto_end={auto_end})")
    shutdown_manager = get_shutdown_manager()
    session_state = SessionState(
        session_id=real_session_id,
        http_client=_sdk_state.resources,
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
    global _sdk_state
    
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
    session_resource = _sdk_state.resources['sessions']
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
    global _sdk_state
    
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
    session_resource = _sdk_state.resources['sessions']
    session_data = await session_resource.acreate_session(session_params)
    
    # Use the session_id returned by the backend
    real_session_id = session_data.get('session_id', real_session_id)
    
    return _finalize_session(real_session_id, session_name, auto_end, providers)


# Deprecated alias for backwards compatibility
def init(
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
    """Initialize the Lucidic SDK.
    
    .. deprecated::
        Use :func:`create_session` instead. This function will be removed in a future version.
    
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
    import warnings
    warnings.warn(
        "init() is deprecated and will be removed in a future version. "
        "Use create_session() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_session(
        session_name=session_name,
        session_id=session_id,
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


def _initialize_telemetry(providers: List[str]) -> None:
    """Initialize telemetry providers.
    
    Args:
        providers: List of provider names
    """
    global _sdk_state
    
    if not _sdk_state.tracer_provider:
        # Import here to avoid circular dependency
        from ..telemetry.lucidic_exporter import LucidicSpanExporter
        from ..telemetry.context_capture_processor import ContextCaptureProcessor
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        
        # Create tracer provider with our processors
        _sdk_state.tracer_provider = TracerProvider()
        
        # Add context capture processor FIRST to capture context before export
        context_processor = ContextCaptureProcessor()
        _sdk_state.tracer_provider.add_span_processor(context_processor)
        
        # Add exporter processor
        exporter = LucidicSpanExporter()
        export_processor = BatchSpanProcessor(exporter)
        _sdk_state.tracer_provider.add_span_processor(export_processor)
    
    # Instrument providers
    instrument_providers(providers, _sdk_state.tracer_provider, {})
    
    info(f"[Telemetry] Initialized for providers: {providers}")


def set_task_session(session_id: str) -> None:
    """Set session ID for current async task (if in async context)."""
    try:
        if task := asyncio.current_task():
            _sdk_state.task_sessions[task] = session_id
            debug(f"[SDK] Set task-local session {truncate_id(session_id)} for task {task.get_name()}")
    except RuntimeError:
        # Not in async context, ignore
        pass


def clear_task_session() -> None:
    """Clear session ID for current async task (if in async context)."""
    try:
        if task := asyncio.current_task():
            _sdk_state.task_sessions.pop(task, None)
            debug(f"[SDK] Cleared task-local session for task {task.get_name()}")
    except RuntimeError:
        # Not in async context, ignore
        pass


def set_thread_session(session_id: str) -> None:
    """Set session ID for current thread.

    This provides true thread-local storage that doesn't inherit from parent thread.
    """
    _sdk_state.thread_local.session_id = session_id
    current_thread = threading.current_thread()
    debug(f"[SDK] Set thread-local session {truncate_id(session_id)} for thread {current_thread.name}")


def clear_thread_session() -> None:
    """Clear session ID for current thread."""
    if hasattr(_sdk_state.thread_local, 'session_id'):
        delattr(_sdk_state.thread_local, 'session_id')
        current_thread = threading.current_thread()
        debug(f"[SDK] Cleared thread-local session for thread {current_thread.name}")


def get_thread_session() -> Optional[str]:
    """Get session ID from thread-local storage."""
    return getattr(_sdk_state.thread_local, 'session_id', None)


def is_main_thread() -> bool:
    """Check if we're running in the main thread."""
    return threading.current_thread() is threading.main_thread()


def get_session_id() -> Optional[str]:
    """Get the current session ID.

    Priority:
    1. Task-local session (for async tasks)
    2. Thread-local session (for threads) - NO FALLBACK for threads
    3. SDK state session (for main thread)
    4. Context variable session (fallback for main thread only)
    """
    # First check task-local storage for async isolation
    try:
        if task := asyncio.current_task():
            if task_session := _sdk_state.task_sessions.get(task):
                debug(f"[SDK] Using task-local session {truncate_id(task_session)}")
                return task_session
    except RuntimeError:
        # Not in async context
        pass

    # Check if we're in a thread
    if not is_main_thread():
        # For threads, ONLY use thread-local storage - no fallback!
        # This prevents inheriting the parent thread's session
        thread_session = get_thread_session()
        if thread_session:
            debug(f"[SDK] Using thread-local session {truncate_id(thread_session)}")
        else:
            debug(f"[SDK] Thread {threading.current_thread().name} has no thread-local session")
        return thread_session  # Return None if not set - don't fall back!

    # For main thread only: fall back to SDK state or context variable
    return _sdk_state.session_id or current_session_id.get()


def get_http() -> Optional[HttpClient]:
    """Get the HTTP client instance."""
    return _sdk_state.http


def get_resources() -> dict:
    """Get API resource instances."""
    return _sdk_state.resources


def set_http(http: HttpClient) -> None:
    """Set the HTTP client instance in SDK state."""
    global _sdk_state
    _sdk_state.http = http


def set_resources(resources: dict) -> None:
    """Set API resource instances in SDK state."""
    global _sdk_state
    _sdk_state.resources = resources


def _get_credentials(api_key: Optional[str], agent_id: Optional[str]) -> tuple[str, Optional[str]]:
    """Get API credentials from arguments or environment.
    
    Returns:
        Tuple of (api_key, agent_id)
        
    Raises:
        APIKeyVerificationError: If API key is not available
    """
    from dotenv import load_dotenv
    import os
    from ..core.errors import APIKeyVerificationError
    
    load_dotenv()
    
    if api_key is None:
        api_key = os.getenv("LUCIDIC_API_KEY", None)
        if api_key is None:
            raise APIKeyVerificationError(
                "Make sure to either pass your API key or set the LUCIDIC_API_KEY environment variable."
            )
    
    if agent_id is None:
        agent_id = os.getenv("LUCIDIC_AGENT_ID", None)
    
    return api_key, agent_id


def _init_http_and_resources(api_key: str, agent_id: Optional[str]) -> dict:
    """Initialize HTTP client and resources.
    
    Returns:
        Dictionary of API resources
    """
    global _sdk_state
    
    # Create or reuse HTTP client
    if not _sdk_state.http:
        debug("[SDK] Creating HTTP client for standalone use")
        config = SDKConfig.from_env(api_key=api_key, agent_id=agent_id)
        _sdk_state.http = HttpClient(config)
    
    # Create resources if not already present
    if not _sdk_state.resources:
        _sdk_state.resources = {}
    
    if 'datasets' not in _sdk_state.resources:
        debug("[SDK] Creating DatasetResource for standalone use")
        _sdk_state.resources['datasets'] = DatasetResource(_sdk_state.http)
    
    return _sdk_state.resources


def ensure_http_and_resources(api_key: Optional[str] = None, agent_id: Optional[str] = None) -> dict:
    """Ensure HTTP client and resources are initialized, creating them if needed (synchronous).
    
    This function checks if the HTTP client and resources already exist in SDK state.
    If not, it creates them and stores them in SDK state for reuse.
    
    Args:
        api_key: API key (uses env if not provided)
        agent_id: Agent ID (uses env if not provided)
        
    Returns:
        Dictionary of API resources with 'datasets' key
        
    Raises:
        APIKeyVerificationError: If API key is not available
    """
    global _sdk_state
    
    # If we already have resources with datasets, return them
    if _sdk_state.resources and 'datasets' in _sdk_state.resources:
        return _sdk_state.resources
    
    api_key, agent_id = _get_credentials(api_key, agent_id)
    return _init_http_and_resources(api_key, agent_id)


async def aensure_http_and_resources(api_key: Optional[str] = None, agent_id: Optional[str] = None) -> dict:
    """Ensure HTTP client and resources are initialized, creating them if needed (asynchronous).
    
    This function checks if the HTTP client and resources already exist in SDK state.
    If not, it creates them and stores them in SDK state for reuse.
    
    Note: This async version shares the same initialization logic as the sync version
    since the HTTP client and resources creation is synchronous. The async version
    exists to maintain consistent API patterns in async contexts.
    
    Args:
        api_key: API key (uses env if not provided)
        agent_id: Agent ID (uses env if not provided)
        
    Returns:
        Dictionary of API resources with 'datasets' key
        
    Raises:
        APIKeyVerificationError: If API key is not available
    """
    global _sdk_state
    
    # If we already have resources with datasets, return them
    if _sdk_state.resources and 'datasets' in _sdk_state.resources:
        return _sdk_state.resources
    
    api_key, agent_id = _get_credentials(api_key, agent_id)
    return _init_http_and_resources(api_key, agent_id)


def get_tracer_provider() -> Optional[TracerProvider]:
    """Get the tracer provider instance."""
    return _sdk_state.tracer_provider


def clear_state() -> None:
    """Clear SDK state (for testing)."""
    global _sdk_state
    debug("[SDK] Clearing SDK state")
    _sdk_state.reset()
    _sdk_state = SDKState()