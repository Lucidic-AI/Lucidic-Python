"""SDK initialization module.

This module handles SDK initialization, separating concerns from the main __init__.py
"""
import logging
import uuid
from typing import List, Optional

from ..api.client import HttpClient
from ..api.resources.event import EventResource
from ..api.resources.session import SessionResource
from ..core.config import SDKConfig, get_config, set_config
from ..utils.queue import EventQueue
from ..context import set_active_session, current_session_id
from .error_boundary import register_cleanup_handler
from .shutdown_manager import get_shutdown_manager, SessionState
from ..telemetry.telemetry_init import instrument_providers
from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger("Lucidic")


class SDKState:
    """Container for SDK runtime state."""
    
    def __init__(self):
        self.http: Optional[HttpClient] = None
        self.event_queue: Optional[EventQueue] = None
        self.session_id: Optional[str] = None
        self.tracer_provider: Optional[TracerProvider] = None
        self.resources = {}
    
    def reset(self):
        """Reset SDK state."""
        if self.event_queue:
            self.event_queue.shutdown()
        if self.http:
            self.http.close()
        
        self.http = None
        self.event_queue = None
        self.session_id = None
        self.tracer_provider = None
        self.resources = {}


# Global SDK state
_sdk_state = SDKState()


def init(
    session_name: Optional[str] = None,
    session_id: Optional[str] = None,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    task: Optional[str] = None,
    providers: Optional[List[str]] = None,
    production_monitoring: bool = False,
    experiment_id: Optional[str] = None,
    rubrics: Optional[List] = None,
    tags: Optional[List] = None,
    dataset_item_id: Optional[str] = None,
    masking_function: Optional[callable] = None,
    auto_end: bool = True,
    capture_uncaught: bool = True,
) -> str:
    """Initialize the Lucidic SDK.
    
    Args:
        session_name: Name for the session
        session_id: Custom session ID (optional)
        api_key: API key (uses env if not provided)
        agent_id: Agent ID (uses env if not provided)
        task: Task description
        providers: List of telemetry providers to instrument
        production_monitoring: Enable production monitoring
        experiment_id: Experiment ID to associate with session
        rubrics: Evaluation rubrics
        tags: Session tags
        dataset_item_id: Dataset item ID
        masking_function: Function to mask sensitive data
        auto_end: Automatically end session on exit
        capture_uncaught: Capture uncaught exceptions
        
    Returns:
        Session ID
        
    Raises:
        APIKeyVerificationError: If API credentials are invalid
    """
    global _sdk_state
    
    # Create or update configuration
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
    
    # Set global config
    set_config(config)
    
    # Initialize HTTP client
    if not _sdk_state.http:
        _sdk_state.http = HttpClient(config)
    
    # Initialize resources
    if not _sdk_state.resources:
        _sdk_state.resources = {
            'events': EventResource(_sdk_state.http),
            'sessions': SessionResource(_sdk_state.http)
        }
    
    # Initialize event queue
    if not _sdk_state.event_queue:
        # Create a mock client object for backward compatibility
        # The queue needs a client with make_request method
        class ClientAdapter:
            def make_request(self, endpoint, method, data):
                return _sdk_state.http.request(method, endpoint, json=data)
        
        _sdk_state.event_queue = EventQueue(ClientAdapter())
        
        # Register cleanup handler
        register_cleanup_handler(lambda: _sdk_state.event_queue.force_flush())
    
    # Create or retrieve session
    if session_id:
        # Use provided session ID
        real_session_id = session_id
    else:
        # Create new session
        real_session_id = str(uuid.uuid4())
    
    # Create session via API - only send non-None values
    session_params = {
        'session_id': real_session_id,
        'session_name': session_name or 'Unnamed Session',
        'agent_id': config.agent_id,
    }
    
    # Only add optional fields if they have values
    if task:
        session_params['task'] = task
    if tags:
        session_params['tags'] = tags
    if experiment_id:
        session_params['experiment_id'] = experiment_id
    if dataset_item_id:
        session_params['dataset_item_id'] = dataset_item_id
    if rubrics:
        session_params['rubrics'] = rubrics
    if production_monitoring:
        session_params['production_monitoring'] = production_monitoring
    
    session_resource = _sdk_state.resources['sessions']
    session_data = session_resource.create_session(session_params)
    
    # Use the session_id returned by the backend
    real_session_id = session_data.get('session_id', real_session_id)
    _sdk_state.session_id = real_session_id
    
    # Set active session in context
    set_active_session(real_session_id)
    
    # Register session with shutdown manager
    shutdown_manager = get_shutdown_manager()
    session_state = SessionState(
        session_id=real_session_id,
        http_client=_sdk_state.resources,  # Pass resources dict which has sessions
        event_queue=_sdk_state.event_queue,
        auto_end=auto_end
    )
    shutdown_manager.register_session(real_session_id, session_state)
    
    # Initialize telemetry if providers specified
    if providers:
        _initialize_telemetry(providers)
    
    logger.info(f"SDK initialized with session {real_session_id[:8]}...")
    
    return real_session_id


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
    
    logger.info(f"Telemetry initialized for providers: {providers}")


def get_session_id() -> Optional[str]:
    """Get the current session ID."""
    return _sdk_state.session_id or current_session_id.get()


def get_http() -> Optional[HttpClient]:
    """Get the HTTP client instance."""
    return _sdk_state.http


def get_event_queue() -> Optional[EventQueue]:
    """Get the event queue instance."""
    return _sdk_state.event_queue


def get_resources() -> dict:
    """Get API resource instances."""
    return _sdk_state.resources


def clear_state() -> None:
    """Clear SDK state (for testing)."""
    global _sdk_state
    _sdk_state.reset()
    _sdk_state = SDKState()