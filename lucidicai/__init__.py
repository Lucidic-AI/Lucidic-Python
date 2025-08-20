import atexit
import logging
import os
import signal
import sys
import traceback
import threading
from typing import List, Literal, Optional

from .client import Client
from .errors import APIKeyVerificationError, InvalidOperationError, LucidicNotInitializedError, PromptError
from .event import Event
from .session import Session
from .step import Step

# Import OpenTelemetry-based handlers
from .telemetry.otel_handlers import (
    OTelOpenAIHandler,
    OTelAnthropicHandler,
    OTelLangChainHandler,
    OTelPydanticAIHandler,
    OTelOpenAIAgentsHandler,
    OTelLiteLLMHandler
)

# Import telemetry manager
from .telemetry.otel_init import LucidicTelemetry

# Import decorators
from .decorators import step, event
from .context import (
    set_active_session,
    bind_session,
    bind_session_async,
    clear_active_session,
    current_session_id,
    session,
    session_async,
    run_session,
    run_in_session,
)

ProviderType = Literal[
    "openai",
    "anthropic",
    "langchain",
    "pydantic_ai",
    "openai_agents",
    "litellm",
    "bedrock",
    "aws_bedrock",
    "amazon_bedrock",
    "google",
    "google_generativeai",
    "vertexai",
    "vertex_ai",
    "cohere",
    "groq",
]

# Configure logging
logger = logging.getLogger("Lucidic")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[Lucidic] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Crash/exit capture configuration
MAX_ERROR_DESCRIPTION_LENGTH = 16384
_crash_handlers_installed = False
_original_sys_excepthook = None
_original_threading_excepthook = None
_shutdown_lock = threading.Lock()
_is_shutting_down = False


def _mask_and_truncate(text: Optional[str]) -> Optional[str]:
    """Apply masking and truncate to a safe length. Best effort; never raises."""
    if text is None:
        return text
    try:
        masked = Client().mask(text)
    except Exception:
        masked = text
    if masked is None:
        return masked
    return masked[:MAX_ERROR_DESCRIPTION_LENGTH]


def _post_fatal_event(exit_code: int, description: str, extra: Optional[dict] = None) -> None:
    """Best-effort creation of a final Lucidic event on fatal paths.

    - Idempotent using a process-wide shutdown flag to avoid duplicates when
      multiple hooks fire (signal + excepthook).
    - Swallows all exceptions to avoid interfering with shutdown.
    """
    global _is_shutting_down
    with _shutdown_lock:
        if _is_shutting_down:
            return
        _is_shutting_down = True
    try:
        client = Client()
        session = getattr(client, 'session', None)
        if not session or getattr(session, 'is_finished', False):
            return
        arguments = {"exit_code": exit_code}
        if extra:
            try:
                arguments.update(extra)
            except Exception:
                pass

        event_id = session.create_event(
            description=_mask_and_truncate(description),
            result=f"process exited with code {exit_code}",
            function_name="__process_exit__",
            arguments=arguments,
        )
        session.update_event(event_id=event_id, is_finished=True)
    except Exception:
        # Never raise during shutdown
        pass


def _install_crash_handlers() -> None:
    """Install global uncaught exception handlers (idempotent)."""
    global _crash_handlers_installed, _original_sys_excepthook, _original_threading_excepthook
    if _crash_handlers_installed:
        return

    _original_sys_excepthook = sys.excepthook

    def _sys_hook(exc_type, exc, tb):
        try:
            trace_str = ''.join(traceback.format_exception(exc_type, exc, tb))
        except Exception:
            trace_str = f"Uncaught exception: {getattr(exc_type, '__name__', str(exc_type))}: {exc}"

        # Emit final event and end the session as unsuccessful
        _post_fatal_event(1, trace_str, {
            "exception_type": getattr(exc_type, "__name__", str(exc_type)),
            "exception_message": str(exc),
            "thread_name": threading.current_thread().name,
        })
        try:
            # Prevent auto_end double work
            client = Client()
            try:
                client.auto_end = False
            except Exception:
                pass
            # End session explicitly as unsuccessful
            end_session()
        except Exception:
            pass
        # Best-effort force flush and shutdown telemetry
        try:
            telemetry = LucidicTelemetry()
            if telemetry.is_initialized():
                try:
                    telemetry.force_flush()
                except Exception:
                    pass
                try:
                    telemetry.uninstrument_all()
                except Exception:
                    pass
        except Exception:
            pass
        # Chain to original to preserve default printing/behavior
        try:
            _original_sys_excepthook(exc_type, exc, tb)
        except Exception:
            # Avoid recursion/errors in fatal path
            pass

    sys.excepthook = _sys_hook

    # For Python 3.8+, only treat main-thread exceptions as fatal (process-exiting)
    if hasattr(threading, 'excepthook'):
        _original_threading_excepthook = threading.excepthook

        def _thread_hook(args):
            try:
                if args.thread is threading.main_thread():
                    _sys_hook(args.exc_type, args.exc_value, args.exc_traceback)
            except Exception:
                pass
            try:
                _original_threading_excepthook(args)
            except Exception:
                pass

        threading.excepthook = _thread_hook

    _crash_handlers_installed = True

def _setup_providers(client: Client, providers: List[ProviderType]) -> None:
    """Set up providers for the client, avoiding duplication
    
    Args:
        client: The Lucidic client instance
        providers: List of provider types to set up
    """
    # Track which providers have been set up to avoid duplication
    setup_providers = set()
    
    # Initialize telemetry if using OpenTelemetry
    if providers:
        telemetry = LucidicTelemetry()
        if not telemetry.is_initialized():
            telemetry.initialize(agent_id=client.agent_id)
    
    for provider in providers:
        if provider in setup_providers:
            continue
            
        if provider == "openai":
            client.set_provider(OTelOpenAIHandler())
            setup_providers.add("openai")
        elif provider == "anthropic":
            client.set_provider(OTelAnthropicHandler())
            setup_providers.add("anthropic")
        elif provider == "langchain":
            client.set_provider(OTelLangChainHandler())
            logger.info("For LangChain, make sure to create a handler and attach it to your top-level Agent class.")
            setup_providers.add("langchain")
        elif provider == "pydantic_ai":
            client.set_provider(OTelPydanticAIHandler())
            setup_providers.add("pydantic_ai")
        elif provider == "openai_agents":
            try:
                client.set_provider(OTelOpenAIAgentsHandler())
                setup_providers.add("openai_agents")
            except Exception as e:
                logger.error(f"Failed to set up OpenAI Agents provider: {e}")
                raise
        elif provider == "litellm":
            client.set_provider(OTelLiteLLMHandler())
            setup_providers.add("litellm")
        elif provider in ("bedrock", "aws_bedrock", "amazon_bedrock"):
            from .telemetry.otel_handlers import OTelBedrockHandler
            client.set_provider(OTelBedrockHandler())
            setup_providers.add("bedrock")
        elif provider in ("google", "google_generativeai"):
            from .telemetry.otel_handlers import OTelGoogleGenerativeAIHandler
            client.set_provider(OTelGoogleGenerativeAIHandler())
            setup_providers.add("google")
        elif provider in ("vertexai", "vertex_ai"):
            from .telemetry.otel_handlers import OTelVertexAIHandler
            client.set_provider(OTelVertexAIHandler())
            setup_providers.add("vertexai")
        elif provider == "cohere":
            from .telemetry.otel_handlers import OTelCohereHandler
            client.set_provider(OTelCohereHandler())
            setup_providers.add("cohere")
        elif provider == "groq":
            from .telemetry.otel_handlers import OTelGroqHandler
            client.set_provider(OTelGroqHandler())
            setup_providers.add("groq")

__all__ = [
    'Client',
    'Session',
    'Step',
    'Event',
    'init',
    'continue_session',
    'create_step',
    'end_step',
    'update_step',
    'create_event',
    'update_event',
    'end_event',
    'end_session',
    'get_prompt',
    'get_session',
    'ProviderType',
    'APIKeyVerificationError',
    'LucidicNotInitializedError',
    'PromptError',
    'InvalidOperationError',
    'step',
    'event',
    'set_active_session',
    'bind_session',
    'bind_session_async',
    'clear_active_session',
    'session',
    'session_async',
    'run_session',
    'run_in_session',
]


def init(
    session_name: Optional[str] = None,
    session_id: Optional[str] = None,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    task: Optional[str] = None,
    providers: Optional[List[ProviderType]] = [],
    production_monitoring: Optional[bool] = False,
    mass_sim_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    rubrics: Optional[list] = None,
    tags: Optional[list] = None,
    masking_function = None,
    auto_end: Optional[bool] = True,
    capture_uncaught: Optional[bool] = True,
) -> str:
    """
    Initialize the Lucidic client.
    
    Args:
        session_name: The display name of the session.
        session_id: Custom ID of the session. If not provided, a random ID will be generated.
        api_key: API key for authentication. If not provided, will use the LUCIDIC_API_KEY environment variable.
        agent_id: Agent ID. If not provided, will use the LUCIDIC_AGENT_ID environment variable.
        task: Task description.
        providers: List of provider types ("openai", "anthropic", "langchain", "pydantic_ai").
        mass_sim_id: Optional mass simulation ID, if session is to be part of a mass simulation.
        experiment_id: Optional experiment ID, if session is to be part of an experiment.
        rubrics: Optional rubrics for evaluation, list of strings.
        tags: Optional tags for the session, list of strings.
        masking_function: Optional function to mask sensitive data.
        auto_end: If True, automatically end the session on process exit. Defaults to True.
    
    Raises:
        InvalidOperationError: If the client is already initialized.
        APIKeyVerificationError: If the API key is invalid.
    """
    
    # get current client which will be NullClient if never lai is never initialized
    client = Client()
    # if not yet initialized or still the NullClient -> creaet a real client when init is called
    if not getattr(client, 'initialized', False):
        if api_key is None:
            api_key = os.getenv("LUCIDIC_API_KEY", None)
            if api_key is None:
                raise APIKeyVerificationError("Make sure to either pass your API key into lai.init() or set the LUCIDIC_API_KEY environment variable.")
        if agent_id is None:
            agent_id = os.getenv("LUCIDIC_AGENT_ID", None)
            if agent_id is None:
                raise APIKeyVerificationError("Lucidic agent ID not specified. Make sure to either pass your agent ID into lai.init() or set the LUCIDIC_AGENT_ID environment variable.")
        client = Client(api_key=api_key, agent_id=agent_id)
    else:
        # Already initialized, this is a re-init
        api_key = api_key or os.getenv("LUCIDIC_API_KEY", None)
        agent_id = agent_id or os.getenv("LUCIDIC_AGENT_ID", None)
        client.agent_id = agent_id
        if api_key is not None and agent_id is not None and (api_key != client.api_key or agent_id != client.agent_id):
            client.set_api_key(api_key)
            client.agent_id = agent_id
        
    
    # Handle auto_end with environment variable support
    if auto_end is None:
        auto_end = os.getenv("LUCIDIC_AUTO_END", "True").lower() == "true"
    
    # Set up providers
    _setup_providers(client, providers)
    real_session_id = client.init_session(
        session_name=session_name,
        mass_sim_id=mass_sim_id,
        task=task,
        rubrics=rubrics,
        tags=tags,
        production_monitoring=production_monitoring,
        session_id=session_id,
        experiment_id=experiment_id,
    )
    if masking_function:
        client.masking_function = masking_function
    
    # Set the auto_end flag on the client
    client.auto_end = auto_end
    # Bind this session id to the current execution context for async-safety
    try:
        set_active_session(real_session_id)
    except Exception:
        pass
    # Install crash handlers unless explicitly disabled
    try:
        if capture_uncaught:
            _install_crash_handlers()
    except Exception:
        pass
    
    logger.info("Session initialized successfully")
    return real_session_id


def continue_session(
    session_id: str,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    providers: Optional[List[ProviderType]] = [],
    masking_function = None,
    auto_end: Optional[bool] = True,
):
    if api_key is None:
        api_key = os.getenv("LUCIDIC_API_KEY", None)
        if api_key is None:
            raise APIKeyVerificationError("Make sure to either pass your API key into lai.init() or set the LUCIDIC_API_KEY environment variable.")
    if agent_id is None:
        agent_id = os.getenv("LUCIDIC_AGENT_ID", None)
        if agent_id is None:
            raise APIKeyVerificationError("Lucidic agent ID not specified. Make sure to either pass your agent ID into lai.init() or set the LUCIDIC_AGENT_ID environment variable.")
    
    client = Client()
    if client.session:
        raise InvalidOperationError("[Lucidic] Session already in progress. Please call lai.end_session() or lai.reset_sdk() first.")
    # if not yet initialized or still the NullClient -> create a real client when init is called
    if not getattr(client, 'initialized', False):
        client = Client(api_key=api_key, agent_id=agent_id)
    
    # Handle auto_end with environment variable support
    if auto_end is None:
        auto_end = os.getenv("LUCIDIC_AUTO_END", "True").lower() == "true"
    
    # Set up providers
    _setup_providers(client, providers)
    session_id = client.continue_session(session_id=session_id)
    if masking_function:
        client.masking_function = masking_function
    
    # Set the auto_end flag on the client
    client.auto_end = auto_end
    
    logger.info(f"Session {session_id} continuing...")
    # Bind this session id to the current execution context for async-safety
    try:
        set_active_session(session_id)
    except Exception:
        pass
    return session_id  # For consistency


def update_session(
    task: Optional[str] = None,
    session_eval: Optional[float] = None,
    session_eval_reason: Optional[str] = None,
    is_successful: Optional[bool] = None,
    is_successful_reason: Optional[str] = None
) -> None:
    """
    Update the current session.
    
    Args:
        task: Task description.
        session_eval: Session evaluation.
        session_eval_reason: Session evaluation reason.
        is_successful: Whether the session was successful.
        is_successful_reason: Session success reason.
    """
    # Prefer context-bound session over global active session
    client = Client()
    target_sid = None
    try:
        target_sid = current_session_id.get(None)
    except Exception:
        target_sid = None
    if not target_sid and client.session:
        target_sid = client.session.session_id
    if not target_sid:
        return
    # Use ephemeral session facade to avoid mutating global state
    session = client.session if (client.session and client.session.session_id == target_sid) else Session(agent_id=client.agent_id, session_id=target_sid)
    session.update_session(**locals())


def end_session(
    session_eval: Optional[float] = None,
    session_eval_reason: Optional[str] = None,
    is_successful: Optional[bool] = None,
    is_successful_reason: Optional[str] = None
) -> None:
    """
    End the current session.
    
    Args:
        session_eval: Session evaluation.
        session_eval_reason: Session evaluation reason.
        is_successful: Whether the session was successful.
        is_successful_reason: Session success reason.
    """
    client = Client()
    # Prefer context-bound session id
    target_sid = None
    try:
        target_sid = current_session_id.get(None)
    except Exception:
        target_sid = None
    if not target_sid and client.session:
        target_sid = client.session.session_id
    if not target_sid:
        return

    # If ending the globally active session, keep existing cleanup behavior
    if client.session and client.session.session_id == target_sid:
        # Wait for any pending LiteLLM callbacks before ending session
        for provider in client.providers:
            if hasattr(provider, '_callback') and hasattr(provider._callback, 'wait_for_pending_callbacks'):
                logger.info("Waiting for LiteLLM callbacks to complete before ending session...")
                provider._callback.wait_for_pending_callbacks(timeout=5.0)
        client.session.update_session(is_finished=True, **locals())
        client.clear()
        return

    # Otherwise, end the specified session id without clearing global state
    temp = Session(agent_id=client.agent_id, session_id=target_sid)
    temp.update_session(is_finished=True, **locals())


def reset_sdk() -> None:
    """
    DEPRECATED: Reset the SDK.
    """
    return

    client = Client()
    if not client.initialized:
        return
    
    # Shutdown OpenTelemetry if it was initialized
    telemetry = LucidicTelemetry()
    if telemetry.is_initialized():
        telemetry.uninstrument_all()
    
    client.clear()


def _cleanup_telemetry():
    """Cleanup function for OpenTelemetry shutdown"""
    try:
        telemetry = LucidicTelemetry()
        if telemetry.is_initialized():
            telemetry.uninstrument_all()
            logger.info("OpenTelemetry instrumentation cleaned up")
    except Exception as e:
        logger.error(f"Error during telemetry cleanup: {e}")


def _auto_end_session():
    """Automatically end session on exit if auto_end is enabled"""
    try:
        client = Client()
        if hasattr(client, 'auto_end') and client.auto_end and client.session and not client.session.is_finished:
            logger.info("Auto-ending active session on exit")
            client.auto_end = False  # To avoid repeating auto-end on exit
            end_session()
    except Exception as e:
        logger.debug(f"Error during auto-end session: {e}")


def _signal_handler(signum, frame):
    """Handle interruption signals"""
    # Best-effort final event for signal exits
    try:
        try:
            name = signal.Signals(signum).name
        except Exception:
            name = str(signum)
        try:
            stack_str = ''.join(traceback.format_stack(frame)) if frame else ''
        except Exception:
            stack_str = ''
        desc = _mask_and_truncate(f"Received signal {name}\n{stack_str}")
        _post_fatal_event(128 + signum, desc, {"signal": name, "signum": signum})
    except Exception:
        pass
    _auto_end_session()
    _cleanup_telemetry()
    # Re-raise the signal for default handling
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Register cleanup functions (auto-end runs first due to LIFO order)
atexit.register(_cleanup_telemetry)
atexit.register(_auto_end_session)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def create_mass_sim(
    mass_sim_name: str,
    total_num_sessions: int,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    task: Optional[str] = None,
    tags: Optional[list] = None
) -> str:
    """
    Create a new mass simulation.
    
    Args:
        mass_sim_name: Name of the mass simulation.
        total_num_sessions: Total intended number of sessions. More sessions can be added later.
        api_key: API key for authentication. If not provided, will use the LUCIDIC_API_KEY environment variable.
        agent_id: Agent ID. If not provided, will use the LUCIDIC_AGENT_ID environment variable.
        task: Task description.
        tags: Tags for the mass simulation.
    
    Returns:
        mass_sim_id: ID of the created mass simulation. Pass this to lai.init() to create a new session in the mass sim.
    """
    if api_key is None:
        api_key = os.getenv("LUCIDIC_API_KEY", None)
        if api_key is None:
            raise APIKeyVerificationError("Make sure to either pass your API key into lai.init() or set the LUCIDIC_API_KEY environment variable.")
    if agent_id is None:
        agent_id = os.getenv("LUCIDIC_AGENT_ID", None)
        if agent_id is None:
            raise APIKeyVerificationError("Lucidic agent ID not specified. Make sure to either pass your agent ID into lai.init() or set the LUCIDIC_AGENT_ID environment variable.")
    try:
        client = Client()
    except LucidicNotInitializedError:
        client = Client( # TODO: fail hard if incorrect API key or agent ID provided and wrong, fail silently if not provided
            api_key=api_key,
            agent_id=agent_id,
        )
    mass_sim_id = client.init_mass_sim(mass_sim_name=mass_sim_name, total_num_sims=total_num_sessions, task=task, tags=tags)  # TODO: change total_num_sims to total_num_sessions everywhere
    logger.info(f"Created mass simulation with ID: {mass_sim_id}")
    return mass_sim_id


def create_step(
    state: Optional[str] = None, 
    action: Optional[str] = None, 
    goal: Optional[str] = None,
    eval_score: Optional[float] = None,
    eval_description: Optional[str] = None,
    screenshot: Optional[str] = None,
    screenshot_path: Optional[str] = None
) -> None:
    """
    Create a new step. Previous step must be finished to create a new step.
    
    Args:
        state: State description.
        action: Action description.
        goal: Goal description.
        eval_score: Evaluation score.
        eval_description: Evaluation description.
        screenshot: Screenshot encoded in base64. Provide either screenshot or screenshot_path.
        screenshot_path: Screenshot path. Provide either screenshot or screenshot_path.
    """
    client = Client()
    if not client.session:
        return
    return client.session.create_step(**locals())


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
    """
    Update the current step.
    
    Args:
        step_id: ID of the step to update.
        state: State description.
        action: Action description.
        goal: Goal description.
        eval_score: Evaluation score.
        eval_description: Evaluation description.
        screenshot: Screenshot encoded in base64. Provide either screenshot or screenshot_path.
        screenshot_path: Screenshot path. Provide either screenshot or screenshot_path.
    """
    client = Client()
    if not client.session:
        return
    if not client.session.active_step:
        raise InvalidOperationError("No active step to update")
    client.session.update_step(**locals())


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
    """
    End the current step.
    
    Args:
        step_id: ID of the step to end.
        state: State description.
        action: Action description.
        goal: Goal description.
        eval_score: Evaluation score.
        eval_description: Evaluation description.
        screenshot: Screenshot encoded in base64. Provide either screenshot or screenshot_path.
        screenshot_path: Screenshot path.
    """
    client = Client()
    if not client.session:
        return
    
    if not client.session.active_step and step_id is None:
        raise InvalidOperationError("No active step to end")
    
    # Filter out None values from locals
    params = locals()
    kwargs = {k: v for k, v in params.items() if v is not None and k not in ['client', 'params']}
    kwargs['is_finished'] = True
    
    client.session.update_step(**kwargs)


def create_event(
    step_id: Optional[str] = None,
    description: Optional[str] = None,
    result: Optional[str] = None,
    cost_added: Optional[float] = None, 
    model: Optional[str] = None,
    screenshots: Optional[List[str]] = None,
    function_name: Optional[str] = None,
    arguments: Optional[dict] = None,
) -> str:
    """
    Create a new event in the current step. Current step must not be finished.
    
    Args:
        description: Description of the event.
        result: Result of the event.
        cost_added: Cost added by the event.
        model: Model used for the event.
        screenshots: List of screenshots encoded in base64.
        function_name: Name of the function that created the event.
        arguments: Arguments of the function that created the event.
    """

    client = Client()
    if not client.session:
        return
    return client.session.create_event(**locals())


def update_event(
    event_id: Optional[str] = None,
    description: Optional[str] = None,
    result: Optional[str] = None,
    cost_added: Optional[float] = None, 
    model: Optional[str] = None,
    screenshots: Optional[List[str]] = None,
    function_name: Optional[str] = None,
    arguments: Optional[dict] = None,
) -> None:
    """
    Update the event with the given ID in the current step.
    
    Args:
        event_id: ID of the event to update.
        description: Description of the event.
        result: Result of the event.
        cost_added: Cost added by the event.
        model: Model used for the event.
        screenshots: List of screenshots encoded in base64.
        function_name: Name of the function that created the event.
        arguments: Arguments of the function that created the event.
    """
    client = Client()
    if not client.session:
        return
    client.session.update_event(**locals())


def end_event(
    event_id: Optional[str] = None,
    description: Optional[str] = None,
    result: Optional[str] = None,
    cost_added: Optional[float] = None, 
    model: Optional[str] = None,
    screenshots: Optional[List[str]] = None,
    function_name: Optional[str] = None,
    arguments: Optional[dict] = None,
) -> None:
    """
    End the latest event in the current step.
    
    Args:
        event_id: ID of the event to end.
        description: Description of the event.
        result: Result of the event.
        cost_added: Cost added by the event.
        model: Model used for the event.
        screenshots: List of screenshots encoded in base64.
        function_name: Name of the function that created the event.
        arguments: Arguments of the function that created the event.
    """
    client = Client()
    if not client.session:
        return
    client.session.update_event(is_finished=True, **locals())


def get_prompt(
    prompt_name: str, 
    variables: Optional[dict] = None,
    cache_ttl: Optional[int] = 300,
    label: Optional[str] = 'production'
) -> str:
    """
    Get a prompt from the prompt database.
    
    Args:
        prompt_name: Name of the prompt.
        variables: {{Variables}} to replace in the prompt, supplied as a dictionary.
        cache_ttl: Time-to-live for the prompt in the cache in seconds (default: 300). Set to -1 to cache forever. Set to 0 to disable caching.
        label: Optional label for the prompt.
    
    Returns:
        str: The prompt.
    """
    client = Client()
    if not client.session:
        return ""
    prompt = client.get_prompt(prompt_name, cache_ttl, label)
    if variables:
        for key, val in variables.items():
            index = prompt.find("{{" + key +"}}")
            if index == -1:
                raise PromptError("Supplied variable not found in prompt")
            prompt = prompt.replace("{{" + key +"}}", str(val))
    if "{{" in prompt and "}}" in prompt and prompt.find("{{") < prompt.find("}}"):
        logger.warning("Unreplaced variable(s) left in prompt. Please check your prompt.")
    return prompt


def get_session():
    """Get the current session object
    
    Returns:
        Session: The current session object, or None if no session exists
    """
    try:
        client = Client()
        return client.session
    except (LucidicNotInitializedError, AttributeError) as e:
        logger.debug(f"No active session: {str(e)}")
        return None


