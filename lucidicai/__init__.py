import atexit
import logging
import os
import signal
import sys
import traceback
import threading
from typing import List, Literal, Optional

from dotenv import load_dotenv

from .client import Client
from .errors import APIKeyVerificationError, InvalidOperationError, LucidicNotInitializedError, PromptError
from .event import Event
from .session import Session
from .singleton import clear_singletons

# Import decorators
from .decorators import event
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
from .dataset import get_dataset, get_dataset_items
from .feature_flag import (
    get_feature_flag,
    get_bool_flag,
    get_int_flag,
    get_float_flag,
    get_string_flag,
    get_json_flag,
    clear_feature_flag_cache,
    FeatureFlagError
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

        # Create a single immutable event describing the crash
        session.create_event(
            type="error_traceback",
            error=_mask_and_truncate(description),
            traceback="",
            metadata={"exit_code": exit_code, **({} if not extra else extra)},
        )
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
        
        # Follow proper shutdown sequence to prevent broken pipes
        try:
            client = Client()
            
            # 1. Flush OpenTelemetry spans first
            if hasattr(client, '_tracer_provider'):
                try:
                    client._tracer_provider.force_flush(timeout_millis=5000)
                except Exception:
                    pass
            
            # 2. Flush and shutdown EventQueue (with active sessions cleared)
            if hasattr(client, "_event_queue"):
                try:
                    # Clear active sessions to allow shutdown
                    client._event_queue._active_sessions.clear()
                    client._event_queue.force_flush()
                    client._event_queue.shutdown(timeout=5.0)
                except Exception:
                    pass
            
            # 3. Shutdown TracerProvider after EventQueue
            if hasattr(client, '_tracer_provider'):
                try:
                    client._tracer_provider.shutdown()
                except Exception:
                    pass
            
            # 4. Mark client as shutting down to prevent new requests
            client._shutdown = True
            
            # 5. Prevent auto_end double work
            try:
                client.auto_end = False
            except Exception:
                pass
            
            # 6. End session explicitly as unsuccessful
            end_session()
            
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
                    # For main thread exceptions, use full shutdown sequence
                    _sys_hook(args.exc_type, args.exc_value, args.exc_traceback)
                else:
                    # For non-main threads, just flush spans without full shutdown
                    try:
                        client = Client()
                        # Flush any pending spans from this thread
                        if hasattr(client, '_tracer_provider'):
                            client._tracer_provider.force_flush(timeout_millis=1000)
                        # Force flush events but don't shutdown
                        if hasattr(client, "_event_queue"):
                            client._event_queue.force_flush()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                _original_threading_excepthook(args)
            except Exception:
                pass

        threading.excepthook = _thread_hook

    _crash_handlers_installed = True

__all__ = [
    'Session',
    'Event',
    'init',
    'create_experiment',
    'create_event',
    'end_session',
    'get_prompt',
    'get_session',
    'get_dataset',
    'get_dataset_items',
    'get_feature_flag',
    'get_bool_flag',
    'get_int_flag',
    'get_float_flag',
    'get_string_flag',
    'get_json_flag',
    'clear_feature_flag_cache',
    'FeatureFlagError',
    'ProviderType',
    'APIKeyVerificationError',
    'LucidicNotInitializedError',
    'PromptError',
    'InvalidOperationError',
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
    experiment_id: Optional[str] = None,
    rubrics: Optional[list] = None,
    tags: Optional[list] = None,
    dataset_item_id: Optional[str] = None,
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
        experiment_id: Optional experiment ID, if session is to be part of an experiment.
        rubrics: Optional rubrics for evaluation, list of strings.
        tags: Optional tags for the session, list of strings.
        dataset_item_id: Optional dataset item ID to link session to a dataset item.
        masking_function: Optional function to mask sensitive data.
        auto_end: If True, automatically end the session on process exit. Defaults to True.
    
    Raises:
        InvalidOperationError: If the client is already initialized.
        APIKeyVerificationError: If the API key is invalid.
    """

    load_dotenv()

    if os.getenv("LUCIDIC_DEBUG", "False").lower() == "true":
        logger.setLevel(logging.DEBUG)
    
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
    # Use the client's singleton telemetry initialization
    if providers:
        success = client.initialize_telemetry(providers)
        if not success:
            logger.warning("[Telemetry] Failed to initialize telemetry for some providers")
    real_session_id = client.init_session(
        session_name=session_name,
        task=task,
        rubrics=rubrics,
        tags=tags,
        production_monitoring=production_monitoring,
        session_id=session_id,
        experiment_id=experiment_id,
        dataset_item_id=dataset_item_id,
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
            # Also install error event handler for uncaught exceptions
            try:
                from .errors import install_error_handler
                install_error_handler()
            except Exception:
                pass
    except Exception:
        pass
    
    logger.info("Session initialized successfully")
    return real_session_id


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
    is_successful_reason: Optional[str] = None,
    wait_for_flush: bool = True
) -> None:
    """
    End the current session.
    
    Args:
        session_eval: Session evaluation.
        session_eval_reason: Session evaluation reason.
        is_successful: Whether the session was successful.
        is_successful_reason: Session success reason.
        wait_for_flush: Whether to block until event queue is empty (default True).
                 Set to False during signal handling to prevent hangs.
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

    # If ending the globally active session, perform cleanup
    if client.session and client.session.session_id == target_sid:
        # Best-effort: wait for LiteLLM callbacks to flush before ending
        try:
            import litellm  
            cbs = getattr(litellm, 'callbacks', None)
            if cbs:
                for cb in cbs:
                    try:
                        if hasattr(cb, 'wait_for_pending_callbacks'):
                            cb.wait_for_pending_callbacks(timeout=1)
                    except Exception:
                        pass
        except Exception:
            pass
        # CRITICAL: Flush OpenTelemetry spans FIRST (blocking)
        # This ensures all spans are converted to events before we flush the event queue
        try:
            if hasattr(client, '_tracer_provider') and client._tracer_provider:
                logger.debug("[Session] Flushing OpenTelemetry spans before session end...")
                # Force flush with generous timeout to ensure all spans are exported
                # The BatchSpanProcessor now exports every 100ms, so this should be quick
                success = client._tracer_provider.force_flush(timeout_millis=10000)  # 10 second timeout
                if not success:
                    logger.warning("[Session] OpenTelemetry flush timed out - some spans may be lost")
                else:
                    logger.debug("[Session] OpenTelemetry spans flushed successfully")
        except Exception as e:
            logger.debug(f"[Session] Failed to flush telemetry spans: {e}")
        
        # THEN flush event queue (which now contains events from flushed spans)
        try:
            if hasattr(client, '_event_queue'):
                logger.debug("[Session] Flushing event queue...")
                client._event_queue.force_flush(timeout_seconds=10.0)
                
                # Wait for queue to be completely empty (only if blocking)
                if wait_for_flush:
                    import time
                    wait_start = time.time()
                    max_wait = 10.0  # seconds - timeout for blob uploads
                    while not client._event_queue.is_empty():
                        if time.time() - wait_start > max_wait:
                            logger.warning(f"[Session] EventQueue not empty after {max_wait}s timeout")
                            break
                        time.sleep(0.1)
                    
                    if client._event_queue.is_empty():
                        logger.debug("[Session] EventQueue confirmed empty")
                else:
                    logger.debug("[Session] Non-blocking mode - skipping wait for empty queue")
        except Exception as e:
            logger.debug(f"[Session] Failed to flush event queue: {e}")
        
        # Mark session as inactive FIRST (prevents race conditions)
        client.mark_session_inactive(target_sid)
        
        # Send only expected fields to update endpoint
        update_kwargs = {
            "is_finished": True,
            "session_eval": session_eval,
            "session_eval_reason": session_eval_reason,
            "is_successful": is_successful,
            "is_successful_reason": is_successful_reason,
        }
        try:
            client.session.update_session(**update_kwargs)
        except Exception as e:
            logger.warning(f"[Session] Failed to update session: {e}")
        
        # Clear only the global session reference, not the singleton
        # This preserves the client and event queue for other threads
        client.session = None
        logger.debug(f"[Session] Ended global session {target_sid}")
        # DO NOT shutdown event queue - other threads may be using it
        # DO NOT call client.clear() - preserve singleton for other threads
        return

    # Otherwise, end the specified session id without clearing global state
    # First flush telemetry and event queue for non-global sessions too
    try:
        if hasattr(client, '_tracer_provider') and client._tracer_provider:
            logger.debug(f"[Session] Flushing OpenTelemetry spans for session {target_sid[:8]}...")
            success = client._tracer_provider.force_flush(timeout_millis=10000)
            if not success:
                logger.warning("[Session] OpenTelemetry flush timed out")
    except Exception as e:
        logger.debug(f"[Session] Failed to flush telemetry spans: {e}")
    
    # Flush and wait for event queue to empty
    try:
        if hasattr(client, '_event_queue'):
            logger.debug(f"[Session] Flushing event queue for session {target_sid[:8]}...")
            client._event_queue.force_flush(timeout_seconds=10.0)
            
            # Wait for queue to be completely empty (only if blocking)
            if wait_for_flush:
                import time
                wait_start = time.time()
                max_wait = 10.0  # seconds - timeout for blob uploads
                while not client._event_queue.is_empty():
                    if time.time() - wait_start > max_wait:
                        logger.warning(f"[Session] EventQueue not empty after {max_wait}s timeout")
                        break
                    time.sleep(0.1)
                
                if client._event_queue.is_empty():
                    logger.debug(f"[Session] EventQueue confirmed empty for session {target_sid[:8]}")
            else:
                logger.debug(f"[Session] Non-blocking mode - skipping wait for session {target_sid[:8]}")
    except Exception as e:
        logger.debug(f"[Session] Failed to flush event queue: {e}")
    
    # CRITICAL: Mark session as inactive FIRST for ALL sessions
    client.mark_session_inactive(target_sid)
    
    temp = Session(agent_id=client.agent_id, session_id=target_sid)
    update_kwargs = {
        "is_finished": True,
        "session_eval": session_eval,
        "session_eval_reason": session_eval_reason,
        "is_successful": is_successful,
        "is_successful_reason": is_successful_reason,
    }
    try:
        temp.update_session(**update_kwargs)
    except Exception as e:
        logger.warning(f"[Session] Failed to update session: {e}")


def flush(timeout_seconds: float = 2.0) -> bool:
    """
    Manually flush all pending telemetry data.
    
    Flushes both OpenTelemetry spans and queued events to ensure
    all telemetry data is sent to the backend. This is called
    automatically on process exit but can be called manually
    for explicit control.
    
    Args:
        timeout_seconds: Maximum time to wait for flush
        
    Returns:
        True if all flushes succeeded, False otherwise
        
    Example:
        ```python
        import lucidicai as lai
        
        # ... your code using Lucidic ...
        
        # Manually flush before critical operation
        lai.flush()
        ```
    """
    try:
        client = Client()
        success = True
        
        # Flush OpenTelemetry spans first
        if hasattr(client, 'flush_telemetry'):
            span_success = client.flush_telemetry(timeout_seconds)
            success = success and span_success
        
        # Then flush event queue
        if hasattr(client, '_event_queue'):
            client._event_queue.force_flush(timeout_seconds)
            
        logger.debug(f"[Flush] Manual flush completed (success={success})")
        return success
    except Exception as e:
        logger.error(f"Failed to flush telemetry: {e}")
        return False


def _auto_end_session():
    """Automatically end session on exit if auto_end is enabled"""
    try:
        client = Client()
        if hasattr(client, 'auto_end') and client.auto_end and client.session and not client.session.is_finished:
            logger.info("Auto-ending active session on exit")
            client.auto_end = False  # To avoid repeating auto-end on exit
            
            # Flush telemetry
            if hasattr(client, '_tracer_provider'):
                client._tracer_provider.force_flush(timeout_millis=5000)

            # Force flush event queue before ending session
            if hasattr(client, '_event_queue'):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("[Shutdown] Flushing event queue before session end")
                client._event_queue.force_flush(timeout_seconds=5.0)
            
            # Use non-blocking mode during shutdown to prevent hangs
            # The actual wait for queue empty happens in _cleanup_singleton_on_exit
            end_session(wait_for_flush=False)
                
    except Exception as e:
        logger.debug(f"Error during auto-end session: {e}")


def _cleanup_singleton_on_exit():
    """
    Clean up singleton resources only on process exit.
    
    CRITICAL ORDER:
    1. Flush OpenTelemetry spans (blocking) - ensures spans become events
    2. Flush EventQueue - sends all events including those from spans
    3. Close HTTP session - graceful TCP FIN prevents broken pipes
    4. Clear singletons - final cleanup
    
    This order is essential to prevent lost events and broken connections.
    """
    try:
        client = Client()
        
        # 1. FIRST: Flush OpenTelemetry spans (blocking until exported)
        # This is the critical fix - we must flush spans before events
        if hasattr(client, '_tracer_provider') and client._tracer_provider:
            try:
                # Small delay to ensure spans have reached the processor
                import time
                time.sleep(0.1)  # 100ms to let spans reach BatchSpanProcessor
                
                logger.debug("[Exit] Flushing OpenTelemetry spans...")
                # force_flush() blocks until all spans are exported or timeout
                success = client._tracer_provider.force_flush(timeout_millis=3000)
                if success:
                    logger.debug("[Exit] OpenTelemetry spans flushed successfully")
                else:
                    logger.warning("[Exit] OpenTelemetry flush timed out - some spans may be lost")
                
                # DON'T shutdown TracerProvider yet - wait until after EventQueue
                # This prevents losing spans that are still being processed
            except Exception as e:
                logger.debug(f"[Exit] Telemetry cleanup error: {e}")
        
        # 2. SECOND: Flush and shutdown EventQueue
        # Now it contains all events from the flushed spans
        if hasattr(client, '_event_queue'):
            try:
                logger.debug("[Exit] Flushing event queue...")
                client._event_queue.force_flush(timeout_seconds=2.0)
                
                # Wait for queue to be completely empty before proceeding
                import time
                max_wait = 5.0  # seconds
                start_time = time.time()
                while not client._event_queue.is_empty():
                    if time.time() - start_time > max_wait:
                        logger.warning("[Exit] EventQueue not empty after timeout")
                        break
                    time.sleep(0.01)  # Small sleep to avoid busy waiting
                
                if client._event_queue.is_empty():
                    logger.debug("[Exit] EventQueue is empty, proceeding with shutdown")
                
                # Clear any stale active sessions (threads may have died without cleanup)
                if hasattr(client, '_active_sessions'):
                    with client._active_sessions_lock:
                        if client._active_sessions:
                            logger.debug(f"[Exit] Clearing {len(client._active_sessions)} remaining active sessions")
                            client._active_sessions.clear()
                
                # Now shutdown EventQueue
                client._event_queue.shutdown()
                logger.debug("[Exit] Event queue shutdown complete")
            except Exception as e:
                logger.debug(f"[Exit] Event queue cleanup error: {e}")
        
        # 3. THIRD: Shutdown TracerProvider after EventQueue is done
        # This ensures all spans can be exported before shutdown
        if hasattr(client, '_tracer_provider') and client._tracer_provider:
            try:
                logger.debug("[Exit] Shutting down TracerProvider...")
                client._tracer_provider.shutdown()
                logger.debug("[Exit] TracerProvider shutdown complete")
            except Exception as e:
                logger.debug(f"[Exit] TracerProvider shutdown error: {e}")
        
        # 4. FOURTH: Close HTTP session ONLY after everything else
        # This prevents broken pipes by ensuring all events are sent first
        if hasattr(client, 'request_session'):
            try:
                # Mark client as shutting down to prevent new requests
                client._shutdown = True
                logger.debug("[Exit] Closing HTTP session (queue empty, worker stopped)")
                client.request_session.close()
                logger.debug("[Exit] HTTP session closed gracefully")
            except Exception as e:
                logger.debug(f"[Exit] HTTP session cleanup error: {e}")
        
        # 5. FINALLY: Clear singletons
        # Safe to destroy now that all data is flushed
        clear_singletons()
        logger.debug("[Exit] Singleton cleanup complete")
        
    except Exception as e:
        # Silent fail on exit to avoid disrupting process termination
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Exit] Cleanup error: {e}")


def _signal_handler(signum, frame):
    """Handle interruption signals with better queue flushing."""
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
    
    # Proper shutdown sequence matching atexit handler
    try:
        client = Client()
        
        # 1. FIRST: Flush OpenTelemetry spans
        if hasattr(client, '_tracer_provider') and client._tracer_provider:
            try:
                logger.debug(f"[Signal] Flushing OpenTelemetry spans on signal {signum}")
                client._tracer_provider.force_flush(timeout_millis=2000)  # Shorter timeout for signals
            except Exception:
                pass
        
        # 2. SECOND: Flush and shutdown EventQueue
        if hasattr(client, "_event_queue"):
            logger.debug(f"[Signal] Flushing event queue on signal {signum}")
            client._event_queue.force_flush(timeout_seconds=2.0)
            
            # Clear active sessions to allow shutdown
            if hasattr(client, '_active_sessions'):
                with client._active_sessions_lock:
                    client._active_sessions.clear()
            
            client._event_queue.shutdown()
        
        # 3. THIRD: Shutdown TracerProvider after EventQueue
        if hasattr(client, '_tracer_provider') and client._tracer_provider:
            logger.debug(f"[Signal] Shutting down TracerProvider on signal {signum}")
            try:
                client._tracer_provider.shutdown()
            except Exception:
                pass
        
        # 4. Mark client as shutting down
        client._shutdown = True
        
    except Exception:
        pass
    
    logger.debug(f"[Signal] Auto-ending session on signal {signum}")
    _auto_end_session()
    # Re-raise the signal for default handling
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Register cleanup functions
atexit.register(_cleanup_singleton_on_exit)  # Clean up singleton resources on exit
atexit.register(_auto_end_session)  # Auto-end session if enabled

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def create_experiment(
    experiment_name: str,
    pass_fail_rubrics: Optional[list] = None,
    score_rubrics: Optional[list] = None,
    description: Optional[str] = None,
    tags: Optional[list] = None,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> str:
    """
    Create a new experiment for grouping and analyzing sessions.
                                                                                                   
    Args:                                                                                      
        experiment_name: Name of the experiment (required)      
        pass_fail_rubrics: List of pass/fail rubric names to associate                        
        description: Description of the experiment                                             
        task: Task description.
        tags: List of tags for categorization                                                  
        score_rubrics: List of score rubric names to associate                                 
        api_key: API key (uses env if not provided)                                            
        agent_id: Agent ID (uses env if not provided)                                          
                                                                                                
    Returns:                                                                                   
        experiment_id: UUID of the created experiment                                          
                                                                                                
    Raises:                                                                                    
        APIKeyVerificationError: If API key is invalid or missing
        InvalidOperationError: If experiment creation fails
        ValueError: If name is empty
    """

    # validation
    if not experiment_name:
        raise ValueError("Experiment name is required")

    if api_key is None:
        api_key = os.getenv("LUCIDIC_API_KEY", None)
        if api_key is None:
            raise APIKeyVerificationError("Make sure to either pass your API key into create_experiment() or set the LUCIDIC_API_KEY environment variable.")
    if agent_id is None:
        agent_id = os.getenv("LUCIDIC_AGENT_ID", None)
        if agent_id is None:
            raise APIKeyVerificationError("Lucidic agent ID not specified. Make sure to either pass your agent ID into create_experiment() or set the LUCIDIC_AGENT_ID environment variable.")

    # combine rubrics into single list
    rubric_names = (pass_fail_rubrics or []) + (score_rubrics or [])

    # get current client which will be NullClient if never lai.init() is never called
    client = Client()
    # if not yet initialized or still the NullClient -> create a real client when init is called
    if not getattr(client, 'initialized', False):
        client = Client(api_key=api_key, agent_id=agent_id)
    else:
        # Already initialized, this is a re-init
        if api_key is not None and agent_id is not None and (api_key != client.api_key or agent_id != client.agent_id):
            client.set_api_key(api_key)
            client.agent_id = agent_id

    # create experiment
    experiment_id = client.create_experiment(experiment_name=experiment_name, rubric_names=rubric_names, description=description, tags=tags)
    logger.info(f"Created experiment with ID: {experiment_id}") 

    return experiment_id


def create_event(
    type: str = "generic",
    **kwargs
) -> str:
    client = Client()
    if not client.session:
        return
    return client.session.create_event(type=type, **kwargs)


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


