"""Async-safe and thread-safe context helpers for session (and step, extensible).

This module exposes context variables and helpers to bind a Lucidic
session to the current execution context (threads/async tasks), so
OpenTelemetry spans can be deterministically attributed to the correct
session under concurrency.
"""

from contextlib import contextmanager, asynccontextmanager
import contextvars
from typing import Optional, Iterator, AsyncIterator, Callable, Any, Dict, TYPE_CHECKING
import logging
import os
import threading

if TYPE_CHECKING:
    from ..client import LucidicAI


# Context variable for the active Lucidic session id
current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "lucidic.session_id", default=None
)


# Context variable for parent event nesting
current_parent_event_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "lucidic.parent_event_id", default=None
)


# Context variable for the active client (for multi-client support)
current_client: contextvars.ContextVar[Optional["LucidicAI"]] = contextvars.ContextVar(
    "lucidic.client", default=None
)


def get_active_client() -> Optional["LucidicAI"]:
    """Get the currently active LucidicAI client from context.

    Returns:
        The active client, or None if no client is bound to the current context.
    """
    return current_client.get(None)


def set_active_session(session_id: Optional[str]) -> None:
    """Bind the given session id to the current execution context.

    Sets both contextvar and thread-local storage when in a thread.
    """
    from .init import set_thread_session, is_main_thread

    current_session_id.set(session_id)

    # Also set thread-local storage if we're in a non-main thread
    if session_id and not is_main_thread():
        set_thread_session(session_id)


def clear_active_session() -> None:
    """Clear any active session binding in the current execution context.

    Clears both contextvar and thread-local storage when in a thread.
    """
    from .init import clear_thread_session, is_main_thread

    current_session_id.set(None)

    # Also clear thread-local storage if we're in a non-main thread
    if not is_main_thread():
        clear_thread_session()


@contextmanager
def bind_session(session_id: str) -> Iterator[None]:
    """Context manager to temporarily bind an active session id.

    Handles both thread-local and context variable storage for proper isolation.
    """
    from .init import set_thread_session, clear_thread_session, is_main_thread

    token = current_session_id.set(session_id)

    # If we're in a non-main thread, also set thread-local storage
    thread_local_set = False
    if not is_main_thread():
        set_thread_session(session_id)
        thread_local_set = True

    try:
        yield
    finally:
        if thread_local_set:
            clear_thread_session()
        current_session_id.reset(token)


@asynccontextmanager
async def bind_session_async(session_id: str) -> AsyncIterator[None]:
    """Async context manager to temporarily bind an active session id."""
    from .init import set_task_session, clear_task_session

    token = current_session_id.set(session_id)

    # Also set task-local for async isolation
    set_task_session(session_id)

    try:
        yield
    finally:
        clear_task_session()
        current_session_id.reset(token)


# NEW: Parent event context managers
@contextmanager
def event_context(event_id: str) -> Iterator[None]:
    token = current_parent_event_id.set(event_id)
    try:
        yield
    finally:
        current_parent_event_id.reset(token)


@asynccontextmanager
async def event_context_async(event_id: str) -> AsyncIterator[None]:
    token = current_parent_event_id.set(event_id)
    try:
        yield
    finally:
        current_parent_event_id.reset(token)


@contextmanager
def session(**init_params) -> Iterator[None]:
    """All-in-one context manager: init → bind → yield → clear → end.

    Notes:
    - Ignores any provided auto_end parameter and ends the session on context exit.
    - If LUCIDIC_DEBUG is true, logs a warning about ignoring auto_end.
    - Handles thread-local storage for proper thread isolation.
    """
    # Lazy import to avoid circular imports
    import lucidicai as lai  # type: ignore
    from .init import set_thread_session, clear_thread_session, is_main_thread

    # Force auto_end to False inside a context manager to control explicit end
    user_auto_end = init_params.get('auto_end', None)
    init_params = dict(init_params)
    init_params['auto_end'] = False

    if os.getenv('LUCIDIC_DEBUG', 'False') == 'True' and user_auto_end is not None:
        logging.getLogger('Lucidic').warning('session(...) ignores auto_end and will end the session at context exit')

    session_id = lai.init(**init_params)
    token = current_session_id.set(session_id)

    # If we're in a non-main thread, also set thread-local storage
    thread_local_set = False
    if not is_main_thread():
        set_thread_session(session_id)
        thread_local_set = True

    try:
        yield
    finally:
        if thread_local_set:
            clear_thread_session()
        current_session_id.reset(token)
        try:
            # Force flush OpenTelemetry spans before ending session
            from .init import get_tracer_provider
            from ..utils.logger import debug, info
            import time

            tracer_provider = get_tracer_provider()
            if tracer_provider:
                debug(f"[Session] Force flushing OpenTelemetry spans for session {session_id}")
                try:
                    # Force flush with 5 second timeout to ensure all spans are exported
                    flush_result = tracer_provider.force_flush(timeout_millis=5000)
                    debug(f"[Session] Tracer provider force_flush returned: {flush_result}")

                    # Give a small additional delay to ensure the exporter processes everything
                    # This is necessary because force_flush on the provider flushes the processors,
                    # but the exporter might still be processing the spans
                    time.sleep(0.5)
                    debug(f"[Session] Successfully flushed spans for session {session_id}")
                except Exception as e:
                    debug(f"[Session] Error flushing spans: {e}")

            # Pass session_id explicitly to avoid context issues
            lai.end_session(session_id=session_id)
        except Exception:
            # Avoid masking the original exception from the with-block
            pass


@asynccontextmanager
async def session_async(**init_params) -> AsyncIterator[None]:
    """Async counterpart of session(...)."""
    import lucidicai as lai  # type: ignore
    from .init import set_task_session, clear_task_session

    user_auto_end = init_params.get('auto_end', None)
    init_params = dict(init_params)
    init_params['auto_end'] = False

    if os.getenv('LUCIDIC_DEBUG', 'False') == 'True' and user_auto_end is not None:
        logging.getLogger('Lucidic').warning('session_async(...) ignores auto_end and will end the session at context exit')

    session_id = lai.init(**init_params)
    token = current_session_id.set(session_id)

    # Set task-local session for true isolation in async
    set_task_session(session_id)

    try:
        yield
    finally:
        # Clear task-local session first
        clear_task_session()
        current_session_id.reset(token)
        try:
            # Force flush OpenTelemetry spans before ending session
            from .init import get_tracer_provider
            from ..utils.logger import debug, info
            import asyncio

            tracer_provider = get_tracer_provider()
            if tracer_provider:
                debug(f"[Session] Force flushing OpenTelemetry spans for async session {session_id}")
                try:
                    # Force flush with 5 second timeout to ensure all spans are exported
                    flush_result = tracer_provider.force_flush(timeout_millis=5000)
                    debug(f"[Session] Tracer provider force_flush returned: {flush_result}")

                    # Give a small additional delay to ensure the exporter processes everything
                    # This is necessary because force_flush on the provider flushes the processors,
                    # but the exporter might still be processing the spans
                    await asyncio.sleep(0.5)
                    debug(f"[Session] Successfully flushed spans for async session {session_id}")
                except Exception as e:
                    debug(f"[Session] Error flushing spans: {e}")

            # Pass session_id explicitly to avoid context issues in async
            lai.end_session(session_id=session_id)
        except Exception:
            pass


def run_session(fn: Callable[..., Any], *fn_args: Any, init_params: Optional[Dict[str, Any]] = None, **fn_kwargs: Any) -> Any:
    """Run a callable within a full Lucidic session lifecycle context."""
    with session(**(init_params or {})):
        return fn(*fn_args, **fn_kwargs)


def run_in_session(session_id: str, fn: Callable[..., Any], *fn_args: Any, **fn_kwargs: Any) -> Any:
    """Run a callable with a bound session id. Does not end the session."""
    with bind_session(session_id):
        return fn(*fn_args, **fn_kwargs)


def thread_worker_with_session(session_id: str, target: Callable[..., Any], *args, **kwargs) -> Any:
    """Wrapper for thread worker functions that ensures proper session isolation.

    Use this as the target function for threads to ensure each thread gets
    its own session context without bleeding from the parent thread.

    Example:
        thread = Thread(
            target=thread_worker_with_session,
            args=(session_id, actual_worker_function, arg1, arg2),
            kwargs={'key': 'value'}
        )
    """
    from .init import set_thread_session, clear_thread_session

    # Set thread-local session immediately
    set_thread_session(session_id)

    try:
        # Also bind to contextvar for compatibility
        with bind_session(session_id):
            return target(*args, **kwargs)
    finally:
        # Clean up thread-local storage
        clear_thread_session()


