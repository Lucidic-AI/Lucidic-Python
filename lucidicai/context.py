"""Async-safe context helpers for session (and step, extensible).

This module exposes context variables and helpers to bind a Lucidic
session to the current execution context (threads/async tasks), so
OpenTelemetry spans can be deterministically attributed to the correct
session under concurrency.
"""

from contextlib import contextmanager, asynccontextmanager
import contextvars
from typing import Optional, Iterator, AsyncIterator, Callable, Any, Dict
import logging
import os


# Context variable for the active Lucidic session id
current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "lucidic.session_id", default=None
)


# NEW: Context variable for parent event nesting
current_parent_event_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "lucidic.parent_event_id", default=None
)


def set_active_session(session_id: Optional[str]) -> None:
    """Bind the given session id to the current execution context."""
    current_session_id.set(session_id)


def clear_active_session() -> None:
    """Clear any active session binding in the current execution context."""
    current_session_id.set(None)


@contextmanager
def bind_session(session_id: str) -> Iterator[None]:
    """Context manager to temporarily bind an active session id."""
    token = current_session_id.set(session_id)
    try:
        yield
    finally:
        current_session_id.reset(token)


@asynccontextmanager
async def bind_session_async(session_id: str) -> AsyncIterator[None]:
    """Async context manager to temporarily bind an active session id."""
    token = current_session_id.set(session_id)
    try:
        yield
    finally:
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
    """
    # Lazy import to avoid circular imports
    import lucidicai as lai  # type: ignore

    # Force auto_end to False inside a context manager to control explicit end
    user_auto_end = init_params.get('auto_end', None)
    init_params = dict(init_params)
    init_params['auto_end'] = False

    if os.getenv('LUCIDIC_DEBUG', 'False') == 'True' and user_auto_end is not None:
        logging.getLogger('Lucidic').warning('session(...) ignores auto_end and will end the session at context exit')

    session_id = lai.init(**init_params)
    token = current_session_id.set(session_id)
    try:
        yield
    finally:
        current_session_id.reset(token)
        try:
            lai.end_session()
        except Exception:
            # Avoid masking the original exception from the with-block
            pass


@asynccontextmanager
async def session_async(**init_params) -> AsyncIterator[None]:
    """Async counterpart of session(...)."""
    import lucidicai as lai  # type: ignore

    user_auto_end = init_params.get('auto_end', None)
    init_params = dict(init_params)
    init_params['auto_end'] = False

    if os.getenv('LUCIDIC_DEBUG', 'False') == 'True' and user_auto_end is not None:
        logging.getLogger('Lucidic').warning('session_async(...) ignores auto_end and will end the session at context exit')

    session_id = lai.init(**init_params)
    token = current_session_id.set(session_id)
    try:
        yield
    finally:
        current_session_id.reset(token)
        try:
            lai.end_session()
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


