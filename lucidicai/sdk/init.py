"""SDK utilities module.

This module provides utility functions for session and context management.
The main entry point for the SDK is the LucidicAI class in client.py.

Note: Global session creation functions (create_session, init) have been removed.
Use the LucidicAI class for all SDK operations.
"""
import asyncio
import threading
from typing import Optional
from weakref import WeakKeyDictionary

from ..utils.logger import debug, truncate_id
from .context import current_session_id

# Module-level storage for async task isolation
_task_sessions: WeakKeyDictionary = WeakKeyDictionary()

# Module-level thread-local storage
_thread_local = threading.local()

# Reference to tracer provider (set by TelemetryManager)
_tracer_provider = None


def set_task_session(session_id: str) -> None:
    """Set session ID for current async task (if in async context)."""
    try:
        if task := asyncio.current_task():
            _task_sessions[task] = session_id
            debug(f"[SDK] Set task-local session {truncate_id(session_id)} for task {task.get_name()}")
    except RuntimeError:
        # Not in async context, ignore
        pass


def clear_task_session() -> None:
    """Clear session ID for current async task (if in async context)."""
    try:
        if task := asyncio.current_task():
            _task_sessions.pop(task, None)
            debug(f"[SDK] Cleared task-local session for task {task.get_name()}")
    except RuntimeError:
        # Not in async context, ignore
        pass


def set_thread_session(session_id: str) -> None:
    """Set session ID for current thread.

    This provides true thread-local storage that doesn't inherit from parent thread.
    """
    _thread_local.session_id = session_id
    current_thread = threading.current_thread()
    debug(f"[SDK] Set thread-local session {truncate_id(session_id)} for thread {current_thread.name}")


def clear_thread_session() -> None:
    """Clear session ID for current thread."""
    if hasattr(_thread_local, 'session_id'):
        delattr(_thread_local, 'session_id')
        current_thread = threading.current_thread()
        debug(f"[SDK] Cleared thread-local session for thread {current_thread.name}")


def get_thread_session() -> Optional[str]:
    """Get session ID from thread-local storage."""
    return getattr(_thread_local, 'session_id', None)


def is_main_thread() -> bool:
    """Check if we're running in the main thread."""
    return threading.current_thread() is threading.main_thread()


def get_session_id() -> Optional[str]:
    """Get the current session ID.

    Priority:
    1. Task-local session (for async tasks)
    2. Thread-local session (for threads) - NO FALLBACK for threads
    3. Context variable session (for main thread)
    """
    # First check task-local storage for async isolation
    try:
        if task := asyncio.current_task():
            if task_session := _task_sessions.get(task):
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

    # For main thread: use context variable
    return current_session_id.get(None)


def get_tracer_provider():
    """Get the tracer provider instance.

    Returns the tracer provider set by the TelemetryManager.
    """
    global _tracer_provider
    if _tracer_provider is not None:
        return _tracer_provider

    # Try to get from TelemetryManager
    try:
        from ..telemetry.telemetry_manager import get_telemetry_manager
        manager = get_telemetry_manager()
        if manager._tracer_provider:
            return manager._tracer_provider
    except Exception:
        pass

    return None


def set_tracer_provider(provider) -> None:
    """Set the tracer provider instance.

    Called by TelemetryManager when initializing telemetry.
    """
    global _tracer_provider
    _tracer_provider = provider
