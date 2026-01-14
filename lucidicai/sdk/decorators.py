"""Decorators for the Lucidic SDK to create typed, nested events.

Supports both client-bound and context-based event tracking:
- Client-bound: Pass a LucidicAI client instance to bind events to that client
- Context-based: Uses global session context when no client is specified
"""
import functools
import inspect
import traceback
from datetime import datetime
import uuid
from typing import Any, Callable, Optional, TypeVar, TYPE_CHECKING

from .context import (
    current_session_id,
    current_client,
    current_parent_event_id,
    event_context,
    event_context_async,
)
from .init import get_session_id
from ..utils.serialization import serialize_value
from ..utils.logger import debug, error as log_error, truncate_id

if TYPE_CHECKING:
    from ..client import LucidicAI

F = TypeVar("F", bound=Callable[..., Any])


def _emit_event_to_client(
    client: "LucidicAI",
    session_id: str,
    type: str,
    **event_data,
) -> Optional[str]:
    """Emit an event using the client's event resource.

    Args:
        client: The LucidicAI client
        session_id: The session ID to associate the event with
        type: The event type (e.g., "function_call", "error_traceback")
        **event_data: Additional event data

    Returns:
        The event ID if created successfully, None otherwise
    """
    try:
        event_payload = {
            "type": type,
            "session_id": session_id,
            **event_data,
        }
        response = client._resources["events"].create(**event_payload)
        return response.get("event_id") if response else None
    except Exception as e:
        debug(f"[Decorator] Failed to emit event: {e}")
        return None


async def _aemit_event_to_client(
    client: "LucidicAI",
    session_id: str,
    type: str,
    **event_data,
) -> Optional[str]:
    """Emit an event using the client's event resource (async).

    Args:
        client: The LucidicAI client
        session_id: The session ID to associate the event with
        type: The event type (e.g., "function_call", "error_traceback")
        **event_data: Additional event data

    Returns:
        The event ID if created successfully, None otherwise
    """
    try:
        event_payload = {
            "type": type,
            "session_id": session_id,
            **event_data,
        }
        response = await client._resources["events"].acreate(**event_payload)
        return response.get("event_id") if response else None
    except Exception as e:
        debug(f"[Decorator] Failed to emit async event: {e}")
        return None


def event(
    client: Optional["LucidicAI"] = None, **decorator_kwargs
) -> Callable[[F], F]:
    """Universal decorator creating FUNCTION_CALL events with nesting and error capture.

    Supports both client-bound and context-based modes:
    - Client-bound: Pass a client instance to bind events to that specific client
    - Context-based: Omit client to use global session context

    Args:
        client: Optional LucidicAI client to bind this decorator to
        **decorator_kwargs: Additional keyword arguments passed to the event

    Returns:
        A decorator function that wraps the target function

    Example:
        # Context-based (uses global session)
        @event()
        def my_function():
            pass

        # Client-bound
        @event(client=my_client)
        def my_function():
            pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Determine session ID and whether we should track
            if client:
                # Client-bound mode: check if this client is active
                active_client = current_client.get(None)
                if active_client is not client:
                    # Not our client - just execute the function
                    return func(*args, **kwargs)
                session_id = current_session_id.get(None)
            else:
                # Context-based mode: use global session
                session_id = get_session_id()

            if not session_id:
                return func(*args, **kwargs)

            # Build arguments snapshot
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            args_dict = {
                name: serialize_value(val) for name, val in bound.arguments.items()
            }

            parent_id = current_parent_event_id.get(None)
            pre_event_id = str(uuid.uuid4())
            debug(
                f"[Decorator] Starting {func.__name__} with event ID {truncate_id(pre_event_id)}, parent: {truncate_id(parent_id)}"
            )
            start_time = datetime.now().astimezone()
            result = None
            error: Optional[BaseException] = None

            try:
                with event_context(pre_event_id):
                    # Also inject into OpenTelemetry context for instrumentors
                    from ..telemetry.context_bridge import inject_lucidic_context
                    from opentelemetry import context as otel_context

                    otel_ctx = inject_lucidic_context()
                    token = otel_context.attach(otel_ctx)
                    try:
                        result = func(*args, **kwargs)
                    finally:
                        otel_context.detach(token)
                return result
            except Exception as e:
                error = e
                log_error(f"[Decorator] {func.__name__} raised exception: {e}")
                raise
            finally:
                try:
                    # Store error as return value with type information
                    if error:
                        return_val = {
                            "error": str(error),
                            "error_type": type(error).__name__,
                        }

                        # Create a separate error_traceback event for the exception
                        try:
                            if client:
                                _emit_event_to_client(
                                    client,
                                    session_id,
                                    type="error_traceback",
                                    error=str(error),
                                    traceback=traceback.format_exc(),
                                    parent_event_id=pre_event_id,
                                )
                            else:
                                from .event import emit_event

                                emit_event(
                                    type="error_traceback",
                                    error=str(error),
                                    traceback=traceback.format_exc(),
                                    parent_event_id=pre_event_id,
                                )
                            debug(
                                f"[Decorator] Created error_traceback event for {func.__name__}"
                            )
                        except Exception as e:
                            debug(
                                f"[Decorator] Failed to create error_traceback event: {e}"
                            )
                    else:
                        return_val = serialize_value(result)

                    # Emit the function_call event
                    if client:
                        _emit_event_to_client(
                            client,
                            session_id,
                            type="function_call",
                            event_id=pre_event_id,
                            parent_event_id=parent_id,
                            function_name=func.__name__,
                            arguments=args_dict,
                            return_value=return_val,
                            error=str(error) if error else None,
                            duration=(
                                datetime.now().astimezone() - start_time
                            ).total_seconds(),
                            **decorator_kwargs,
                        )
                    else:
                        from .event import emit_event

                        emit_event(
                            type="function_call",
                            event_id=pre_event_id,
                            parent_event_id=parent_id,
                            function_name=func.__name__,
                            arguments=args_dict,
                            return_value=return_val,
                            error=str(error) if error else None,
                            duration=(
                                datetime.now().astimezone() - start_time
                            ).total_seconds(),
                            **decorator_kwargs,
                        )
                    debug(f"[Decorator] Created function_call event for {func.__name__}")
                except Exception as e:
                    log_error(f"[Decorator] Failed to create function_call event: {e}")

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Determine session ID and whether we should track
            if client:
                # Client-bound mode: check if this client is active
                active_client = current_client.get(None)
                if active_client is not client:
                    # Not our client - just execute the function
                    return await func(*args, **kwargs)
                session_id = current_session_id.get(None)
            else:
                # Context-based mode: use global session
                session_id = get_session_id()

            if not session_id:
                return await func(*args, **kwargs)

            # Build arguments snapshot
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            args_dict = {
                name: serialize_value(val) for name, val in bound.arguments.items()
            }

            parent_id = current_parent_event_id.get(None)
            pre_event_id = str(uuid.uuid4())
            debug(
                f"[Decorator] Starting {func.__name__} with event ID {truncate_id(pre_event_id)}, parent: {truncate_id(parent_id)}"
            )
            start_time = datetime.now().astimezone()
            result = None
            error: Optional[BaseException] = None

            try:
                async with event_context_async(pre_event_id):
                    # Also inject into OpenTelemetry context for instrumentors
                    from ..telemetry.context_bridge import inject_lucidic_context
                    from opentelemetry import context as otel_context

                    otel_ctx = inject_lucidic_context()
                    token = otel_context.attach(otel_ctx)
                    try:
                        result = await func(*args, **kwargs)
                    finally:
                        otel_context.detach(token)
                return result
            except Exception as e:
                error = e
                log_error(f"[Decorator] {func.__name__} raised exception: {e}")
                raise
            finally:
                try:
                    # Store error as return value with type information
                    if error:
                        return_val = {
                            "error": str(error),
                            "error_type": type(error).__name__,
                        }

                        # Create a separate error_traceback event for the exception
                        try:
                            if client:
                                await _aemit_event_to_client(
                                    client,
                                    session_id,
                                    type="error_traceback",
                                    error=str(error),
                                    traceback=traceback.format_exc(),
                                    parent_event_id=pre_event_id,
                                )
                            else:
                                from .event import emit_event

                                emit_event(
                                    type="error_traceback",
                                    error=str(error),
                                    traceback=traceback.format_exc(),
                                    parent_event_id=pre_event_id,
                                )
                            debug(
                                f"[Decorator] Created error_traceback event for {func.__name__}"
                            )
                        except Exception as e:
                            debug(
                                f"[Decorator] Failed to create error_traceback event: {e}"
                            )
                    else:
                        return_val = serialize_value(result)

                    # Emit the function_call event
                    if client:
                        await _aemit_event_to_client(
                            client,
                            session_id,
                            type="function_call",
                            event_id=pre_event_id,
                            parent_event_id=parent_id,
                            function_name=func.__name__,
                            arguments=args_dict,
                            return_value=return_val,
                            error=str(error) if error else None,
                            duration=(
                                datetime.now().astimezone() - start_time
                            ).total_seconds(),
                            **decorator_kwargs,
                        )
                    else:
                        from .event import emit_event

                        emit_event(
                            type="function_call",
                            event_id=pre_event_id,
                            parent_event_id=parent_id,
                            function_name=func.__name__,
                            arguments=args_dict,
                            return_value=return_val,
                            error=str(error) if error else None,
                            duration=(
                                datetime.now().astimezone() - start_time
                            ).total_seconds(),
                            **decorator_kwargs,
                        )
                    debug(f"[Decorator] Created function_call event for {func.__name__}")
                except Exception as e:
                    log_error(f"[Decorator] Failed to create function_call event: {e}")

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


# Backward compatibility alias
def create_bound_event_decorator(
    client: "LucidicAI", **decorator_kwargs
) -> Callable[[F], F]:
    """Create an event decorator bound to a specific client.

    This is a convenience wrapper around event() for backward compatibility.

    Args:
        client: The LucidicAI client to bind this decorator to
        **decorator_kwargs: Additional keyword arguments passed to the event

    Returns:
        A decorator function that wraps the target function
    """
    return event(client=client, **decorator_kwargs)
