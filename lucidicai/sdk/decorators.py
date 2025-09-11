"""Decorators for the Lucidic SDK to create typed, nested events."""
import functools
import inspect
import json
from datetime import datetime
import uuid
from typing import Any, Callable, Optional, TypeVar
from collections.abc import Iterable

from .event import create_event
from .init import get_session_id
from ..core.errors import LucidicNotInitializedError
from .context import current_parent_event_id, event_context, event_context_async
from ..utils.logger import debug, error as log_error, verbose, truncate_id

F = TypeVar('F', bound=Callable[..., Any])


def _serialize(value: Any):
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [_serialize(v) for v in value]
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


def event(**decorator_kwargs) -> Callable[[F], F]:
    """Universal decorator creating FUNCTION_CALL events with nesting and error capture."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            session_id = get_session_id()
            if not session_id:
                return func(*args, **kwargs)

            # Build arguments snapshot
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            args_dict = {name: _serialize(val) for name, val in bound.arguments.items()}

            parent_id = current_parent_event_id.get(None)
            pre_event_id = str(uuid.uuid4())
            debug(f"[Decorator] Starting {func.__name__} with event ID {truncate_id(pre_event_id)}, parent: {truncate_id(parent_id)}")
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
                            "error_type": type(error).__name__
                        }
                        
                        # Create a separate error_traceback event for the exception
                        import traceback
                        try:
                            create_event(
                                type="error_traceback",
                                error=str(error),
                                traceback=traceback.format_exc(),
                                parent_event_id=pre_event_id  # Parent is the function that threw the error
                            )
                            debug(f"[Decorator] Created error_traceback event for {func.__name__}")
                        except Exception as e:
                            debug(f"[Decorator] Failed to create error_traceback event: {e}")
                    else:
                        return_val = _serialize(result)
                    
                    create_event(
                        type="function_call",
                        event_id=pre_event_id,  # Use the pre-generated ID
                        function_name=func.__name__,
                        arguments=args_dict,
                        return_value=return_val,
                        error=str(error) if error else None,
                        duration=(datetime.now().astimezone() - start_time).total_seconds(),
                        **decorator_kwargs
                    )
                    debug(f"[Decorator] Created function_call event for {func.__name__}")
                except Exception as e:
                    log_error(f"[Decorator] Failed to create function_call event: {e}")

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            session_id = get_session_id()
            if not session_id:
                return await func(*args, **kwargs)

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            args_dict = {name: _serialize(val) for name, val in bound.arguments.items()}

            parent_id = current_parent_event_id.get(None)
            pre_event_id = str(uuid.uuid4())
            debug(f"[Decorator] Starting {func.__name__} with event ID {truncate_id(pre_event_id)}, parent: {truncate_id(parent_id)}")
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
                            "error_type": type(error).__name__
                        }
                        
                        # Create a separate error_traceback event for the exception
                        import traceback
                        try:
                            create_event(
                                type="error_traceback",
                                error=str(error),
                                traceback=traceback.format_exc(),
                                parent_event_id=pre_event_id  # Parent is the function that threw the error
                            )
                            debug(f"[Decorator] Created error_traceback event for {func.__name__}")
                        except Exception as e:
                            debug(f"[Decorator] Failed to create error_traceback event: {e}")
                    else:
                        return_val = _serialize(result)
                    
                    create_event(
                        type="function_call",
                        event_id=pre_event_id,  # Use the pre-generated ID
                        function_name=func.__name__,
                        arguments=args_dict,
                        return_value=return_val,
                        error=str(error) if error else None,
                        duration=(datetime.now().astimezone() - start_time).total_seconds(),
                        **decorator_kwargs
                    )
                    debug(f"[Decorator] Created function_call event for {func.__name__}")
                except Exception as e:
                    log_error(f"[Decorator] Failed to create function_call event: {e}")

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator