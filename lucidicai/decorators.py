"""Decorators for the Lucidic SDK to create typed, nested events."""
import functools
import inspect
import json
import logging
from datetime import datetime
import uuid
from typing import Any, Callable, Optional, TypeVar
from collections.abc import Iterable

from .client import Client
from .errors import LucidicNotInitializedError
from .context import current_parent_event_id, event_context, event_context_async

logger = logging.getLogger("Lucidic")

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
            try:
                client = Client()
                if not client.session:
                    return func(*args, **kwargs)
            except (LucidicNotInitializedError, AttributeError):
                return func(*args, **kwargs)

            # Build arguments snapshot
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            args_dict = {name: _serialize(val) for name, val in bound.arguments.items()}

            parent_id = current_parent_event_id.get(None)
            pre_event_id = str(uuid.uuid4())
            start_time = datetime.now().astimezone()
            result = None
            error: Optional[BaseException] = None

            try:
                with event_context(pre_event_id):
                    result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = e
                raise
            finally:
                try:
                    # Store error as return value with type information
                    if error:
                        return_val = {
                            "error": str(error),
                            "error_type": type(error).__name__
                        }
                    else:
                        return_val = _serialize(result)
                    
                    client.create_event(
                        type="function_call",
                        event_id=pre_event_id,
                        function_name=func.__name__,
                        arguments=args_dict,
                        return_value=return_val,
                        error=str(error) if error else None,
                        parent_event_id=parent_id,
                        duration=(datetime.now().astimezone() - start_time).total_seconds(),
                        **decorator_kwargs
                    )
                except Exception:
                    pass

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                client = Client()
                if not client.session:
                    return await func(*args, **kwargs)
            except (LucidicNotInitializedError, AttributeError):
                return await func(*args, **kwargs)

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            args_dict = {name: _serialize(val) for name, val in bound.arguments.items()}

            parent_id = current_parent_event_id.get(None)
            pre_event_id = str(uuid.uuid4())
            start_time = datetime.now().astimezone()
            result = None
            error: Optional[BaseException] = None

            try:
                async with event_context_async(pre_event_id):
                    result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error = e
                raise
            finally:
                try:
                    # Store error as return value with type information
                    if error:
                        return_val = {
                            "error": str(error),
                            "error_type": type(error).__name__
                        }
                    else:
                        return_val = _serialize(result)
                    
                    client.create_event(
                        type="function_call",
                        event_id=pre_event_id,
                        function_name=func.__name__,
                        arguments=args_dict,
                        return_value=return_val,
                        error=str(error) if error else None,
                        parent_event_id=parent_id,
                        duration=(datetime.now().astimezone() - start_time).total_seconds(),
                        **decorator_kwargs
                    )
                except Exception:
                    pass

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator