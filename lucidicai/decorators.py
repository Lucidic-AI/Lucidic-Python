"""Decorators for the Lucidic SDK to create typed, nested events."""
import functools
import inspect
import json
import logging
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
            ev = client.create_event(
                type="function_call",
                function_name=func.__name__,
                arguments={"args": args_dict},
                parent_event_id=parent_id,
                **decorator_kwargs
            )

            with event_context(ev.event_id):
                try:
                    result = func(*args, **kwargs)
                    client.update_event(
                        event_id=ev.event_id,
                        type="function_call",
                        return_value=_serialize(result)
                    )
                    return result
                except Exception as e:
                    client.create_event(
                        type="error_traceback",
                        error=str(e),
                        traceback=''.join(__import__('traceback').format_exc()),
                        parent_event_id=ev.event_id
                    )
                    raise

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
            ev = client.create_event(
                type="function_call",
                function_name=func.__name__,
                arguments={"args": args_dict},
                parent_event_id=parent_id,
                **decorator_kwargs
            )

            async with event_context_async(ev.event_id):
                try:
                    result = await func(*args, **kwargs)
                    client.update_event(
                        event_id=ev.event_id,
                        type="function_call",
                        return_value=_serialize(result)
                    )
                    return result
                except Exception as e:
                    client.create_event(
                        type="error_traceback",
                        error=str(e),
                        traceback=''.join(__import__('traceback').format_exc()),
                        parent_event_id=ev.event_id
                    )
                    raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator