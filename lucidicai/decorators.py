"""Decorators for the Lucidic SDK to create typed events (linear for decorators)."""
import functools
import inspect
import json
import logging
from datetime import datetime
from typing import Any, Callable, Optional, TypeVar
from collections.abc import Iterable

from .client import Client
from .errors import LucidicNotInitializedError

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
    """Universal decorator creating FUNCTION_CALL events at the end (no updates, no nesting)."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                client = Client()
                if not client.session:
                    return func(*args, **kwargs)
            except (LucidicNotInitializedError, AttributeError):
                return func(*args, **kwargs)

            start_time = datetime.now().astimezone()
            # Build arguments snapshot
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            args_dict = {name: _serialize(val) for name, val in bound.arguments.items()}

            try:
                result = func(*args, **kwargs)
                client.create_event(
                    type="function_call",
                    function_name=func.__name__,
                    arguments={"args": args_dict},
                    return_value=_serialize(result),
                    occurred_at=start_time,
                    duration=(datetime.now().astimezone() - start_time).total_seconds(),
                    **decorator_kwargs
                )
                return result
            except Exception as e:
                import traceback as _tb
                client.create_event(
                    type="error_traceback",
                    error=str(e),
                    traceback=''.join(_tb.format_exc()),
                    occurred_at=start_time,
                    duration=(datetime.now().astimezone() - start_time).total_seconds(),
                    **decorator_kwargs
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

            start_time = datetime.now().astimezone()
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            args_dict = {name: _serialize(val) for name, val in bound.arguments.items()}

            try:
                result = await func(*args, **kwargs)
                client.create_event(
                    type="function_call",
                    function_name=func.__name__,
                    arguments={"args": args_dict},
                    return_value=_serialize(result),
                    occurred_at=start_time,
                    duration=(datetime.now().astimezone() - start_time).total_seconds(),
                    **decorator_kwargs
                )
                return result
            except Exception as e:
                import traceback as _tb
                client.create_event(
                    type="error_traceback",
                    error=str(e),
                    traceback=''.join(_tb.format_exc()),
                    occurred_at=start_time,
                    duration=(datetime.now().astimezone() - start_time).total_seconds(),
                    **decorator_kwargs
                )
                raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator