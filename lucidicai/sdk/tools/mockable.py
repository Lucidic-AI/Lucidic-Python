"""``@mockable`` decorator — capture a tool's surface and route dispatch.

Decoration captures the function's signature, docstring, and source-hash
into a ``ToolSurface``, registers it (via ``register_tool``), and returns
a wrapper that consults ``_current_mock_context()`` at call time.

- **No mock context** (normal observability sessions or no session at
  all): wrapper runs the original function — one contextvar lookup,
  zero overhead beyond that.
- **Mock context active** (session was bound via
  ``ToolsResource._init_session`` after a tool-backed session start):
  wrapper routes the call through the transport layer
  (``emit_call_through_backend`` / ``aemit_call_through_backend``)
  which handles tier dispatch, PASS_THROUGH fallback, and drift.

Sync + async are detected at decoration time via
``inspect.iscoroutinefunction``. Both wrappers attach the captured
``ToolSurface`` as ``__lucidic_surface__`` so test code and framework
adapters can introspect what was captured without going through the
registry.

Lambdas are rejected outright (``func.__name__ == "<lambda>"`` is not a
valid identifier) — the backend's ``Tool.clean`` requires identifier-shaped
names, and an early SDK-side reject produces a much better error.
"""
import functools
import inspect
import uuid
from typing import Any, Callable, TypeVar

from ...core.errors import LucidicError
from .context import _current_mock_context
from .registry import (
    ToolSurface,
    _compute_source_hash,
    _params_from_callable,
    _stringify_annotation,
    register_tool,
)
from .transport import aemit_call_through_backend, emit_call_through_backend


F = TypeVar("F", bound=Callable[..., Any])


def _capture_surface(func: Callable) -> ToolSurface:
    """Build the ``ToolSurface`` for a Python callable.

    Pulls the signature via ``inspect.signature``, the docstring via
    ``inspect.getdoc``, the source body via ``inspect.getsource`` (so
    drift detection responds to behavioral changes), and folds them
    through the shared helpers in ``registry.py``.

    Raises ``LucidicError`` if the callable's name isn't a valid Python
    identifier — the backend's ``Tool.clean`` enforces the same rule and
    rejects on sync with a less helpful error.
    """
    name = getattr(func, "__name__", "")
    if not name or not name.isidentifier():
        raise LucidicError(
            f"@mockable requires a function with a valid Python identifier "
            f"name (got {name!r}); decorate a `def` function, not a lambda"
        )

    try:
        sig = inspect.signature(func)
        return_type = _stringify_annotation(sig.return_annotation)
    except (TypeError, ValueError):
        return_type = None

    signature = {
        "params": _params_from_callable(func),
        "return_type": return_type,
    }
    docstring = inspect.getdoc(func) or ""

    # getsource can fail on builtins, C extensions, lambdas (rejected above),
    # interactively-defined functions, etc. Treat as no-source and emit a
    # hash over the signature only — drift detection still works on
    # signature changes, just not body-only changes.
    try:
        body_source = inspect.getsource(func)
    except (OSError, TypeError):
        body_source = ""

    return ToolSurface(
        name=name,
        signature=signature,
        docstring=docstring,
        return_shape=None,  # v1.5 — Pydantic/dataclass/TypedDict return-shape inference
        source_hash=_compute_source_hash(
            name=name,
            signature=signature,
            body_source=body_source,
        ),
    )


def mockable(func: F) -> F:
    """Mark ``func`` as a mockable tool.

    Captures the surface at decoration time (raises ``LucidicError`` on
    invalid input) and returns a wrapper that — in the v1 of this ticket
    — runs the original function unchanged. The dispatch-intercept layer
    is wired into this wrapper by LUC-577c once the
    ``_current_mock_context()`` machinery exists.

    Sniffs ``async def`` via ``inspect.iscoroutinefunction`` and returns
    the matching wrapper. Attaches ``__lucidic_surface__: ToolSurface``
    to the wrapper for introspection.

    Example::

        @mockable
        def query_unread_emails(sender: str, limit: int = 50) -> list[dict]:
            ...

        query_unread_emails.__lucidic_surface__  # → ToolSurface(name="query_unread_emails", ...)
    """
    surface = _capture_surface(func)
    register_tool(surface)

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = _current_mock_context()
            if ctx is None:
                # Fast path: no mock context bound to this task. One
                # contextvar lookup, then run the user's function
                # unchanged. Hit every tool call in production sessions
                # that aren't part of a tool-backed eval run.
                return await func(*args, **kwargs)
            return await aemit_call_through_backend(
                client=ctx.client,
                session_id=ctx.session_id,
                tool_name=surface.name,
                args=args,
                kwargs=kwargs,
                real_fn=func,
                client_event_id=str(uuid.uuid4()),
            )

        awrapper.__lucidic_surface__ = surface  # type: ignore[attr-defined]
        return awrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = _current_mock_context()
        if ctx is None:
            return func(*args, **kwargs)
        return emit_call_through_backend(
            client=ctx.client,
            session_id=ctx.session_id,
            tool_name=surface.name,
            args=args,
            kwargs=kwargs,
            real_fn=func,
            client_event_id=str(uuid.uuid4()),
        )

    wrapper.__lucidic_surface__ = surface  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
