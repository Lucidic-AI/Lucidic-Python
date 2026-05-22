"""Mock-context binding — the contextvar that says "this call should be mocked".

``MockContext`` carries the two things a ``@mockable`` wrapper needs to
route a call through the backend: the bound ``LucidicAI`` client (the
``mock_calls`` resource lives there) and the session id (mock_call
dispatches per session, with drift-checked against the session's
``tool_version_snapshot``).

Set by ``ToolsResource._init_session`` (LUC-608) when a tool-backed
session is created and ``/sdk/session-init-fixtures`` reports
``initialized=True``. Cleared on session end. ``@mockable`` wrappers
read via ``_current_mock_context()`` on every call — a single
contextvar lookup, no allocation.

Concurrency notes:

- Async tasks each see their own copy (``ContextVar`` semantics).
  Setting in task A doesn't affect task B. Matches how the existing
  ``current_session_id`` is bound — see ``lucidicai/sdk/context.py``.
- Nested sessions clobber: ``bind_mock_context(ctx_inner)`` while one
  is already set replaces the outer. Same pattern as
  ``set_active_session``. ``Token``-based reset is supported for
  callers that need it (the session lifecycle uses this).
"""
import contextvars
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...client import LucidicAI


@dataclass(frozen=True)
class MockContext:
    """Bound state telling ``@mockable`` to route through the backend.

    ``client`` is the parent ``LucidicAI`` — transport pulls
    ``client._resources["mock_calls"]`` to issue the dispatch.
    ``session_id`` is the resolved id from ``client.sessions.create()``;
    backend uses it to look up the per-session ``tool_version_snapshot``
    for drift checking.
    """

    session_id: str
    client: "LucidicAI"


# Module-level contextvar — values automatically scoped per task in
# async, per OS thread in sync (until a child thread runs without
# inheriting). Default None means "no mocking active in this context".
current_mock_context: contextvars.ContextVar[Optional[MockContext]] = contextvars.ContextVar(
    "lucidic.mock_context", default=None,
)


def bind_mock_context(ctx: MockContext) -> contextvars.Token:
    """Set the active mock context for the current execution context.

    Returns a ``Token`` so callers can ``current_mock_context.reset(token)``
    later — used by the session lifecycle to undo the binding when the
    session ends. Most callers can ignore the return value if they
    don't need explicit unbinding (process-end cleanup handles it).
    """
    return current_mock_context.set(ctx)


def clear_mock_context() -> None:
    """Unconditionally clear the bound mock context.

    Sets to ``None`` rather than reset via Token — used by session-end
    paths where the original Token may be lost (cross-thread, REPL).
    Subsequent ``_current_mock_context()`` calls see None until the
    next ``bind_mock_context``.
    """
    current_mock_context.set(None)


def _current_mock_context() -> Optional[MockContext]:
    """Read the active mock context, or None when no mocking is bound.

    Underscore-prefix because this is consumed by ``@mockable`` /
    adapters internally, not part of the public user surface. Public
    callers should use ``client.tools`` methods or the decorator.
    """
    return current_mock_context.get(None)
