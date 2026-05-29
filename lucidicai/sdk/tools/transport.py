"""Transport layer for ``@mockable`` + framework-adapter dispatch.

Sits between the user-facing decorators / adapters and the
``MockCallResource`` HTTP layer. Adds the policy that the raw resource
doesn't know about:

- **PASS_THROUGH fallback**: backend returns ``was_mocked=false`` when
  the Tool's tier is PASS_THROUGH. The transport runs the user's local
  function (``real_fn``) and returns its result, so the user gets real
  behavior for tools the dashboard isn't mocking. If no ``real_fn`` was
  provided (e.g. an adapter where ``impls[name]`` is missing), we raise
  ``LucidicMissingImplError`` — a programming bug that should surface.

- **Drift fallback** (the load-bearing UX contract): backend returns
  409 ``tool_drift`` when the Tool's ``source_hash`` doesn't match the
  session's ``tool_version_snapshot``. The transport logs WARNING and
  runs ``real_fn`` instead — never blocks the user's session on an SDK-
  side reproducibility concern. Future sessions get the new snapshot
  automatically via LUC-608's auto-sync.

- **Network error wrapping**: ``httpx.RequestError`` (connect, timeout,
  read errors that fail before a response exists) get wrapped as
  ``LucidicMockCallError(code="network_error")`` so callers see the
  uniform mock_call error family.

All other typed errors (``LucidicToolBlockedError``,
``LucidicUnsupportedSQLError``, ``LucidicUnknownToolError``, etc.)
propagate to the caller. They're real failures the user needs to see —
no silent fallback would hide them in any useful way.

Positional args note: the v1 transport accepts ``args`` for API
symmetry with the wrapper (which gets called as ``wrapper(*args, **kwargs)``)
but ships them as ``kwargs["__args__"]``. The backend's tier executors
(LUC-582, LUC-583) only consume named params from ``kwargs``, so
positional args don't reach the executor. ``@mockable`` users should
prefer kwargs at call sites; LangChain / OpenAI / Anthropic adapters
already invoke with kwargs natively.
"""
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import httpx

from ...core.errors import (
    LucidicMissingImplError,
    LucidicMockCallError,
    LucidicToolDriftError,
)
from .call_log import record_call

if TYPE_CHECKING:
    from ...client import LucidicAI


logger = logging.getLogger("Lucidic")


def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _short_hash(value: Optional[str]) -> str:
    return (value[:12] + "..") if value else "<none>"


# Dashboard URL to surface in drift warnings. Kept generic — the
# org-specific path lives in the dashboard's tool detail view and is
# easier to reach via the "Tools" tab on the agent than via a deep link.
_DASHBOARD_TOOLS_HINT = "See the agent's Tools tab in the Lucidic dashboard to acknowledge drift."


def _effective_kwargs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Fold positional args into kwargs under the ``__args__`` synthetic key.

    Backend executors don't read ``__args__`` in v1 (see module docstring),
    but the field is reserved so adding positional-arg support later is a
    non-breaking change. The wire payload size impact is negligible.
    """
    if not args:
        return dict(kwargs)
    return {"__args__": list(args), **kwargs}


def _log_drift_warning(
    tool_name: str,
    session_id: str,
    exc: LucidicToolDriftError,
) -> None:
    """Single-line WARNING with both hashes + remediation hint.

    Per-call (not rate-limited) in v1 — drift on a hot tool will spam
    logs, but spam is the correct signal that the user should ack via
    the dashboard. v1.5 follow-up: rate-limit per (session_id, tool_name)
    to one log per session.
    """
    session_hash = (exc.session_hash[:12] + "...") if exc.session_hash else "<none>"
    current_hash = (exc.current_hash[:12] + "...") if exc.current_hash else "<none>"
    sid_short = (session_id[:8] + "...") if len(session_id) > 8 else session_id
    logger.warning(
        "[mock_call] tool_drift tool=%s session=%s session_hash=%s "
        "current_hash=%s — falling back to local impl. %s",
        tool_name, sid_short, session_hash, current_hash, _DASHBOARD_TOOLS_HINT,
    )


def _run_fallback_sync(
    real_fn: Optional[Callable[..., Any]],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    tool_name: str,
    reason: str,
) -> Any:
    """Run the local fallback (or raise ``LucidicMissingImplError``).

    ``reason`` is "tool_drift" or "tier=PASS_THROUGH" or similar — used
    in the exception message so the user knows *why* the SDK needed a
    fallback they didn't provide.
    """
    if real_fn is None:
        raise LucidicMissingImplError(tool_name=tool_name, reason=reason)
    return real_fn(*args, **kwargs)


async def _run_fallback_async(
    real_fn: Optional[Callable[..., Any]],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    tool_name: str,
    reason: str,
) -> Any:
    """Async version of ``_run_fallback_sync``.

    Awaits ``real_fn`` since the async transport is only entered from
    async wrappers — by construction ``real_fn`` is an async callable
    (or None). If a caller violates this contract by passing a sync
    callable, we still ``await`` whatever it returns; for plain values
    Python raises ``TypeError`` from ``await`` which surfaces the bug.
    """
    if real_fn is None:
        raise LucidicMissingImplError(tool_name=tool_name, reason=reason)
    return await real_fn(*args, **kwargs)


def emit_call_through_backend(
    *,
    client: "LucidicAI",
    session_id: str,
    tool_name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    real_fn: Optional[Callable[..., Any]] = None,
    client_event_id: Optional[str] = None,
) -> Any:
    """Route one synchronous tool call through ``POST /sdk/mock-call``.

    Behavior matrix:

    +---------------------------------+----------------------------------------+
    | Backend response                | Transport behavior                     |
    +=================================+========================================+
    | 200, ``was_mocked=True``        | return ``body["return_value"]``        |
    +---------------------------------+----------------------------------------+
    | 200, ``was_mocked=False``       | run ``real_fn(*args, **kwargs)`` —     |
    | (PASS_THROUGH)                  | raise ``LucidicMissingImplError``      |
    |                                 | if ``real_fn`` is None                 |
    +---------------------------------+----------------------------------------+
    | 409 ``tool_drift``              | WARNING + run ``real_fn`` (or raise    |
    |                                 | ``LucidicMissingImplError``). Never    |
    |                                 | re-raises the drift error.             |
    +---------------------------------+----------------------------------------+
    | Other 4xx / 5xx                 | re-raise the typed ``LucidicMockCallError`` |
    |                                 | subclass                               |
    +---------------------------------+----------------------------------------+
    | ``httpx.RequestError``          | raise ``LucidicMockCallError(code=     |
    | (network)                       | "network_error", detail=str(exc))``    |
    +---------------------------------+----------------------------------------+

    ``real_fn`` is the synchronous callable to fall back to. The
    ``@mockable`` decorator passes the wrapped function; framework
    adapters pass ``impls.get(name)``. None is allowed (the adapter
    user may have decorated the tool spec without supplying an impl);
    fallback paths then raise ``LucidicMissingImplError``.
    """
    payload_kwargs = _effective_kwargs(args, kwargs)
    resource = client._resources["mock_calls"]  # type: ignore[index]

    t0 = time.perf_counter()
    try:
        body = resource.call(
            session_id=session_id,
            tool_name=tool_name,
            kwargs=payload_kwargs,
            client_event_id=client_event_id,
        )
    except LucidicToolDriftError as exc:
        _log_drift_warning(tool_name, session_id, exc)
        record_call(
            session_id=session_id, tool_name=tool_name, kwargs=payload_kwargs,
            outcome="DRIFT->local", elapsed_ms=_elapsed_ms(t0),
            extra=f"session_hash={_short_hash(exc.session_hash)} "
                  f"current_hash={_short_hash(exc.current_hash)}",
        )
        return _run_fallback_sync(real_fn, args, kwargs, tool_name, "tool_drift")
    except httpx.RequestError as exc:
        # Connect/timeout/read errors that fail before a response exists.
        # Wrap so callers see the uniform mock_call error family.
        record_call(
            session_id=session_id, tool_name=tool_name, kwargs=payload_kwargs,
            outcome="ERROR", error=f"network_error: {type(exc).__name__}: {exc}",
            elapsed_ms=_elapsed_ms(t0),
        )
        raise LucidicMockCallError(
            code="network_error",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    except LucidicMockCallError as exc:
        # Typed backend failures (impl_error, tool_blocked, unsupported_sql,
        # unknown_tool, session_not_initialized, ...) propagate to the caller —
        # but record them first so the mock log shows the failure in real time.
        record_call(
            session_id=session_id, tool_name=tool_name, kwargs=payload_kwargs,
            outcome="ERROR", error=f"{exc.code}: {exc.detail}",
            elapsed_ms=_elapsed_ms(t0),
        )
        raise

    if body.get("was_mocked", True):
        rv = body.get("return_value")
        record_call(
            session_id=session_id, tool_name=tool_name, kwargs=payload_kwargs,
            outcome="MOCKED", tier=body.get("tier"), result=rv, has_result=True,
            elapsed_ms=_elapsed_ms(t0),
        )
        return rv
    # PASS_THROUGH — backend declined to mock; run the real function.
    tier = body.get("tier", "?")
    record_call(
        session_id=session_id, tool_name=tool_name, kwargs=payload_kwargs,
        outcome="PASS_THROUGH", tier=tier, extra="(ran real fn)",
        elapsed_ms=_elapsed_ms(t0),
    )
    return _run_fallback_sync(real_fn, args, kwargs, tool_name, f"tier={tier}")


async def aemit_call_through_backend(
    *,
    client: "LucidicAI",
    session_id: str,
    tool_name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    real_fn: Optional[Callable[..., Any]] = None,
    client_event_id: Optional[str] = None,
) -> Any:
    """Async sibling of ``emit_call_through_backend``.

    Same behavior matrix; awaits the async resource path
    (``MockCallResource.acall``) and the async fallback runner. ``real_fn``
    is expected to be an async callable when provided (the async transport
    is only entered from async wrappers, by construction).
    """
    payload_kwargs = _effective_kwargs(args, kwargs)
    resource = client._resources["mock_calls"]  # type: ignore[index]

    t0 = time.perf_counter()
    try:
        body = await resource.acall(
            session_id=session_id,
            tool_name=tool_name,
            kwargs=payload_kwargs,
            client_event_id=client_event_id,
        )
    except LucidicToolDriftError as exc:
        _log_drift_warning(tool_name, session_id, exc)
        record_call(
            session_id=session_id, tool_name=tool_name, kwargs=payload_kwargs,
            outcome="DRIFT->local", elapsed_ms=_elapsed_ms(t0),
            extra=f"session_hash={_short_hash(exc.session_hash)} "
                  f"current_hash={_short_hash(exc.current_hash)}",
        )
        return await _run_fallback_async(real_fn, args, kwargs, tool_name, "tool_drift")
    except httpx.RequestError as exc:
        record_call(
            session_id=session_id, tool_name=tool_name, kwargs=payload_kwargs,
            outcome="ERROR", error=f"network_error: {type(exc).__name__}: {exc}",
            elapsed_ms=_elapsed_ms(t0),
        )
        raise LucidicMockCallError(
            code="network_error",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    except LucidicMockCallError as exc:
        record_call(
            session_id=session_id, tool_name=tool_name, kwargs=payload_kwargs,
            outcome="ERROR", error=f"{exc.code}: {exc.detail}",
            elapsed_ms=_elapsed_ms(t0),
        )
        raise

    if body.get("was_mocked", True):
        rv = body.get("return_value")
        record_call(
            session_id=session_id, tool_name=tool_name, kwargs=payload_kwargs,
            outcome="MOCKED", tier=body.get("tier"), result=rv, has_result=True,
            elapsed_ms=_elapsed_ms(t0),
        )
        return rv
    tier = body.get("tier", "?")
    record_call(
        session_id=session_id, tool_name=tool_name, kwargs=payload_kwargs,
        outcome="PASS_THROUGH", tier=tier, extra="(ran real fn)",
        elapsed_ms=_elapsed_ms(t0),
    )
    return await _run_fallback_async(real_fn, args, kwargs, tool_name, f"tier={tier}")
