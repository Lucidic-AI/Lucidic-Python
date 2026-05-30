"""MCP (Model Context Protocol) adapter.

Routes MCP tool calls through ``/sdk/mock-call`` without requiring
``@mockable`` decoration. MCP tools come from a remote server (JSON-RPC
over HTTP/stdio), so the SDK can't wrap them at definition time the way
``@mockable`` does for in-process Python functions. The consumer
(e.g. an MCP client wrapper) calls this adapter at every tool dispatch
site.

Two surfaces:

1. ``register_mcp_tools(tools_list)`` — walks the ``tools/list`` JSON-RPC
   response (or any list of ``{name, description?, inputSchema?,
   outputSchema?}`` dicts). Each entry becomes a ``ToolSurface`` with
   params from the inputSchema (reuses ``_params_from_json_schema``).
   ``source_hash`` is computed over the canonical JSON of the full spec,
   so any change in name, description, or schema drifts the surface.

2. ``dispatch_mcp_tool_call(tool_name, arguments, real_fn)`` (+ async
   sibling) — the per-call dispatch helper. When a ``MockContext`` is
   bound, route through the transport (PASS_THROUGH / drift → fall back
   to ``real_fn()``). Otherwise invoke ``real_fn()`` directly. The
   consumer holds the real HTTP-POST path as a closure / bound method
   and hands it to us as ``real_fn``.

Differs from the openai adapter: no ``impls`` dict. OpenAI hands you N
tool calls at once and you dispatch each through an ``impls[name]``
lookup; MCP dispatch sites already have one specific tool name and one
existing real-call path (the MCP client's HTTP layer). Passing
``real_fn`` directly avoids the lookup, matches MCP's per-call dispatch
shape, and lets the consumer keep its existing call-chain state
(session id, headers, etc.) captured in the closure.
"""
import hashlib
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from ....core.errors import LucidicNotInitializedError
from ..context import _current_mock_context
from ..registry import (
    ToolSurface,
    _params_from_json_schema,
    register_tool,
)
from ..transport import (
    aemit_call_through_backend,
    emit_call_through_backend,
)

if TYPE_CHECKING:
    from ....client import LucidicAI


logger = logging.getLogger("Lucidic")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_mcp_tools(
    tools_list: List[Dict[str, Any]],
    *,
    client: Optional["LucidicAI"] = None,
) -> List[ToolSurface]:
    """Register MCP tool surfaces from a ``tools/list`` response.

    Accepts the ``tools`` array from an MCP ``tools/list`` JSON-RPC
    result — each entry is ``{name, description?, inputSchema?,
    outputSchema?}``. Entries missing a ``name`` are skipped silently.

    Re-registration is idempotent in the last-wins sense — same name
    overwrites the prior surface (matches ``@mockable`` and the openai
    adapter).

    Returns the list of registered ``ToolSurface`` instances (handy for
    tests / REPL inspection; consumers can ignore the return).

    Args:
        tools_list: The ``tools`` array from ``mcp.list_tools()`` or the
            equivalent for whatever MCP client is in use. Bare list,
            not the full JSON-RPC envelope.
        client: Optional explicit ``LucidicAI`` instance. When None,
            registration uses the active client from the contextvar.
            Pass explicitly in multi-client processes where the active
            context may not point at the right one.
    """
    surfaces: List[ToolSurface] = []
    for entry in tools_list:
        if not isinstance(entry, dict):
            continue
        if not entry.get("name"):
            continue
        surface = _surface_from_mcp_tool(entry)
        surfaces.append(surface)
        _register_into(surface, client)
    logger.debug(
        "[MCP adapter] registered %d/%d tool(s)",
        len(surfaces), len(tools_list),
    )
    return surfaces


def _surface_from_mcp_tool(spec: Dict[str, Any]) -> ToolSurface:
    """Build a ``ToolSurface`` from one MCP tool spec.

    MCP's ``inputSchema`` is a JSON Schema object — flatten via the
    shared helper so we emit the same shape as @mockable / openai /
    anthropic. ``outputSchema`` (MCP 2025+) lands in ``return_shape``
    when present; older servers omit it and we leave it null.
    """
    name = spec["name"]
    docstring = spec.get("description", "") or ""
    input_schema = spec.get("inputSchema") or {}
    params = _params_from_json_schema(input_schema if isinstance(input_schema, dict) else {})
    output_schema = spec.get("outputSchema")
    return_shape = output_schema if isinstance(output_schema, dict) else None
    return ToolSurface(
        name=name,
        signature={"params": params, "return_type": None},
        docstring=docstring,
        return_shape=return_shape,
        source_hash=_compute_mcp_source_hash(spec),
    )


def _compute_mcp_source_hash(spec: Dict[str, Any]) -> str:
    """SHA256 over the spec's canonical JSON form. Sort keys at every
    level. Always 64-char lowercase hex (matches the backend's expected
    shape via ``SyncAgentToolsSerializer``).

    Hashing the whole spec — name + description + inputSchema +
    outputSchema — means ANY server-side surface change drifts. The
    real Python impl (whatever the MCP server runs on its end) is
    invisible to us; impl-level drift is documented-as-undetected, same
    as the openai adapter.
    """
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _register_into(
    surface: ToolSurface, client: Optional["LucidicAI"],
) -> None:
    """Route ``register_tool`` through an explicit client or the active
    contextvar.

    Same pattern as the openai adapter — explicit binding wins over
    context detection, otherwise fall through to the shared
    ``register_tool`` which uses ``get_active_client()``.
    """
    if client is not None and hasattr(client, "tools"):
        client.tools._registry[surface.name] = surface  # noqa: SLF001
        return
    register_tool(surface)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch_mcp_tool_call(
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    *,
    real_fn: Callable[..., Any],
    client: Optional["LucidicAI"] = None,
    client_event_id: Optional[str] = None,
) -> Any:
    """Dispatch one MCP tool call.

    Two paths, decided by the ``MockContext`` bound to the current
    execution context:

    - **No mock context** (normal observability / no session):
      invoke ``real_fn()`` directly. The consumer's existing call chain
      runs unchanged.

    - **Mock context active**: route through ``emit_call_through_backend``.
      PASS_THROUGH and drift fall back to ``real_fn()`` automatically.

    Args:
        tool_name: The MCP tool name being dispatched.
        arguments: Keyword arguments for the tool. None and missing
            both serialize as an empty dict to the backend.
        real_fn: Callable with the consumer's existing real-call path
            (typically a closure capturing the MCP client + headers +
            session state). Invoked with NO arguments — bake everything
            into the closure. Must NOT recurse back into this adapter,
            or PASS_THROUGH will loop.
        client: Optional explicit ``LucidicAI``. Defaults to the
            mock-context's bound client; raises
            ``LucidicNotInitializedError`` if neither is available AND
            a mock context is set.
        client_event_id: Backend idempotency key. Auto-generated as a
            fresh UUID per call when omitted.

    Returns:
        Whatever the resolved invocation returned — backend's
        ``return_value`` for mocked calls, ``real_fn()``'s return for
        the local paths.
    """
    kwargs = dict(arguments) if arguments else {}
    ctx = _current_mock_context()

    if ctx is None:
        # No mock context bound. Run the real path directly.
        return real_fn()

    resolved_client = client or ctx.client
    if resolved_client is None:
        raise LucidicNotInitializedError()

    return emit_call_through_backend(
        client=resolved_client,
        session_id=ctx.session_id,
        tool_name=tool_name,
        args=(),
        kwargs=kwargs,
        real_fn=_zero_arg_to_kwargs_sync(real_fn),
        client_event_id=client_event_id or str(uuid.uuid4()),
    )


async def adispatch_mcp_tool_call(
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    *,
    real_fn: Callable[..., Awaitable[Any]],
    client: Optional["LucidicAI"] = None,
    client_event_id: Optional[str] = None,
) -> Any:
    """Async sibling of ``dispatch_mcp_tool_call``.

    Same behavior matrix; awaits the async transport. ``real_fn`` is
    expected to be an async callable (the MCP client is async by spec).
    """
    kwargs = dict(arguments) if arguments else {}
    ctx = _current_mock_context()

    if ctx is None:
        return await real_fn()

    resolved_client = client or ctx.client
    if resolved_client is None:
        raise LucidicNotInitializedError()

    return await aemit_call_through_backend(
        client=resolved_client,
        session_id=ctx.session_id,
        tool_name=tool_name,
        args=(),
        kwargs=kwargs,
        real_fn=_zero_arg_to_kwargs_async(real_fn),
        client_event_id=client_event_id or str(uuid.uuid4()),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _zero_arg_to_kwargs_sync(
    real_fn: Callable[..., Any],
) -> Callable[..., Any]:
    """Adapt a zero-arg ``real_fn`` to the transport's ``real_fn(*args, **kwargs)``
    contract.

    The transport calls ``real_fn(*args, **kwargs)`` because @mockable
    wraps a function whose signature mirrors the tool's. MCP consumers
    hand us a closure that already has the args baked in (since the MCP
    client's HTTP path takes name + args as positional, not kwargs),
    so we ignore whatever the transport forwards.
    """
    def _adapter(*_args: Any, **_kwargs: Any) -> Any:
        return real_fn()
    return _adapter


def _zero_arg_to_kwargs_async(
    real_fn: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Async sibling of ``_zero_arg_to_kwargs_sync``."""
    async def _adapter(*_args: Any, **_kwargs: Any) -> Any:
        return await real_fn()
    return _adapter
