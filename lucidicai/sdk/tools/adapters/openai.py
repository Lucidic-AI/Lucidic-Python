"""OpenAI function-calling adapter (LUC-579).

Two surfaces:

1. ``register_openai_tools(tools)`` — walks the ``tools=`` list passed
   to ``openai.chat.completions.create``. Each ``{"type": "function",
   "function": {...}}`` entry becomes a ``ToolSurface``. Non-function
   entries (``code_interpreter``, ``file_search``, etc.) are skipped
   silently. Reuses the shared ``_params_from_json_schema`` helper
   from ``registry.py`` so the on-wire signature shape is identical
   to ``@mockable`` + Anthropic + LangChain adapters.

2. ``dispatch_openai_tool_call(call, impls)`` (+ async sibling) — the
   user-side dispatch helper. OpenAI hands us JSON tool calls; the
   actual Python implementations live in user code (``impls`` dict).
   When a ``MockContext`` is bound, route through the transport
   (PASS_THROUGH / drift → fall back to ``impls[name]``). Otherwise
   invoke ``impls[name]`` directly.

The asymmetry with LangChain is structural — OpenAI's "tool" is a
JSON Schema, not a Python callable we can wrap. The user holds the
implementations and must invoke our dispatch helper at every dispatch
site.

Wire shape (frozen by the OpenAI client SDK):

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": str,
                "description": str,
                "parameters": <JSON Schema>,
            },
        },
        ...
    ]

    response.choices[0].message.tool_calls[i] has:
        .id: str
        .function.name: str
        .function.arguments: str (JSON-encoded kwargs)

We duck-type call extraction (attribute access or dict subscript) so
this adapter works with both the official Pydantic-typed response
objects and hand-rolled dict shapes used in tests.
"""
import hashlib
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ....core.errors import LucidicNotInitializedError
from ...context import get_active_client
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


def register_openai_tools(
    tools: List[Dict[str, Any]],
    *,
    client: Optional["LucidicAI"] = None,
) -> List[ToolSurface]:
    """Register OpenAI function-calling tools for mockable dispatch.

    Walks the ``tools=`` argument typically passed to
    ``openai.chat.completions.create``. Each ``{"type": "function",
    "function": {...}}`` entry becomes a ``ToolSurface`` registered via
    the shared registry (LUC-577a) — either directly into the given
    client's registry, or via the module-level buffer if no client
    exists yet (the next ``LucidicAI(...)`` drains it).

    Non-function entries (``code_interpreter``, ``file_search``,
    ``web_search``, etc.) are silently skipped — they're not mockable
    surfaces, just OpenAI-managed tools the model can invoke without
    going through user code.

    Re-registration is idempotent in the last-wins sense — same name
    overwrites the prior surface (matches ``@mockable`` semantics).

    Returns the list of ``ToolSurface`` instances that were registered
    (useful for tests + REPL inspection; user code can ignore).

    Args:
        tools: The list passed to ``chat.completions.create(tools=...)``.
        client: Optional explicit ``LucidicAI`` instance. When None,
            registration uses the active client from the contextvar
            (set by the most recent ``LucidicAI(...)`` construction).
            Pass explicitly in multi-client processes where the active
            context may not point at the right one.
    """
    surfaces: List[ToolSurface] = []
    for entry in tools:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "function":
            continue
        spec = entry.get("function")
        if not isinstance(spec, dict):
            continue
        surface = _surface_from_openai_function(spec)
        surfaces.append(surface)
        _register_into(surface, client)
    logger.debug(
        "[OpenAI adapter] registered %d/%d function tool(s)",
        len(surfaces), len(tools),
    )
    return surfaces


def _surface_from_openai_function(spec: Dict[str, Any]) -> ToolSurface:
    """Build a ``ToolSurface`` from one OpenAI function spec.

    ``source_hash`` is the canonical JSON encoding of the spec (sorted
    keys, no whitespace). Means drift detection responds to ANY change
    in the spec — name, description, parameters schema. The user's
    Python impl is invisible to us at registration time, so impl-level
    drift goes undetected (documented limitation; the ``@mockable``
    decorator path catches both).
    """
    name = spec["name"]
    docstring = spec.get("description", "") or ""
    params = _params_from_json_schema(spec.get("parameters", {}) or {})
    return ToolSurface(
        name=name,
        signature={"params": params, "return_type": None},
        docstring=docstring,
        return_shape=None,
        source_hash=_compute_openai_source_hash(spec),
    )


def _compute_openai_source_hash(spec: Dict[str, Any]) -> str:
    """SHA256 over the spec's canonical JSON form. Sort keys at every
    level. Always 64-char lowercase hex (matches the backend's
    expected shape via ``SyncAgentToolsSerializer``).
    """
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _register_into(
    surface: ToolSurface, client: Optional["LucidicAI"],
) -> None:
    """Route ``register_tool`` through an explicit client or the active
    contextvar.

    If ``client`` is given, write directly into ``client.tools._registry``
    — explicit binding wins over context detection (matches the
    "instance-canonical, module passthrough" choice from LUC-608).
    Otherwise fall through to the shared ``register_tool`` which uses
    ``get_active_client()`` to find the right registry (or buffers).
    """
    if client is not None and hasattr(client, "tools"):
        client.tools._registry[surface.name] = surface  # noqa: SLF001
        return
    register_tool(surface)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch_openai_tool_call(
    call: Any,
    impls: Dict[str, Callable[..., Any]],
    *,
    client: Optional["LucidicAI"] = None,
    client_event_id: Optional[str] = None,
) -> Any:
    """Dispatch one OpenAI tool_call.

    Two paths, decided by the ``MockContext`` bound to the current
    execution context (set when a tool-backed session is created —
    LUC-608):

    - **No mock context** (normal observability / no session):
      invoke ``impls[name](**kwargs)`` directly. Missing ``impls[name]``
      raises ``KeyError`` — the user forgot to wire the implementation.

    - **Mock context active**: route through ``emit_call_through_backend``
      from the transport layer. PASS_THROUGH and drift fall back to
      ``impls.get(name)`` (None is acceptable; the transport raises
      ``LucidicMissingImplError`` if needed).

    Args:
        call: An OpenAI ``ChatCompletionMessageToolCall`` (or any
            object / dict with ``.function.name`` + ``.function.arguments``).
            ``arguments`` is a JSON string per the OpenAI spec; we
            parse it here.
        impls: Mapping from tool name → real Python implementation.
            The user holds these; OpenAI's tools= ships only JSON
            Schema. Required even in mock context — fallback paths
            (PASS_THROUGH, drift) need it.
        client: Optional explicit ``LucidicAI``. Defaults to the
            contextvar-bound active client; raises
            ``LucidicNotInitializedError`` if neither is available
            AND a mock context is set (a dispatch request without
            either is an unrecoverable config error).
        client_event_id: Backend idempotency key. Auto-generated as a
            fresh UUID per call when omitted.

    Returns:
        Whatever the resolved invocation returned — backend's
        ``return_value`` for mocked calls, ``impls[name]``'s return
        for the local paths.
    """
    name, kwargs = _extract_call(call)
    ctx = _current_mock_context()

    if ctx is None:
        # No mock context bound. Run the local impl directly — missing
        # impl is a natural KeyError that surfaces the user's bug.
        return impls[name](**kwargs)

    # Mock context active. Resolve the client for transport.
    resolved_client = client or ctx.client
    if resolved_client is None:
        raise LucidicNotInitializedError()

    real_fn = impls.get(name)
    return emit_call_through_backend(
        client=resolved_client,
        session_id=ctx.session_id,
        tool_name=name,
        args=(),
        kwargs=kwargs,
        real_fn=real_fn,
        client_event_id=client_event_id or str(uuid.uuid4()),
    )


async def adispatch_openai_tool_call(
    call: Any,
    impls: Dict[str, Callable[..., Any]],
    *,
    client: Optional["LucidicAI"] = None,
    client_event_id: Optional[str] = None,
) -> Any:
    """Async sibling of ``dispatch_openai_tool_call``.

    Same behavior matrix; awaits the async transport
    (``aemit_call_through_backend``) and the async impl. The user's
    ``impls[name]`` is expected to be an async callable when the
    adapter is invoked via this path.
    """
    name, kwargs = _extract_call(call)
    ctx = _current_mock_context()

    if ctx is None:
        return await impls[name](**kwargs)

    resolved_client = client or ctx.client
    if resolved_client is None:
        raise LucidicNotInitializedError()

    real_fn = impls.get(name)
    return await aemit_call_through_backend(
        client=resolved_client,
        session_id=ctx.session_id,
        tool_name=name,
        args=(),
        kwargs=kwargs,
        real_fn=real_fn,
        client_event_id=client_event_id or str(uuid.uuid4()),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _extract_call(call: Any) -> Tuple[str, Dict[str, Any]]:
    """Get ``(name, kwargs)`` from a Pydantic ``ChatCompletionMessageToolCall``
    OR a hand-rolled dict.

    Duck-typed so tests can construct fake calls with
    ``SimpleNamespace`` and real client code can hand us the SDK's
    typed objects. ``arguments`` is a JSON string per the OpenAI spec
    — parsed here. Empty / malformed arguments default to ``{}``
    rather than raising; backend will return ``invalid_kwargs`` if
    the missing context matters.
    """
    if hasattr(call, "function"):
        fn = call.function
        name = fn.name if hasattr(fn, "name") else fn["name"]
        raw_args = fn.arguments if hasattr(fn, "arguments") else fn["arguments"]
    elif isinstance(call, dict):
        fn = call["function"]
        name = fn["name"]
        raw_args = fn["arguments"]
    else:
        raise TypeError(
            f"dispatch_openai_tool_call: expected ChatCompletionMessageToolCall "
            f"or dict, got {type(call).__name__}"
        )

    if not raw_args:
        return name, {}
    try:
        parsed = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "[OpenAI adapter] tool_call %r has un-parseable arguments %r; "
            "passing empty kwargs",
            name, raw_args,
        )
        return name, {}
    if not isinstance(parsed, dict):
        logger.warning(
            "[OpenAI adapter] tool_call %r arguments parsed to %s (expected "
            "dict); passing empty kwargs",
            name, type(parsed).__name__,
        )
        return name, {}
    return name, parsed
