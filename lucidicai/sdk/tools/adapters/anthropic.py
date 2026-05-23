"""Anthropic tool-blocks adapter (LUC-580).

Near-clone of the OpenAI adapter (LUC-579) with two structural
differences:

1. **Flat tool spec**: Anthropic's ``tools=`` entries are
   ``{name, description, input_schema}`` at the top level — no
   ``{type: function, function: {...}}`` nesting like OpenAI. We
   accept that flat shape directly.

2. **Parsed ``input`` dict**: Anthropic's ``ToolUseBlock.input`` is
   already a Python dict (unlike OpenAI's ``ChatCompletionMessageToolCall
   .function.arguments`` which is a JSON-encoded string). No
   ``json.loads`` needed.

Two surfaces, matching LUC-579 conventions:

- ``register_anthropic_tools(tools)`` — walks ``tools=`` from
  ``messages.create``, registers each into the active client's
  registry (or buffers if no client). Same last-wins semantics.

- ``dispatch_anthropic_tool_call(block, impls)`` (+ async sibling)
  — user-side dispatch. With a mock context: route through transport.
  Without: invoke ``impls[name]`` directly.

Source hash strategy is identical to LUC-579: sha256 over canonical
JSON of the spec. Same documented limitation that impl-level drift
(user's Python impl changing) is invisible to spec-level drift
detection.

Wire shape (frozen by the Anthropic client SDK):

    TOOLS = [
        {
            "name": str,
            "description": str,
            "input_schema": <JSON Schema>,
        },
        ...
    ]

    message.content[i] when block.type == "tool_use":
        .id: str (e.g. "toolu_01ABC")
        .name: str
        .input: dict (already parsed)
"""
import hashlib
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

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


def register_anthropic_tools(
    tools: List[Dict[str, Any]],
    *,
    client: Optional["LucidicAI"] = None,
) -> List[ToolSurface]:
    """Register Anthropic tools for mockable dispatch.

    Walks the ``tools=`` argument typically passed to
    ``anthropic.Anthropic().messages.create``. Each entry becomes a
    ``ToolSurface`` registered via the shared registry (LUC-577a) —
    either directly into the given client's registry, or via the
    module-level buffer if no client exists yet.

    Skips malformed entries (non-dict, missing ``name``) silently —
    matches the lenient handling in the OpenAI adapter so a single
    bad entry doesn't break registration of valid ones.

    Returns the list of registered ``ToolSurface`` instances.

    Args:
        tools: The list passed to ``messages.create(tools=...)``.
        client: Optional explicit ``LucidicAI`` instance. When None,
            registration uses the active client from the contextvar.
            Pass explicitly in multi-client processes.
    """
    surfaces: List[ToolSurface] = []
    for spec in tools:
        if not isinstance(spec, dict):
            continue
        if not spec.get("name"):
            continue
        surface = _surface_from_anthropic_tool(spec)
        surfaces.append(surface)
        _register_into(surface, client)
    logger.debug(
        "[Anthropic adapter] registered %d/%d tool(s)",
        len(surfaces), len(tools),
    )
    return surfaces


def _surface_from_anthropic_tool(spec: Dict[str, Any]) -> ToolSurface:
    """Build a ``ToolSurface`` from one Anthropic tool spec.

    ``source_hash`` is the canonical JSON encoding of the spec —
    matches the LUC-579 strategy. Drift detection responds to any
    spec change (name, description, input_schema). Impl-level drift
    in the user's Python is invisible (documented limitation).
    """
    name = spec["name"]
    docstring = spec.get("description", "") or ""
    params = _params_from_json_schema(spec.get("input_schema", {}) or {})
    return ToolSurface(
        name=name,
        signature={"params": params, "return_type": None},
        docstring=docstring,
        return_shape=None,
        source_hash=_compute_anthropic_source_hash(spec),
    )


def _compute_anthropic_source_hash(spec: Dict[str, Any]) -> str:
    """SHA256 over the spec's canonical JSON form. Sort keys at every
    level for determinism."""
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _register_into(
    surface: ToolSurface, client: Optional["LucidicAI"],
) -> None:
    """Route ``register_tool`` through an explicit client or the active
    contextvar — mirrors the OpenAI adapter's logic."""
    if client is not None and hasattr(client, "tools"):
        client.tools._registry[surface.name] = surface  # noqa: SLF001
        return
    register_tool(surface)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch_anthropic_tool_call(
    block: Any,
    impls: Dict[str, Callable[..., Any]],
    *,
    client: Optional["LucidicAI"] = None,
    client_event_id: Optional[str] = None,
) -> Any:
    """Dispatch one Anthropic ``tool_use`` block.

    Two paths, decided by the ``MockContext`` bound to the current
    execution context (LUC-608):

    - **No mock context** (normal observability / no session):
      invoke ``impls[name](**input)`` directly. Missing impl raises
      a natural ``KeyError``.

    - **Mock context active**: route through ``emit_call_through_backend``
      from the transport layer. PASS_THROUGH and drift fall back to
      ``impls.get(name)``.

    Args:
        block: An Anthropic ``ToolUseBlock`` (or any object / dict
            with ``.name`` + ``.input``). ``input`` is already a dict
            in the official SDK; we don't parse JSON.
        impls: Mapping from tool name → real implementation. Required
            even in mock context for fallback paths.
        client: Optional explicit ``LucidicAI``. Defaults to the
            contextvar-bound active client.
        client_event_id: Backend idempotency key. Auto-generated UUID
            when omitted.
    """
    name, kwargs = _extract_block(block)
    ctx = _current_mock_context()

    if ctx is None:
        return impls[name](**kwargs)

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


async def adispatch_anthropic_tool_call(
    block: Any,
    impls: Dict[str, Callable[..., Any]],
    *,
    client: Optional["LucidicAI"] = None,
    client_event_id: Optional[str] = None,
) -> Any:
    """Async sibling of ``dispatch_anthropic_tool_call``.

    Awaits the async transport (``aemit_call_through_backend``) and
    the async impl. ``impls[name]`` is expected to be an async
    callable when invoked via this path.
    """
    name, kwargs = _extract_block(block)
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


def _extract_block(block: Any) -> Tuple[str, Dict[str, Any]]:
    """Get ``(name, input_dict)`` from a Pydantic ``ToolUseBlock`` OR
    a hand-rolled dict.

    Anthropic's block has fields at the top level (no ``.function``
    nesting like OpenAI). ``.input`` is already a parsed dict — no
    JSON parsing needed.

    Duck-typed so tests can use ``SimpleNamespace`` fakes and real
    client code can hand us the SDK's typed objects.
    """
    if isinstance(block, dict):
        name = block.get("name")
        raw_input = block.get("input")
    else:
        name = getattr(block, "name", None)
        raw_input = getattr(block, "input", None)

    if not name or not isinstance(name, str):
        raise TypeError(
            f"dispatch_anthropic_tool_call: block must have a 'name' "
            f"string field; got {type(block).__name__}"
        )

    if raw_input is None:
        return name, {}
    if isinstance(raw_input, dict):
        return name, raw_input
    # Defensive: if the SDK changes to ship `input` as a JSON string
    # (some experimental Anthropic flows do this for streaming), try
    # to parse. Fall back to empty kwargs on parse failure.
    if isinstance(raw_input, str):
        try:
            parsed = json.loads(raw_input)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "[Anthropic adapter] tool_use %r has un-parseable input "
                "string %r; passing empty kwargs",
                name, raw_input,
            )
            return name, {}
        if isinstance(parsed, dict):
            return name, parsed

    logger.warning(
        "[Anthropic adapter] tool_use %r input is %s (expected dict); "
        "passing empty kwargs",
        name, type(raw_input).__name__,
    )
    return name, {}
