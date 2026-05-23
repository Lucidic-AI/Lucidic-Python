"""LangChain framework adapter (LUC-578).

Structurally different from the OpenAI / Anthropic adapters because
LangChain hands us **Python callables** (``BaseTool.func``), not JSON
specs. We can transparently replace the callable with a mockable
wrapper — the framework's dispatch loop (``agent.invoke``,
``tool.invoke``, ``tool._run``) calls our wrapper instead of the
user's original function. No user-side ``dispatch_*`` helper needed.

Single surface:

- ``register_langchain_tools(tools)`` — walks a list of
  ``BaseTool`` instances. For each: extracts the tool surface from
  ``args_schema`` (Pydantic) + name + description, registers via
  the shared registry, then replaces ``tool.func`` with a mockable
  wrapper. Idempotent via the ``__lucidic_wrapped__`` sentinel that
  ``make_mockable_wrapper`` stamps.

Why no separate ``register_langchain_agent``: the modern (1.x)
LangChain agent API doesn't expose ``agent.tools`` consistently across
``AgentExecutor`` / ``RunnableAgent`` / ``langgraph`` paths. Asking
the user to pass the tool list explicitly is one extra line and
avoids LangChain-version-specific introspection. Add a thin agent
walker later if a real workflow needs it.

Pydantic schema → params: LangChain accepts both Pydantic v1 and v2
args schemas. ``model_json_schema()`` is available on both in modern
versions (v1 via the ``pydantic_v1`` compat shim). Using JSON Schema
as the intermediate form lets us reuse ``_params_from_json_schema``
from registry.py — same surface shape as OpenAI / Anthropic adapters.

Source hash strategy combines the JSON schema (signature equivalent)
+ the tool's function source. Behavioral drift in the user's impl
DOES surface here (unlike OpenAI / Anthropic adapters where we only
see the spec) — LangChain hands us the actual Python function.

Async tools (``tool.coroutine`` or ``StructuredTool.from_function(..., coroutine=...)``)
are handled — ``make_mockable_wrapper`` sniffs ``iscoroutinefunction``
on the wrapped callable and produces the matching wrapper. The
``tool.coroutine`` field is wrapped separately from ``tool.func`` so
both sync and async dispatch paths through LangChain route correctly.
"""
import hashlib
import inspect
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from ..mockable import make_mockable_wrapper
from ..registry import (
    ToolSurface,
    _params_from_callable,
    _params_from_json_schema,
    _stringify_annotation,
    register_tool,
)

if TYPE_CHECKING:
    from ....client import LucidicAI


logger = logging.getLogger("Lucidic")


# ---------------------------------------------------------------------------
# Registration + wrapping
# ---------------------------------------------------------------------------


def register_langchain_tools(
    tools: List[Any],
    *,
    client: Optional["LucidicAI"] = None,
) -> List[ToolSurface]:
    """Register and wrap a list of LangChain ``BaseTool`` instances.

    For each tool: extract a ``ToolSurface`` and register it, then
    transparently replace ``tool.func`` (and ``tool.coroutine`` if
    present) with a mockable wrapper. After this call, normal LangChain
    dispatch (``tool.invoke(...)``, ``agent.invoke(...)``) routes
    through our backend when a ``MockContext`` is bound, and runs the
    original function otherwise.

    Idempotent: re-registering the same tool is a no-op for the wrap
    step (sentinel check via ``__lucidic_wrapped__``). The registry
    side is last-wins as elsewhere.

    Args:
        tools: List of ``BaseTool`` instances (``Tool``, ``StructuredTool``,
            or any ``BaseTool`` subclass with a ``.func`` attribute).
        client: Optional explicit ``LucidicAI`` for the registry binding.
            Defaults to the contextvar-bound active client.

    Returns:
        The list of ``ToolSurface`` instances captured. Order matches
        the input list.
    """
    surfaces: List[ToolSurface] = []
    for tool in tools:
        # Skip non-tool objects defensively — tests + REPL workflows
        # sometimes pass a mixed list.
        if not _looks_like_basetool(tool):
            logger.debug(
                "[LangChain adapter] skipping %r — not a BaseTool",
                type(tool).__name__,
            )
            continue

        surface = _surface_from_langchain_tool(tool)
        surfaces.append(surface)
        _register_into(surface, client)
        _wrap_tool_callables(tool, surface)

    logger.debug(
        "[LangChain adapter] registered %d/%d tool(s)",
        len(surfaces), len(tools),
    )
    return surfaces


def _looks_like_basetool(obj: Any) -> bool:
    """Duck-type check for LangChain ``BaseTool`` shape.

    Avoids a hard import of ``langchain_core.tools.BaseTool`` so the
    SDK doesn't require LangChain at import time for users who don't
    use it. The minimal contract: has ``name`` (str) and either
    ``func`` or ``coroutine`` (callable).
    """
    name = getattr(obj, "name", None)
    if not isinstance(name, str):
        return False
    func = getattr(obj, "func", None)
    coro = getattr(obj, "coroutine", None)
    return callable(func) or callable(coro)


# ---------------------------------------------------------------------------
# Surface extraction
# ---------------------------------------------------------------------------


def _surface_from_langchain_tool(tool: Any) -> ToolSurface:
    """Build a ``ToolSurface`` for one ``BaseTool``.

    Source for each field:

    - ``name``: ``tool.name`` (required by ``BaseTool``).
    - ``docstring``: ``tool.description`` (LangChain treats this as
      the tool's docstring — same role as the OpenAI ``description``).
    - ``signature.params``: ``tool.args_schema.model_json_schema()`` if
      ``args_schema`` is set (the typical ``StructuredTool`` /
      ``Tool(args_schema=...)`` path), otherwise inspect
      ``tool.func`` directly via ``_params_from_callable``.
    - ``signature.return_type``: stringified return annotation of
      ``tool.func`` (best-effort; many LangChain tools don't annotate
      and we end up with None).
    - ``source_hash``: combines name + signature + body source. Body
      source comes from ``inspect.getsource(tool.func)`` so drift
      detection picks up behavioral changes — a real advantage over
      the OpenAI / Anthropic adapters where we only see the spec.
    """
    name = tool.name
    docstring = (tool.description or "") if hasattr(tool, "description") else ""
    func = getattr(tool, "func", None) or getattr(tool, "coroutine", None)

    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None and hasattr(args_schema, "model_json_schema"):
        try:
            params = _params_from_json_schema(args_schema.model_json_schema())
        except Exception:
            # Defensive: a v1 / v2 compat mismatch shouldn't kill
            # registration. Fall back to callable introspection.
            logger.warning(
                "[LangChain adapter] args_schema.model_json_schema() failed "
                "for tool %r; falling back to callable introspection",
                name,
            )
            params = _params_from_callable(func) if callable(func) else []
    elif callable(func):
        params = _params_from_callable(func)
    else:
        params = []

    return_type: Optional[str] = None
    if callable(func):
        try:
            sig = inspect.signature(func)
            return_type = _stringify_annotation(sig.return_annotation)
        except (TypeError, ValueError):
            return_type = None

    body_source = ""
    if callable(func):
        try:
            body_source = inspect.getsource(func)
        except (OSError, TypeError):
            body_source = ""

    signature = {"params": params, "return_type": return_type}

    return ToolSurface(
        name=name,
        signature=signature,
        docstring=docstring,
        return_shape=None,
        source_hash=_compute_langchain_source_hash(
            name=name,
            signature=signature,
            body_source=body_source,
        ),
    )


def _compute_langchain_source_hash(
    *,
    name: str,
    signature: dict,
    body_source: str,
) -> str:
    """SHA256 over ``name + signature_json + body_source``.

    Uses the same canonical encoding pattern as
    ``registry._compute_source_hash`` but keyed on the LangChain
    surface (signature derived from args_schema + body from
    ``inspect.getsource``). Newlines in body normalized to ``\\n`` so
    editor line-ending churn doesn't trip drift.
    """
    body = (body_source or "").replace("\r\n", "\n").replace("\r", "\n")
    sig_json = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    payload = f"{name}\n{sig_json}\n{body}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# tool.func / tool.coroutine replacement
# ---------------------------------------------------------------------------


def _wrap_tool_callables(tool: Any, surface: ToolSurface) -> None:
    """Replace ``tool.func`` and/or ``tool.coroutine`` with mockable wrappers.

    Idempotent via the ``__lucidic_wrapped__`` sentinel that
    ``make_mockable_wrapper`` stamps onto its output. Re-wrapping an
    already-wrapped tool is a no-op — common in dev workflows where
    the agent is re-instantiated on hot reload, or registration runs
    inside a loop.

    Each callable (sync ``func``, async ``coroutine``) is wrapped
    independently so the right wrapper-flavor is produced. LangChain
    dispatches to ``func`` for sync ``invoke()`` and ``coroutine`` for
    ``ainvoke()``; both must route through us when mocking is active.
    """
    func = getattr(tool, "func", None)
    if callable(func) and not getattr(func, "__lucidic_wrapped__", False):
        try:
            tool.func = make_mockable_wrapper(surface, func)
        except (TypeError, AttributeError) as exc:
            # Some BaseTool subclasses make `func` immutable (Pydantic
            # frozen=True). Surface as a warning rather than silently
            # failing — user can use @mockable directly or switch to a
            # mutable Tool variant.
            logger.warning(
                "[LangChain adapter] couldn't replace tool.func on %r (%s); "
                "this tool won't route through the backend. Decorate the "
                "underlying function with @mockable instead.",
                surface.name, exc,
            )

    coroutine = getattr(tool, "coroutine", None)
    if callable(coroutine) and not getattr(coroutine, "__lucidic_wrapped__", False):
        try:
            tool.coroutine = make_mockable_wrapper(surface, coroutine)
        except (TypeError, AttributeError) as exc:
            logger.warning(
                "[LangChain adapter] couldn't replace tool.coroutine on %r (%s)",
                surface.name, exc,
            )


def _register_into(
    surface: ToolSurface, client: Optional["LucidicAI"],
) -> None:
    """Route ``register_tool`` through an explicit client or the active
    contextvar — mirrors the OpenAI + Anthropic adapter logic."""
    if client is not None and hasattr(client, "tools"):
        client.tools._registry[surface.name] = surface  # noqa: SLF001
        return
    register_tool(surface)
