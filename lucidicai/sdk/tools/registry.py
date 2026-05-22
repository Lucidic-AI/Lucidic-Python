"""Tool surface registry + shared capture helpers.

``ToolSurface`` is the SDK-side mirror of the backend ``Tool`` model
(``api/models/tool_models.py``) and matches the wire shape that
``POST /sdk/agent-tools/sync`` accepts via ``SyncAgentToolsSerializer``.
The structural contract is frozen — adding a field here means adding it
to the serializer, otherwise the backend rejects the sync.

Two storage targets:

- ``_REGISTRY`` — keyed by tool name; last-wins semantics. Decorators
  applied in the same process push here directly when a client is active.
- ``_PENDING_BUFFER`` — append-only list captured when no ``LucidicAI``
  client exists at decoration time (e.g. ``@mockable`` at module import,
  ``LucidicAI(...)`` constructed later in ``main()``). ``LucidicAI.__init__``
  in LUC-577c calls ``drain_buffer_into(client)`` to flush these into the
  client's tools resource registry.

The module exposes shared capture helpers used by ``@mockable`` (LUC-577a),
the OpenAI adapter (LUC-579), the Anthropic adapter (LUC-580), and the
LangChain adapter (LUC-578) so all four paths emit identical ``ToolSurface``
shapes and consistent ``source_hash`` values.
"""
import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ...client import LucidicAI


@dataclass(frozen=True)
class ToolSurface:
    """One tool's discovered surface.

    Field shapes match the backend ``SyncAgentToolsSerializer`` (see
    ``api/serializers/agent_tools.py`` in AnalyticsAPIBackend). ``return_shape``
    is reserved for future return-type inference; null in v1.

    ``source_hash`` is the drift-detection key. The backend stores whatever
    the SDK sends and compares verbatim on the next sync, so the byte-level
    definition (see ``_compute_source_hash``) is the canonical source of
    truth — changing it invalidates every customer's stored snapshots.
    """

    name: str
    signature: dict
    docstring: str
    return_shape: dict | None
    source_hash: str


_REGISTRY: dict[str, ToolSurface] = {}
_PENDING_BUFFER: list[ToolSurface] = []


def register_tool(surface: ToolSurface) -> None:
    """Add ``surface`` to whichever store is appropriate.

    When a ``LucidicAI`` client is active (set via the ``current_client``
    contextvar by ``LucidicAI.__init__``), the surface lands directly in
    the client's tool registry — implemented in LUC-577c when the
    ``client.tools`` namespace exists. Until then, all calls fall through
    to ``_PENDING_BUFFER``.

    Last-wins on duplicate names. Re-decorating a function (e.g. in a hot
    reload) replaces the prior surface; the registry never grows unbounded
    from repeated decoration.
    """
    # Direct-to-client path activates in LUC-577c. Until then everything
    # buffers, and 577c's drain_buffer_into flushes on first sync.
    from ..context import get_active_client

    client = get_active_client()
    if client is not None and hasattr(client, "tools"):
        client.tools._registry[surface.name] = surface  # noqa: SLF001
        return

    _REGISTRY[surface.name] = surface
    _PENDING_BUFFER.append(surface)


def snapshot_registry() -> list[ToolSurface]:
    """Return all registered surfaces in stable name order.

    Caller is typically ``client.tools.sync()`` building the
    ``/sdk/agent-tools/sync`` request body. Stable ordering keeps the
    debounce fingerprint (LUC-577c) deterministic across runs.
    """
    return sorted(_REGISTRY.values(), key=lambda s: s.name)


def drain_buffer_into(client: "LucidicAI") -> int:
    """Flush ``_PENDING_BUFFER`` into ``client.tools._registry`` and clear it.

    Called by ``LucidicAI.__init__`` in LUC-577c. Returns the number of
    surfaces drained so the caller can decide whether to trigger an
    opportunistic sync. Idempotent — calling on an empty buffer is a
    cheap no-op.

    Multi-client processes: this attaches *all* buffered surfaces to the
    client that calls it. Whichever client constructs first wins; later
    clients see an empty buffer. Documented as the explicit semantic in
    LUC-577c — multi-client workflows should decorate per-client via
    ``client.tools.mockable(fn)`` rather than module-level ``@mockable``.
    """
    if not _PENDING_BUFFER:
        return 0
    count = 0
    for surface in _PENDING_BUFFER:
        client.tools._registry[surface.name] = surface  # noqa: SLF001
        count += 1
    _PENDING_BUFFER.clear()
    # Keep the module-level _REGISTRY mirror in sync — adapters and
    # standalone callers may still consult it directly.
    return count


# ---------------------------------------------------------------------------
# Shared capture helpers (consumed by mockable + framework adapters)
# ---------------------------------------------------------------------------


def _stringify_annotation(annotation: Any) -> str | None:
    """Reduce a Python annotation to a short string for the wire.

    Returns ``None`` for the sentinel ``inspect.Parameter.empty`` /
    ``inspect.Signature.empty`` since the serializer treats unannotated
    parameters as ``type: "any"`` after this passes through
    ``_params_from_callable``. Generic aliases (``list[str]``,
    ``Optional[int]``) fall through to ``str(annotation)`` — readable
    even if not round-trippable.
    """
    if annotation is inspect.Parameter.empty or annotation is inspect.Signature.empty:
        return None
    if annotation is type(None):
        return "None"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _serialize_default(default: Any) -> Any:
    """Best-effort JSON-safe representation of a parameter default.

    The serializer's ``default`` field is permissive (``JSONField``,
    allow_null=True) so we don't have to be conservative. We do guard
    against arbitrary objects whose ``repr`` includes a memory address —
    those would make ``source_hash`` non-deterministic. Containers are
    accepted if they survive ``json.dumps``; everything else falls back
    to ``repr``.
    """
    if default is inspect.Parameter.empty:
        # "no default" — caller signals this separately via required=True.
        return None
    if default is None or isinstance(default, (str, int, float, bool)):
        return default
    if isinstance(default, (list, tuple, dict)):
        try:
            json.dumps(default)
            return list(default) if isinstance(default, tuple) else default
        except (TypeError, ValueError):
            return repr(default)
    return repr(default)


def _params_from_callable(fn: Callable) -> list[dict]:
    """Extract parameter specs from a Python callable.

    Used by ``@mockable`` (LUC-577a) and the LangChain legacy ``Tool(func=...)``
    path (LUC-578). The OpenAI / Anthropic adapters call
    ``_params_from_json_schema`` instead because their tool surface is JSON
    Schema, not a Python callable.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # Builtins without introspectable signatures — emit empty params
        # rather than failing the registration. The user can correct by
        # wrapping in a typed Python def.
        return []
    out: list[dict] = []
    for p in sig.parameters.values():
        # Skip *args / **kwargs — neither serializes cleanly in the v1
        # signature shape. Backend's typed kwargs dispatch ignores them too.
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        out.append({
            "name": p.name,
            "type": _stringify_annotation(p.annotation) or "any",
            "default": _serialize_default(p.default),
            "required": p.default is inspect.Parameter.empty,
        })
    return out


def _params_from_json_schema(schema: dict) -> list[dict]:
    """Convert a JSON Schema object into ``ToolSurface.signature.params``.

    Shared by the OpenAI adapter (function-call ``parameters``), the
    Anthropic adapter (tool ``input_schema``), and the LangChain adapter
    (Pydantic ``model_json_schema()``). Keeps the on-wire shape identical
    regardless of which framework the surface came from.

    Permissive on missing fields — JSON Schema in the wild often omits
    ``type`` on parameters that the model is expected to infer; we
    default to ``"any"`` rather than rejecting.
    """
    if not isinstance(schema, dict):
        return []
    required = set(schema.get("required") or [])
    out: list[dict] = []
    for prop_name, prop_schema in (schema.get("properties") or {}).items():
        if not isinstance(prop_schema, dict):
            prop_schema = {}
        out.append({
            "name": prop_name,
            "type": prop_schema.get("type", "any"),
            "default": prop_schema.get("default"),
            "required": prop_name in required,
        })
    return out


def _canonical_signature_json(signature: dict) -> str:
    """Stable JSON encoding for the source-hash input.

    Sort dict keys at every level so re-ordering Python kwargs doesn't
    change the hash. ``sort_keys=True`` handles top-level; for nested
    objects we rely on ``json.dumps``'s recursive sort.
    """
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _compute_source_hash(
    *,
    name: str,
    signature: dict,
    body_source: str | None,
) -> str:
    """SHA256 hex digest of the canonical surface representation.

    Input bytes are ``name + "\\n" + canonical_signature_json + "\\n" + body``
    so that signature changes drift independently from body changes — a
    docstring tweak (which doesn't touch the signature) only drifts if
    the source body changes, and vice versa.

    ``body_source`` is ``inspect.getsource(fn)`` for ``@mockable``, the
    canonical JSON spec for the adapter paths (OpenAI / Anthropic /
    LangChain with Pydantic), or an empty string when no source is
    available (LangChain legacy ``Tool`` with an opaque callable). Newlines
    are normalized to ``\\n`` so editor line-ending churn doesn't trip
    drift on otherwise-identical content.

    Backend stores this verbatim and compares on the next sync. Changing
    the encoding here invalidates every existing customer's snapshot.
    Pin via tests in ``test_registry.py`` before any refactor.
    """
    body = (body_source or "").replace("\r\n", "\n").replace("\r", "\n")
    sig_json = _canonical_signature_json(signature)
    payload = f"{name}\n{sig_json}\n{body}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
