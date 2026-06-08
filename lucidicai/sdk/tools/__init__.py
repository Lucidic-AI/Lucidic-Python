"""SDK tool registry + ``@mockable`` decorator for v2 tool-backed dispatch.

Public surface (consumed by user code and adapter modules):

- ``ToolSurface`` — captured surface metadata (LUC-577a).
- ``register_tool`` — push a captured surface into the active client's
  registry, or buffer when no client is alive yet (LUC-577a).
- ``snapshot_registry`` — read all currently-registered surfaces.
- ``mockable`` — decorator capturing + routing dispatch (LUC-577a + 608).
- ``drain_buffer_into`` — internal hook for ``LucidicAI.__init__``.
- ``ToolsResource`` — the ``client.tools`` namespace (LUC-608).
- ``MockContext`` — bound state telling ``@mockable`` to route through
  the backend; set by ``ToolsResource._init_session`` on tool-backed
  session start (LUC-608).
- ``bind_mock_context`` / ``clear_mock_context`` — lifecycle hooks for
  session start/end (LUC-608).
- ``emit_call_through_backend`` / ``aemit_call_through_backend`` —
  transport layer used by ``@mockable`` wrappers and framework
  adapters (LUC-607).

Adapter modules (LUC-578/579/580) consume the registry's shared
helpers (``_params_from_json_schema``, ``_params_from_callable``,
``_compute_source_hash``) directly from ``registry.py`` — not re-exported
here to keep this module the *user-facing* surface.
"""
from .context import (
    MockContext,
    bind_mock_context,
    clear_mock_context,
    current_mock_context,
)
from .mockable import mockable
from .registry import (
    ToolSurface,
    drain_buffer_into,
    register_tool,
    snapshot_registry,
)
from .resource import ToolsResource
from .transport import aemit_call_through_backend, emit_call_through_backend


__all__ = [
    # Surface capture + registry (LUC-577a)
    "ToolSurface",
    "drain_buffer_into",
    "mockable",
    "register_tool",
    "snapshot_registry",
    # Transport (LUC-607)
    "aemit_call_through_backend",
    "emit_call_through_backend",
    # Client namespace + mock context (LUC-608)
    "MockContext",
    "ToolsResource",
    "bind_mock_context",
    "clear_mock_context",
    "current_mock_context",
]
