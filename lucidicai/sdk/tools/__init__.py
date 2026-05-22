"""SDK tool registry + ``@mockable`` decorator for v2 tool-backed dispatch.

Public surface exported here:

- ``ToolSurface`` — dataclass capturing one tool's signature/docstring/hash.
- ``register_tool`` — push a captured ``ToolSurface`` into the active client's
  registry (or buffer it when no client is alive yet).
- ``snapshot_registry`` — read the current set of registered ``ToolSurface``s,
  used by ``client.tools.sync()`` (LUC-577c) to build the request body for
  ``POST /sdk/agent-tools/sync``.
- ``mockable`` — decorator that captures a function's surface at decoration
  time and (in this ticket) returns a fast-path wrapper that runs the
  function unchanged. The dispatch-intercept layer is wired in LUC-577c.
- ``drain_buffer_into`` — internal hook used by ``LucidicAI.__init__``
  (LUC-577c) to flush decorators that ran before the client was constructed.

Adapter modules (LUC-578/579/580) consume helpers from ``registry`` for
their JSON-schema / Pydantic / callable paths; they're exported there
rather than re-exported here to keep this module the user-facing surface.
"""
from .mockable import mockable
from .registry import (
    ToolSurface,
    drain_buffer_into,
    register_tool,
    snapshot_registry,
)


__all__ = [
    "ToolSurface",
    "drain_buffer_into",
    "mockable",
    "register_tool",
    "snapshot_registry",
]
