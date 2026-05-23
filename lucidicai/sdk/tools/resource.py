"""``client.tools`` namespace — the user-facing API for tool dispatch.

Holds the per-client registry (filled by ``@mockable`` decorations +
adapter ``register_*`` calls), exposes the canonical instance-bound
decorator (``client.tools.mockable``), and drives the two backend
side effects that turn registrations into live dispatch:

- ``sync()`` → ``POST /sdk/agent-tools/sync`` — push the registry to
  the backend so the dashboard surfaces tools and ``mock_call`` can
  resolve them. Auto-fired on first tool-backed session start via
  ``_maybe_sync_on_session_start``; manually callable for power users.
- ``_init_session(session_id)`` → ``POST /sdk/session-init-fixtures`` —
  ask the backend to materialize per-session DuckDB state and write
  the ``tool_version_snapshot``. On success, binds a ``MockContext``
  for the session so subsequent ``@mockable`` calls route through the
  backend. Called by ``SessionResource.create()``.

Debounce semantics: ``_last_synced_hash`` records the sha256 of the
sorted registry's source_hashes. Re-firing ``_maybe_sync_on_session_start``
when the fingerprint hasn't changed is a cheap no-op — customer code
that creates many sessions in a loop (eval runs, batch processing)
syncs exactly once per registry change, not per session.
"""
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ...api.resources.agent_tools_sync import SyncAgentToolsResource
from ...api.resources.session_init_fixtures import SessionInitFixturesResource
from ...core.errors import LucidicError
from .context import MockContext, bind_mock_context
from .registry import ToolSurface

if TYPE_CHECKING:
    from ...client import LucidicAI


logger = logging.getLogger("Lucidic")


class ToolsResource:
    """User-facing ``client.tools`` namespace.

    Constructed once per ``LucidicAI`` client; held in
    ``client._resources["tools"]`` and exposed via the ``client.tools``
    property. Threading: the registry dict is read/written under the
    GIL only — no explicit lock since decorations happen at import
    time (single-threaded) and ``sync()`` snapshots before sending.
    """

    def __init__(self, client: "LucidicAI"):
        self._client = client
        self._registry: Dict[str, ToolSurface] = {}
        # Sync debounce — sha256 of sorted source_hashes. Re-firing
        # _maybe_sync_on_session_start with an unchanged fingerprint
        # skips the network round trip.
        self._last_synced_hash: Optional[str] = None

        self._sync_api = SyncAgentToolsResource(client._http)
        self._init_fixtures_api = SessionInitFixturesResource(client._http)

        # Note: ``LucidicAI.__init__`` calls ``drain_buffer_into(self)``
        # after wiring this resource into ``_resources["tools"]``.
        # Draining here would deadlock — ``drain_buffer_into`` accesses
        # ``client.tools`` which returns from ``_resources["tools"]``
        # that's still being constructed.

    # ==================== Public surface ====================

    def mockable(self, func: Callable) -> Callable:
        """Canonical instance-bound decorator.

        Equivalent to module-level ``lucidic.mockable`` but binds the
        captured surface to this client's registry directly (no buffer
        round trip). Useful when multiple ``LucidicAI`` instances coexist
        in one process — explicit binding avoids the first-wins buffer
        semantics.

        Example::

            client = lucidic.LucidicAI(api_key="...", agent_id="...")

            @client.tools.mockable
            def query_emails(sender: str) -> list[dict]:
                ...
        """
        # Delegate to the module-level decorator. Its register_tool
        # call will see this client via get_active_client() if the
        # client is bound to the current context — otherwise the
        # surface falls into the buffer. We force the registry
        # binding here so this method works even when no client is
        # currently bound to the contextvar.
        from .mockable import mockable as _mockable_decorator

        wrapped = _mockable_decorator(func)
        # Directly stamp this client's registry — the decorator's
        # register_tool may have buffered if no contextvar client is
        # active. Idempotent: last-wins matches registry semantics.
        surface = wrapped.__lucidic_surface__
        self._registry[surface.name] = surface
        return wrapped

    def register(self, surface: ToolSurface) -> None:
        """Add a ``ToolSurface`` directly to this client's registry.

        Used by adapter helpers (LUC-578/579/580) that build surfaces
        from external tool definitions (LangChain ``BaseTool``, OpenAI
        function spec, Anthropic tool block) rather than via the
        ``@mockable`` decoration path.
        """
        self._registry[surface.name] = surface

    # ==================== Framework adapters (LUC-578/579/580) ====================

    def register_openai(self, tools: List[Dict[str, Any]]) -> List[ToolSurface]:
        """Canonical instance-bound version of ``register_openai_tools``.

        Walks the ``tools=`` list typically passed to
        ``openai.chat.completions.create`` and registers each
        function-typed entry into this client's registry. See
        ``lucidicai.sdk.tools.adapters.openai.register_openai_tools``
        for the full contract.
        """
        from .adapters.openai import register_openai_tools

        return register_openai_tools(tools, client=self._client)

    def dispatch_openai(
        self,
        call: Any,
        impls: Dict[str, Callable],
        *,
        client_event_id: Optional[str] = None,
    ) -> Any:
        """Canonical instance-bound version of ``dispatch_openai_tool_call``.

        Routes one OpenAI tool_call through the mock-call backend when
        a ``MockContext`` is bound, otherwise invokes ``impls[name]``.
        See ``lucidicai.sdk.tools.adapters.openai.dispatch_openai_tool_call``
        for the full behavior matrix.
        """
        from .adapters.openai import dispatch_openai_tool_call

        return dispatch_openai_tool_call(
            call, impls,
            client=self._client,
            client_event_id=client_event_id,
        )

    async def adispatch_openai(
        self,
        call: Any,
        impls: Dict[str, Callable],
        *,
        client_event_id: Optional[str] = None,
    ) -> Any:
        """Async sibling of ``dispatch_openai``."""
        from .adapters.openai import adispatch_openai_tool_call

        return await adispatch_openai_tool_call(
            call, impls,
            client=self._client,
            client_event_id=client_event_id,
        )

    # ----- Anthropic adapter (LUC-580) -----

    def register_anthropic(self, tools: List[Dict[str, Any]]) -> List[ToolSurface]:
        """Canonical instance-bound version of ``register_anthropic_tools``.

        Walks the ``tools=`` list typically passed to
        ``anthropic.Anthropic().messages.create`` and registers each
        entry into this client's registry. See
        ``lucidicai.sdk.tools.adapters.anthropic.register_anthropic_tools``
        for the full contract.
        """
        from .adapters.anthropic import register_anthropic_tools

        return register_anthropic_tools(tools, client=self._client)

    def dispatch_anthropic(
        self,
        block: Any,
        impls: Dict[str, Callable],
        *,
        client_event_id: Optional[str] = None,
    ) -> Any:
        """Canonical instance-bound version of ``dispatch_anthropic_tool_call``.

        Routes one Anthropic ``tool_use`` block through the mock-call
        backend when a ``MockContext`` is bound, otherwise invokes
        ``impls[name]``. See
        ``lucidicai.sdk.tools.adapters.anthropic.dispatch_anthropic_tool_call``
        for the full behavior matrix.
        """
        from .adapters.anthropic import dispatch_anthropic_tool_call

        return dispatch_anthropic_tool_call(
            block, impls,
            client=self._client,
            client_event_id=client_event_id,
        )

    async def adispatch_anthropic(
        self,
        block: Any,
        impls: Dict[str, Callable],
        *,
        client_event_id: Optional[str] = None,
    ) -> Any:
        """Async sibling of ``dispatch_anthropic``."""
        from .adapters.anthropic import adispatch_anthropic_tool_call

        return await adispatch_anthropic_tool_call(
            block, impls,
            client=self._client,
            client_event_id=client_event_id,
        )

    def snapshot(self) -> List[ToolSurface]:
        """All registered surfaces in stable name order.

        Used by ``sync()`` to build the wire payload; also useful for
        inspection / debugging (``client.tools.snapshot()`` in a REPL).
        """
        return sorted(self._registry.values(), key=lambda s: s.name)

    def sync(self) -> Dict[str, Any]:
        """Push the current registry to the backend.

        Builds the wire payload from ``snapshot()``, POSTs to
        ``/sdk/agent-tools/sync``, updates ``_last_synced_hash`` on
        success. Returns the backend's ``stats`` payload.

        No-op when the registry is empty (backend would 200 with
        zero updates; we save the round trip). Logs at INFO so the
        user sees sync activity in normal verbose logs.

        Raises ``LucidicError`` on backend failures (422 validation,
        503 lock contention). Doesn't retry — caller decides.
        """
        surfaces = self.snapshot()
        if not surfaces:
            logger.debug("[ToolsResource] sync() called with empty registry; no-op")
            return {"synced": True, "stats": {"tools": 0}}

        agent_id = self._client._config.agent_id
        if not agent_id:
            raise LucidicError(
                "client.tools.sync() requires a configured agent_id "
                "(LUCIDIC_AGENT_ID env var or LucidicAI(agent_id=...))"
            )

        payload = [_surface_to_wire(s) for s in surfaces]
        logger.info(
            "[ToolsResource] syncing %d tool(s) to backend for agent %s",
            len(payload), str(agent_id)[:8] + "...",
        )
        result = self._sync_api.sync(agent_id=str(agent_id), tools=payload)
        self._last_synced_hash = self._registry_fingerprint(surfaces)
        return result

    async def async_(self) -> Dict[str, Any]:
        """Async sibling of ``sync``. Trailing underscore avoids the
        ``async`` keyword collision."""
        surfaces = self.snapshot()
        if not surfaces:
            return {"synced": True, "stats": {"tools": 0}}

        agent_id = self._client._config.agent_id
        if not agent_id:
            raise LucidicError(
                "client.tools.sync() requires a configured agent_id"
            )

        payload = [_surface_to_wire(s) for s in surfaces]
        result = await self._sync_api.async_sync(
            agent_id=str(agent_id), tools=payload,
        )
        self._last_synced_hash = self._registry_fingerprint(surfaces)
        return result

    # ==================== Lifecycle hooks (called from SessionResource) ====================

    def _init_session(self, session_id: str) -> None:
        """Initialize per-session fixture state + bind MockContext.

        Called from ``SessionResource.create()`` immediately after the
        session is created on the backend, when the user provided a
        ``datasetitem_id`` (the signal that they intend tool-backed
        dispatch). Three outcomes:

        1. Backend returns ``initialized=True``: session is tool-backed
           and ready. Bind ``MockContext(session_id, client)`` for
           ``@mockable`` to consult. Fire the debounced auto-sync so
           the registry is on the backend before any ``mock_call``.

        2. Backend returns ``initialized=False`` (no DatasetItem on
           server side, mismatch, no Resources/Tools to snapshot):
           log DEBUG and continue. Session works as normal observability
           session; ``@mockable`` calls short-circuit to running the
           local function.

        3. Backend error or network failure: log WARNING and continue.
           ``mock_call`` will return ``session_not_initialized`` later
           if the user attempts dispatch, which surfaces the issue
           with full typed-exception context.

        NEVER raises — session creation must not be blocked by tool
        init failures. The session is observability-functional even
        without tool state.
        """
        try:
            result = self._init_fixtures_api.init(session_id)
        except LucidicError as exc:
            logger.warning(
                "[ToolsResource] session-init-fixtures failed for %s: %s — "
                "mock_call attempts in this session will return "
                "session_not_initialized",
                session_id[:8] + "...", exc,
            )
            return

        if not result.get("initialized"):
            logger.debug(
                "[ToolsResource] session %s is not tool-backed (reason: %s); "
                "skipping MockContext binding",
                session_id[:8] + "...", result.get("reason", "?"),
            )
            return

        # Bind MockContext so @mockable wrappers route through transport
        bind_mock_context(MockContext(session_id=session_id, client=self._client))
        logger.debug(
            "[ToolsResource] bound MockContext for session %s "
            "(fixtures=%d, tools_snapshotted=%d)",
            session_id[:8] + "...",
            len(result.get("fixture_ids", [])),
            len(result.get("tools_snapshotted", [])),
        )

        # First-session-start sync trigger (debounced). Done after
        # MockContext binds so any mockable call that fires immediately
        # after session creation has the backend already aware of the
        # tools (avoids a transient unknown_tool 404).
        self._maybe_sync_on_session_start()

    async def _ainit_session(self, session_id: str) -> None:
        """Async sibling of ``_init_session``. Mirrors all the same
        contracts including the no-raise guarantee."""
        try:
            result = await self._init_fixtures_api.ainit(session_id)
        except LucidicError as exc:
            logger.warning(
                "[ToolsResource] session-init-fixtures (async) failed for %s: %s",
                session_id[:8] + "...", exc,
            )
            return

        if not result.get("initialized"):
            logger.debug(
                "[ToolsResource] async session %s not tool-backed (reason: %s)",
                session_id[:8] + "...", result.get("reason", "?"),
            )
            return

        bind_mock_context(MockContext(session_id=session_id, client=self._client))
        # Sync is best-effort here; await the async version
        try:
            await self._maybe_async_sync_on_session_start()
        except LucidicError as exc:
            logger.warning("[ToolsResource] auto-sync failed: %s", exc)

    # ==================== Internals ====================

    def _maybe_sync_on_session_start(self) -> None:
        """Debounced auto-sync of the registry.

        Fingerprints the current registry; if it matches the last
        sync's fingerprint, no-op (registry hasn't changed → backend
        already has this exact tool catalog). Otherwise syncs.

        Wraps ``sync()`` errors in WARNING + suppresses — auto-sync is
        a convenience, not a contract. Manual ``client.tools.sync()``
        re-raises so power users see real failures.
        """
        surfaces = self.snapshot()
        fingerprint = self._registry_fingerprint(surfaces)
        if fingerprint == self._last_synced_hash:
            logger.debug(
                "[ToolsResource] auto-sync: fingerprint unchanged, skipping"
            )
            return
        try:
            self.sync()
        except LucidicError as exc:
            logger.warning("[ToolsResource] auto-sync failed: %s", exc)

    async def _maybe_async_sync_on_session_start(self) -> None:
        surfaces = self.snapshot()
        fingerprint = self._registry_fingerprint(surfaces)
        if fingerprint == self._last_synced_hash:
            return
        try:
            await self.async_()
        except LucidicError as exc:
            logger.warning("[ToolsResource] async auto-sync failed: %s", exc)

    def _registry_fingerprint(
        self, surfaces: Optional[List[ToolSurface]] = None,
    ) -> str:
        """SHA256 of sorted ``name:source_hash`` pairs for debounce.

        Cheap: hashing one line per tool. Stable: sorted by name so
        re-ordering doesn't trip the comparison.
        """
        if surfaces is None:
            surfaces = self.snapshot()
        lines = [f"{s.name}:{s.source_hash}" for s in surfaces]
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _surface_to_wire(surface: ToolSurface) -> Dict[str, Any]:
    """Convert a ``ToolSurface`` to the JSON shape ``SyncAgentToolsSerializer``
    accepts.

    The dataclass field names happen to match the serializer's fields
    1:1; the explicit mapping here documents the contract and keeps a
    seam for future shape drift.
    """
    return {
        "name": surface.name,
        "signature": surface.signature,
        "docstring": surface.docstring,
        "return_shape": surface.return_shape,
        "source_hash": surface.source_hash,
    }
