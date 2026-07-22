"""client.agents — agent reads (LUC-906).

Read the org's agents: ``list`` them (discover / enumerate ``agent_id`` s),
``get`` one back, or inspect an agent's ``tool_catalog``.

(A brand-new caller with *no* ``agent_id`` at all still can't construct the
client to reach ``list`` — ``LucidicAI`` requires one today; a read-only /
bootstrap construction mode is a planned follow-up. With any existing key +
``agent_id`` you can enumerate the rest.)

Reads are data-bearing, so — unlike the telemetry-write resources — they do
NOT swallow errors in production: a failure raises the typed ``LucidicError``
subclass the transport decoded (``NotFoundError``, ``InsufficientScopeError``,
…). Every method has an ``a``-prefixed async sibling.
"""
from typing import Any, AsyncIterator, Dict, Iterator, Optional

from ..client import HttpClient
from ..models.agent import Agent, AgentToolCatalog
from ..models.base import CursorPage
from ..pagination import apaginate, paginate

_AGENTS = "sdk/v2/agents"


class AgentsResource:
    """Handle for the ``/sdk/v2/agents`` read endpoints."""

    def __init__(self, http: HttpClient):
        self.http = http

    # ==================== list ====================

    def list(
        self, *, page_size: Optional[int] = None, ordering: Optional[str] = None
    ) -> Iterator[Agent]:
        """Lazily iterate the org's agents (backend default order: newest first),
        following pages.

        ``for agent in client.agents.list(): ...`` — a bound key sees only its
        one agent. ``ordering`` accepts ``id`` / ``-id`` (the only
        client-selectable ordering field).

        Note: this is a lazy generator — request errors (403/etc.) surface on
        the first iteration, not at the ``list()`` call. Use ``list_page`` for
        an eager single-page fetch.
        """
        return paginate(
            lambda cursor: self._page(cursor, page_size, ordering), model=Agent
        )

    def alist(
        self, *, page_size: Optional[int] = None, ordering: Optional[str] = None
    ) -> AsyncIterator[Agent]:
        """Async sibling of ``list`` — ``async for agent in client.agents.alist()``."""
        return apaginate(
            lambda cursor: self._apage(cursor, page_size, ordering), model=Agent
        )

    def list_page(
        self,
        *,
        cursor: Optional[str] = None,
        page_size: Optional[int] = None,
        ordering: Optional[str] = None,
    ) -> CursorPage:
        """Fetch a single page of agents (manual pagination control)."""
        return CursorPage.from_body(self._page(cursor, page_size, ordering), model=Agent)

    async def alist_page(
        self,
        *,
        cursor: Optional[str] = None,
        page_size: Optional[int] = None,
        ordering: Optional[str] = None,
    ) -> CursorPage:
        """Async sibling of ``list_page``."""
        body = await self._apage(cursor, page_size, ordering)
        return CursorPage.from_body(body, model=Agent)

    # ==================== get ====================

    def get(self, agent_id: str) -> Agent:
        """Read one agent by id. Raises ``NotFoundError`` if it doesn't exist
        or the (bound) key can't see it."""
        return Agent.from_dict(self.http.get(f"{_AGENTS}/{agent_id}"))

    async def aget(self, agent_id: str) -> Agent:
        """Async sibling of ``get``."""
        return Agent.from_dict(await self.http.aget(f"{_AGENTS}/{agent_id}"))

    # ==================== tool catalog ====================

    def tool_catalog(self, agent_id: str) -> AgentToolCatalog:
        """Read the agent's tool catalog: per-tool tier/impl/drift + denormalized
        usage (``call_count`` / ``last_called_at``) and the reachable Resources."""
        return AgentToolCatalog.from_dict(self.http.get(f"{_AGENTS}/{agent_id}/tool-catalog"))

    async def atool_catalog(self, agent_id: str) -> AgentToolCatalog:
        """Async sibling of ``tool_catalog``."""
        body = await self.http.aget(f"{_AGENTS}/{agent_id}/tool-catalog")
        return AgentToolCatalog.from_dict(body)

    # ==================== internals ====================

    @staticmethod
    def _params(
        cursor: Optional[str], page_size: Optional[int], ordering: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if cursor:
            params["cursor"] = cursor
        if page_size is not None:
            params["page_size"] = page_size
        if ordering is not None:
            params["ordering"] = ordering
        return params or None

    def _page(
        self, cursor: Optional[str], page_size: Optional[int], ordering: Optional[str]
    ) -> Dict[str, Any]:
        return self.http.get(_AGENTS, self._params(cursor, page_size, ordering))

    async def _apage(
        self, cursor: Optional[str], page_size: Optional[int], ordering: Optional[str]
    ) -> Dict[str, Any]:
        return await self.http.aget(_AGENTS, self._params(cursor, page_size, ordering))
