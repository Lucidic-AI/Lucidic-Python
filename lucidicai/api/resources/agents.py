"""client.agents — agent reads + writes (LUC-906 / LUC-912).

Read the org's agents (``list`` / ``get`` / ``tool_catalog``) and provision them
(``create`` / ``update``). ``create`` is the SDK-first bootstrap — since LUC-926
a client needs only an api_key to construct, so ``client.agents.create(...)``
can mint an org's first agent purely from code. There is no ``delete`` (that
cascades to every session/event/tool/prompt under the agent — dashboard-only).

Both reads and writes are data-bearing, so — unlike the telemetry resources —
they do NOT swallow errors in production: a failure raises the typed
``LucidicError`` subclass the transport decoded (``NotFoundError``,
``InsufficientScopeError``, ``ValidationError``, …). Every method has an
``a``-prefixed async sibling.
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

    # ==================== write (LUC-912) ====================

    def create(
        self, name: str, *, icon: Optional[str] = None, project_id: Optional[str] = None,
    ) -> Agent:
        """Create an agent in the key's org — the SDK-first bootstrap.

        Needs ``agent:write``. An agent-**bound** key can't create agents (it
        could only ever see its one agent) → the backend returns 403 →
        ``InsufficientScopeError`` (a binding restriction, not a missing scope;
        see that error's docs). ``icon`` defaults backend-side ("compass") when
        omitted; ``project_id`` optionally files the new agent under a project.
        """
        return Agent.from_dict(self.http.post(_AGENTS, self._create_body(name, icon, project_id)))

    async def acreate(
        self, name: str, *, icon: Optional[str] = None, project_id: Optional[str] = None,
    ) -> Agent:
        """Async sibling of ``create``."""
        body = self._create_body(name, icon, project_id)
        return Agent.from_dict(await self.http.apost(_AGENTS, body))

    def update(
        self, agent_id: str, *, name: Optional[str] = None, icon: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Agent:
        """Update an agent's name / icon / project (e.g. move it between
        projects). Only the fields you pass change; ``None`` leaves a field
        untouched. An over-length name/icon → ``ValidationError``."""
        return Agent.from_dict(
            self.http.put(f"{_AGENTS}/{agent_id}", self._update_body(name, icon, project_id))
        )

    async def aupdate(
        self, agent_id: str, *, name: Optional[str] = None, icon: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Agent:
        """Async sibling of ``update``."""
        return Agent.from_dict(
            await self.http.aput(f"{_AGENTS}/{agent_id}", self._update_body(name, icon, project_id))
        )

    @staticmethod
    def _create_body(
        name: str, icon: Optional[str], project_id: Optional[str]
    ) -> Dict[str, Any]:
        # Omit unset optional fields uniformly (like _update_body) so the backend
        # owns their defaults — don't hardcode "compass" here.
        body: Dict[str, Any] = {"name": name}
        if icon is not None:
            body["icon"] = icon
        if project_id is not None:
            body["project_id"] = project_id
        return body

    @staticmethod
    def _update_body(
        name: Optional[str], icon: Optional[str], project_id: Optional[str]
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if icon is not None:
            body["icon"] = icon
        if project_id is not None:
            body["project_id"] = project_id
        return body

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
