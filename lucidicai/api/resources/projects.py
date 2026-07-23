"""client.projects — project CRUD (LUC-913).

A new namespace. Projects are org-level groupings agents can be filed under
(``Agent.project``). Org-scoped: a bound key still reaches them all (agent
binding is a no-op here). ``delete`` is low-risk — ``Agent.project`` is
SET_NULL, so deleting a project un-projects its agents rather than destroying
them.

Reads and writes are data-bearing — they do NOT swallow in production; typed
transport errors propagate. Every method has an ``a``-prefixed async sibling.
"""
from typing import Any, AsyncIterator, Dict, Iterator, Optional

from ..client import HttpClient
from ..models.base import CursorPage
from ..models.project import Project
from ..pagination import apaginate, paginate

_PROJECTS = "sdk/v2/projects"


class ProjectsResource:
    """Handle for the ``/sdk/v2/projects`` endpoints."""

    def __init__(self, http: HttpClient):
        self.http = http

    # ==================== list ====================

    def list(
        self, *, ordering: Optional[str] = None, page_size: Optional[int] = None
    ) -> Iterator[Project]:
        """Lazily iterate the org's projects, newest first. ``ordering`` accepts
        ``id`` (± prefix). Handy for resolving a ``project_id`` by name."""
        base = self._params(ordering, page_size)
        return paginate(lambda c: self._page_get(base, c), model=Project)

    def alist(
        self, *, ordering: Optional[str] = None, page_size: Optional[int] = None
    ) -> AsyncIterator[Project]:
        """Async sibling of ``list``."""
        base = self._params(ordering, page_size)
        return apaginate(lambda c: self._apage_get(base, c), model=Project)

    def list_page(
        self, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of projects."""
        return CursorPage.from_body(
            self._page_get(self._params(ordering, page_size), cursor), model=Project
        )

    async def alist_page(
        self, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``list_page``."""
        body = await self._apage_get(self._params(ordering, page_size), cursor)
        return CursorPage.from_body(body, model=Project)

    # ==================== get ====================

    def get(self, project_id: str) -> Project:
        """Read one project by id (raises ``NotFoundError`` if absent)."""
        return Project.from_dict(self.http.get(f"{_PROJECTS}/{project_id}"))

    async def aget(self, project_id: str) -> Project:
        """Async sibling of ``get``."""
        return Project.from_dict(await self.http.aget(f"{_PROJECTS}/{project_id}"))

    # ==================== create ====================

    def create(self, name: str, icon: str, description: Optional[str] = None) -> Project:
        """Create a project in the key's org. ``name`` and ``icon`` are required
        by the backend; an over-length field → ``ValidationError``."""
        return Project.from_dict(self.http.post(_PROJECTS, self._create_body(name, icon, description)))

    async def acreate(self, name: str, icon: str, description: Optional[str] = None) -> Project:
        """Async sibling of ``create``."""
        body = self._create_body(name, icon, description)
        return Project.from_dict(await self.http.apost(_PROJECTS, body))

    # ==================== update ====================

    def update(
        self, project_id: str, *, name: Optional[str] = None,
        description: Optional[str] = None, icon: Optional[str] = None,
    ) -> Project:
        """Rename / re-describe / re-icon a project. Only the fields you pass
        change; ``None`` leaves a field untouched."""
        return Project.from_dict(
            self.http.put(f"{_PROJECTS}/{project_id}", self._update_body(name, description, icon))
        )

    async def aupdate(
        self, project_id: str, *, name: Optional[str] = None,
        description: Optional[str] = None, icon: Optional[str] = None,
    ) -> Project:
        """Async sibling of ``update``."""
        return Project.from_dict(
            await self.http.aput(f"{_PROJECTS}/{project_id}", self._update_body(name, description, icon))
        )

    # ==================== delete ====================

    def delete(self, project_id: str) -> None:
        """Delete a project (needs ``project:delete``). Its agents survive,
        un-projected (``Agent.project`` is SET_NULL). A missing project →
        ``NotFoundError``."""
        self.http.delete(f"{_PROJECTS}/{project_id}")

    async def adelete(self, project_id: str) -> None:
        """Async sibling of ``delete``."""
        await self.http.adelete(f"{_PROJECTS}/{project_id}")

    # ==================== internals ====================

    @staticmethod
    def _params(ordering: Optional[str], page_size: Optional[int]) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if ordering is not None:
            params["ordering"] = ordering
        if page_size is not None:
            params["page_size"] = page_size
        return params or None

    def _page_get(self, base: Optional[Dict[str, Any]], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base or {})
        if cursor:
            params["cursor"] = cursor
        return self.http.get(_PROJECTS, params or None)

    async def _apage_get(self, base: Optional[Dict[str, Any]], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base or {})
        if cursor:
            params["cursor"] = cursor
        return await self.http.aget(_PROJECTS, params or None)

    @staticmethod
    def _create_body(name: str, icon: str, description: Optional[str]) -> Dict[str, Any]:
        # name + icon are required by the backend; description is optional.
        body: Dict[str, Any] = {"name": name, "icon": icon}
        if description is not None:
            body["description"] = description
        return body

    @staticmethod
    def _update_body(
        name: Optional[str], description: Optional[str], icon: Optional[str]
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if icon is not None:
            body["icon"] = icon
        return body
