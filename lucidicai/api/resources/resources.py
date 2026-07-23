"""client.resources — SQL-substrate Resource CRUD (LUC-917).

A new namespace. Resources are the org-scoped definitions of external services the
client mocks at runtime (SQL / API / CUSTOM), attached to Datasets and Tools — the
SQL substrate backing tool-config mocking. Org-scoped: a bound key still reaches
all of the org's resources (agent binding is a no-op here).

Reads and writes are data-bearing — they do NOT swallow in production; typed
transport errors propagate. Every method has an ``a``-prefixed async sibling.
"""
from typing import Any, AsyncIterator, Dict, Iterator, Optional

from ..client import HttpClient
from ..models.base import CursorPage
from ..models.resource import Resource
from ..pagination import apaginate, paginate

_RESOURCES = "sdk/v2/resources"


class ResourcesResource:
    """Handle for the ``/sdk/v2/resources`` endpoints."""

    def __init__(self, http: HttpClient):
        self.http = http

    # ==================== list ====================

    def list(
        self, *, ordering: Optional[str] = None, page_size: Optional[int] = None
    ) -> Iterator[Resource]:
        """Lazily iterate the org's resources, newest first. ``ordering`` accepts
        ``id`` (± prefix). Handy for resolving a ``resource_id`` by name."""
        base = self._params(ordering, page_size)
        return paginate(lambda c: self._page_get(base, c), model=Resource)

    def alist(
        self, *, ordering: Optional[str] = None, page_size: Optional[int] = None
    ) -> AsyncIterator[Resource]:
        """Async sibling of ``list``."""
        base = self._params(ordering, page_size)
        return apaginate(lambda c: self._apage_get(base, c), model=Resource)

    def list_page(
        self, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of resources."""
        return CursorPage.from_body(
            self._page_get(self._params(ordering, page_size), cursor), model=Resource
        )

    async def alist_page(
        self, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``list_page``."""
        body = await self._apage_get(self._params(ordering, page_size), cursor)
        return CursorPage.from_body(body, model=Resource)

    # ==================== get ====================

    def get(self, resource_id: str) -> Resource:
        """Read one resource by id (raises ``NotFoundError`` if absent/cross-org)."""
        return Resource.from_dict(self.http.get(f"{_RESOURCES}/{resource_id}"))

    async def aget(self, resource_id: str) -> Resource:
        """Async sibling of ``get``."""
        return Resource.from_dict(await self.http.aget(f"{_RESOURCES}/{resource_id}"))

    # ==================== create ====================

    def create(
        self, *, type: str, tool_name: str, name: str,
        spec: Optional[Dict[str, Any]] = None, ddl: Optional[str] = None,
        dialect: Optional[str] = None, description: Optional[str] = None,
        hydration_config: Optional[Dict[str, Any]] = None,
    ) -> Resource:
        """Create a resource in the key's org (POST /sdk/v2/resources; needs
        ``resource:write``). ``type`` is ``"SQL"`` / ``"API"`` / ``"CUSTOM"``, and
        ``dialect`` (e.g. ``"POSTGRES"``) is required when ``type == "SQL"``. Supply
        the schema exactly one way — either a parsed ``spec`` dict or raw ``ddl``
        (``CREATE TABLE`` SQL the backend parses into ``spec``); passing both or
        neither → ``ValidationError``. ``name`` and ``tool_name`` are each unique
        per org — a duplicate → ``ConflictError``. The org is taken from the API
        key, never the body."""
        return Resource.from_dict(self.http.post(
            _RESOURCES,
            self._create_body(type, tool_name, name, spec, ddl, dialect, description, hydration_config),
        ))

    async def acreate(
        self, *, type: str, tool_name: str, name: str,
        spec: Optional[Dict[str, Any]] = None, ddl: Optional[str] = None,
        dialect: Optional[str] = None, description: Optional[str] = None,
        hydration_config: Optional[Dict[str, Any]] = None,
    ) -> Resource:
        """Async sibling of ``create``."""
        return Resource.from_dict(await self.http.apost(
            _RESOURCES,
            self._create_body(type, tool_name, name, spec, ddl, dialect, description, hydration_config),
        ))

    # ==================== update ====================

    def update(
        self, resource_id: str, *, name: Optional[str] = None,
        tool_name: Optional[str] = None, description: Optional[str] = None,
        spec: Optional[Dict[str, Any]] = None, ddl: Optional[str] = None,
        hydration_config: Optional[Dict[str, Any]] = None,
    ) -> Resource:
        """Partially update a resource (PUT /sdk/v2/resources/{id}; needs
        ``resource:write``). Send only the fields to change — ``name`` /
        ``tool_name`` / ``description`` / ``hydration_config``, and the schema via a
        new ``spec`` or ``ddl`` (re-parsed into ``spec``). ``type`` and ``dialect``
        are immutable after creation and are intentionally not accepted here. A
        rename / ``tool_name`` collision → ``ConflictError``."""
        return Resource.from_dict(self.http.put(
            f"{_RESOURCES}/{resource_id}",
            self._update_body(name, tool_name, description, spec, ddl, hydration_config),
        ))

    async def aupdate(
        self, resource_id: str, *, name: Optional[str] = None,
        tool_name: Optional[str] = None, description: Optional[str] = None,
        spec: Optional[Dict[str, Any]] = None, ddl: Optional[str] = None,
        hydration_config: Optional[Dict[str, Any]] = None,
    ) -> Resource:
        """Async sibling of ``update``."""
        return Resource.from_dict(await self.http.aput(
            f"{_RESOURCES}/{resource_id}",
            self._update_body(name, tool_name, description, spec, ddl, hydration_config),
        ))

    # ==================== delete ====================

    def delete(self, resource_id: str) -> None:
        """Delete a resource (DELETE /sdk/v2/resources/{id}; needs
        ``resource:delete``). A resource still attached to any Dataset, Fixture, or
        Tool is protected → ``ConflictError`` (detach it there first). A missing /
        cross-org id → ``NotFoundError``."""
        self.http.delete(f"{_RESOURCES}/{resource_id}")

    async def adelete(self, resource_id: str) -> None:
        """Async sibling of ``delete``."""
        await self.http.adelete(f"{_RESOURCES}/{resource_id}")

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
        return self.http.get(_RESOURCES, params or None)

    async def _apage_get(self, base: Optional[Dict[str, Any]], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base or {})
        if cursor:
            params["cursor"] = cursor
        return await self.http.aget(_RESOURCES, params or None)

    @staticmethod
    def _create_body(
        type_: str, tool_name: str, name: str, spec: Optional[Dict[str, Any]],
        ddl: Optional[str], dialect: Optional[str], description: Optional[str],
        hydration_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # type/tool_name/name are always required; the backend enforces the
        # spec/ddl XOR and the dialect-required-when-SQL rule, so we just forward
        # what was provided (omit-None) and let it validate.
        body: Dict[str, Any] = {"type": type_, "tool_name": tool_name, "name": name}
        if spec is not None:
            body["spec"] = spec
        if ddl is not None:
            body["ddl"] = ddl
        if dialect is not None:
            body["dialect"] = dialect
        if description is not None:
            body["description"] = description
        if hydration_config is not None:
            body["hydration_config"] = hydration_config
        return body

    @staticmethod
    def _update_body(
        name: Optional[str], tool_name: Optional[str], description: Optional[str],
        spec: Optional[Dict[str, Any]], ddl: Optional[str],
        hydration_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # omit-None: PUT is a partial update — send only the fields being changed.
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if tool_name is not None:
            body["tool_name"] = tool_name
        if description is not None:
            body["description"] = description
        if spec is not None:
            body["spec"] = spec
        if ddl is not None:
            body["ddl"] = ddl
        if hydration_config is not None:
            body["hydration_config"] = hydration_config
        return body
