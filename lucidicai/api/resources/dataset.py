"""Dataset resource API operations."""
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from ..client import HttpClient
from ..models.base import CursorPage
from ..models.dataset import DatasetItem, DatasetSchema, Fixture
from ..pagination import apaginate, paginate

logger = logging.getLogger("Lucidic")

_SCHEMAS = "sdk/v2/datasets/schemas"
_FIXTURES = "sdk/v2/fixtures"


class DatasetResource:
    """Handle dataset-related API operations."""

    def __init__(
        self,
        http: HttpClient,
        agent_id: Optional[str] = None,
        production: bool = False,
    ):
        """Initialize dataset resource.

        Args:
            http: HTTP client instance
            agent_id: Default agent ID for datasets
            production: Whether to suppress errors in production mode
        """
        self.http = http
        self._agent_id = agent_id
        self._production = production
        # v2 sub-namespaces (LUC-918): client.datasets.schemas / .fixtures.
        # Stateless (ids passed per call), so one instance each is reused.
        self._schemas = DatasetSchemasResource(http)
        self._fixtures = DatasetFixturesResource(http)

    # ==================== v2 surface (LUC-918) ====================
    #
    # Data-bearing v2 reads/writes: they do NOT swallow in production (typed
    # transport errors propagate), unlike the gen-3 methods below. ``items`` is a
    # cursor-paginated read of a dataset's items; ``schemas`` / ``fixtures`` are
    # sub-namespaces. Each has an async sibling.

    @property
    def schemas(self) -> "DatasetSchemasResource":
        """Dataset schema CRUD — ``client.datasets.schemas.list()`` /
        ``create`` / ``get`` / ``update`` / ``delete`` (org-scoped typed
        definitions of a dataset item's ``input`` shape)."""
        return self._schemas

    @property
    def fixtures(self) -> "DatasetFixturesResource":
        """Fixture authoring — ``client.datasets.fixtures.create(...)`` builds a
        base DuckDB fixture for a ``(dataset, resource)`` pair from JSON rows."""
        return self._fixtures

    def items(self, dataset_id: str, *, page_size: Optional[int] = None) -> Iterator[DatasetItem]:
        """Lazily iterate a dataset's items, newest first (GET
        /sdk/v2/datasets/{id}/items; needs ``dataset-item:read``). Cursor-paginated
        — transparently walks every page. Unknown / cross-org dataset →
        ``NotFoundError`` on first iteration."""
        base = {"page_size": page_size} if page_size is not None else {}
        path = f"sdk/v2/datasets/{dataset_id}/items"
        return paginate(lambda c: self._v2_page_get(path, base, c), model=DatasetItem)

    def aitems(self, dataset_id: str, *, page_size: Optional[int] = None) -> AsyncIterator[DatasetItem]:
        """Async sibling of ``items``."""
        base = {"page_size": page_size} if page_size is not None else {}
        path = f"sdk/v2/datasets/{dataset_id}/items"
        return apaginate(lambda c: self._v2_apage_get(path, base, c), model=DatasetItem)

    def items_page(
        self, dataset_id: str, *, cursor: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of a dataset's items."""
        base = {"page_size": page_size} if page_size is not None else {}
        body = self._v2_page_get(f"sdk/v2/datasets/{dataset_id}/items", base, cursor)
        return CursorPage.from_body(body, model=DatasetItem)

    async def aitems_page(
        self, dataset_id: str, *, cursor: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``items_page``."""
        base = {"page_size": page_size} if page_size is not None else {}
        body = await self._v2_apage_get(f"sdk/v2/datasets/{dataset_id}/items", base, cursor)
        return CursorPage.from_body(body, model=DatasetItem)

    def _v2_page_get(self, path: str, base: Dict[str, Any], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base)
        if cursor:
            params["cursor"] = cursor
        return self.http.get(path, params or None)

    async def _v2_apage_get(self, path: str, base: Dict[str, Any], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base)
        if cursor:
            params["cursor"] = cursor
        return await self.http.aget(path, params or None)

    # ==================== Dataset Methods ====================

    def list(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """List all datasets for agent.

        Args:
            agent_id: Optional agent ID to filter by (uses default if not provided)

        Returns:
            Dictionary with num_datasets and datasets list
        """
        try:
            params = {}
            if agent_id or self._agent_id:
                params["agent_id"] = agent_id or self._agent_id
            return self.http.get("sdk/datasets", params)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to list datasets: {e}")
                return {"num_datasets": 0, "datasets": []}
            raise

    def create(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        suggested_flag_config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create new dataset.

        Args:
            name: Dataset name (must be unique per agent)
            description: Optional description
            tags: Optional list of tags
            suggested_flag_config: Optional flag configuration
            agent_id: Optional agent ID (uses default if not provided)

        Returns:
            Dictionary with dataset_id
        """
        try:
            data: Dict[str, Any] = {"name": name}
            if description is not None:
                data["description"] = description
            if tags is not None:
                data["tags"] = tags
            if suggested_flag_config is not None:
                data["suggested_flag_config"] = suggested_flag_config
            data["agent_id"] = agent_id or self._agent_id
            return self.http.post("sdk/datasets/create", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to create dataset: {e}")
                return {}
            raise

    def get(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset with all items.

        Args:
            dataset_id: Dataset UUID

        Returns:
            Full dataset data including all items
        """
        try:
            return self.http.get("getdataset", {"dataset_id": dataset_id})
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to get dataset: {e}")
                return {}
            raise

    def update(self, dataset_id: str, **kwargs) -> Dict[str, Any]:
        """Update dataset metadata.

        Args:
            dataset_id: Dataset UUID
            **kwargs: Fields to update (name, description, tags, suggested_flag_config)

        Returns:
            Updated dataset data
        """
        try:
            data = {"dataset_id": dataset_id}
            data.update(kwargs)
            return self.http.put("sdk/datasets/update", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to update dataset: {e}")
                return {}
            raise

    def delete(self, dataset_id: str) -> Dict[str, Any]:
        """Delete dataset and all items.

        Args:
            dataset_id: Dataset UUID

        Returns:
            Success message
        """
        try:
            return self.http.delete("sdk/datasets/delete", {"dataset_id": dataset_id})
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to delete dataset: {e}")
                return {}
            raise

    # ==================== Dataset Item Methods ====================

    def create_item(
        self,
        dataset_id: str,
        name: str,
        input_data: Dict[str, Any],
        expected_output: Optional[Any] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        flag_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create dataset item.

        Args:
            dataset_id: Dataset UUID
            name: Item name
            input_data: Input data dictionary
            expected_output: Optional expected output
            description: Optional description
            tags: Optional list of tags
            metadata: Optional metadata dictionary
            flag_overrides: Optional flag overrides

        Returns:
            Dictionary with datasetitem_id
        """
        try:
            data: Dict[str, Any] = {
                "dataset_id": dataset_id,
                "name": name,
                "input": input_data
            }

            if expected_output is not None:
                data["expected_output"] = expected_output
            if description is not None:
                data["description"] = description
            if tags is not None:
                data["tags"] = tags
            if metadata is not None:
                data["metadata"] = metadata
            if flag_overrides is not None:
                data["flag_overrides"] = flag_overrides

            return self.http.post("sdk/datasets/items/create", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to create item: {e}")
                return {}
            raise

    def get_item(self, dataset_id: str, item_id: str) -> Dict[str, Any]:
        """Get specific dataset item.

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Dataset item data
        """
        try:
            return self.http.get("sdk/datasets/items/get", {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            })
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to get item: {e}")
                return {}
            raise

    def update_item(self, dataset_id: str, item_id: str, **kwargs) -> Dict[str, Any]:
        """Update dataset item.

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID
            **kwargs: Fields to update

        Returns:
            Updated item data
        """
        try:
            data = {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            }
            data.update(kwargs)
            return self.http.put("sdk/datasets/items/update", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to update item: {e}")
                return {}
            raise

    def delete_item(self, dataset_id: str, item_id: str) -> Dict[str, Any]:
        """Delete dataset item.

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Success message
        """
        try:
            return self.http.delete("sdk/datasets/items/delete", {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            })
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to delete item: {e}")
                return {}
            raise

    def list_item_sessions(self, dataset_id: str, item_id: str) -> Dict[str, Any]:
        """List all sessions for a dataset item.

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Dictionary with num_sessions and sessions list
        """
        try:
            return self.http.get("sdk/datasets/items/sessions", {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            })
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to list item sessions: {e}")
                return {"num_sessions": 0, "sessions": []}
            raise

    # ==================== Asynchronous Dataset Methods ====================

    async def alist(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """List all datasets for agent (asynchronous).

        Args:
            agent_id: Optional agent ID to filter by (uses default if not provided)

        Returns:
            Dictionary with num_datasets and datasets list
        """
        try:
            params = {}
            if agent_id or self._agent_id:
                params["agent_id"] = agent_id or self._agent_id
            return await self.http.aget("sdk/datasets", params)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to list datasets: {e}")
                return {"num_datasets": 0, "datasets": []}
            raise

    async def acreate(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        suggested_flag_config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create new dataset (asynchronous).

        Args:
            name: Dataset name (must be unique per agent)
            description: Optional description
            tags: Optional list of tags
            suggested_flag_config: Optional flag configuration
            agent_id: Optional agent ID (uses default if not provided)

        Returns:
            Dictionary with dataset_id
        """
        try:
            data: Dict[str, Any] = {"name": name}
            if description is not None:
                data["description"] = description
            if tags is not None:
                data["tags"] = tags
            if suggested_flag_config is not None:
                data["suggested_flag_config"] = suggested_flag_config
            data["agent_id"] = agent_id or self._agent_id
            return await self.http.apost("sdk/datasets/create", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to create dataset: {e}")
                return {}
            raise

    async def aget(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset with all items (asynchronous).

        Args:
            dataset_id: Dataset UUID

        Returns:
            Full dataset data including all items
        """
        try:
            return await self.http.aget("getdataset", {"dataset_id": dataset_id})
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to get dataset: {e}")
                return {}
            raise

    async def aupdate(self, dataset_id: str, **kwargs) -> Dict[str, Any]:
        """Update dataset metadata (asynchronous).

        Args:
            dataset_id: Dataset UUID
            **kwargs: Fields to update (name, description, tags, suggested_flag_config)

        Returns:
            Updated dataset data
        """
        try:
            data = {"dataset_id": dataset_id}
            data.update(kwargs)
            return await self.http.aput("sdk/datasets/update", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to update dataset: {e}")
                return {}
            raise

    async def adelete(self, dataset_id: str) -> Dict[str, Any]:
        """Delete dataset and all items (asynchronous).

        Args:
            dataset_id: Dataset UUID

        Returns:
            Success message
        """
        try:
            return await self.http.adelete("sdk/datasets/delete", {"dataset_id": dataset_id})
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to delete dataset: {e}")
                return {}
            raise

    # ==================== Asynchronous Item Methods ====================

    async def acreate_item(
        self,
        dataset_id: str,
        name: str,
        input_data: Dict[str, Any],
        expected_output: Optional[Any] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        flag_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create dataset item (asynchronous).

        Args:
            dataset_id: Dataset UUID
            name: Item name
            input_data: Input data dictionary
            expected_output: Optional expected output
            description: Optional description
            tags: Optional list of tags
            metadata: Optional metadata dictionary
            flag_overrides: Optional flag overrides

        Returns:
            Dictionary with datasetitem_id
        """
        try:
            data: Dict[str, Any] = {
                "dataset_id": dataset_id,
                "name": name,
                "input": input_data
            }

            if expected_output is not None:
                data["expected_output"] = expected_output
            if description is not None:
                data["description"] = description
            if tags is not None:
                data["tags"] = tags
            if metadata is not None:
                data["metadata"] = metadata
            if flag_overrides is not None:
                data["flag_overrides"] = flag_overrides

            return await self.http.apost("sdk/datasets/items/create", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to create item: {e}")
                return {}
            raise

    async def aget_item(self, dataset_id: str, item_id: str) -> Dict[str, Any]:
        """Get specific dataset item (asynchronous).

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Dataset item data
        """
        try:
            return await self.http.aget("sdk/datasets/items/get", {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            })
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to get item: {e}")
                return {}
            raise

    async def aupdate_item(self, dataset_id: str, item_id: str, **kwargs) -> Dict[str, Any]:
        """Update dataset item (asynchronous).

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID
            **kwargs: Fields to update

        Returns:
            Updated item data
        """
        try:
            data = {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            }
            data.update(kwargs)
            return await self.http.aput("sdk/datasets/items/update", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to update item: {e}")
                return {}
            raise

    async def adelete_item(self, dataset_id: str, item_id: str) -> Dict[str, Any]:
        """Delete dataset item (asynchronous).

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Success message
        """
        try:
            return await self.http.adelete("sdk/datasets/items/delete", {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            })
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to delete item: {e}")
                return {}
            raise

    async def alist_item_sessions(self, dataset_id: str, item_id: str) -> Dict[str, Any]:
        """List all sessions for a dataset item (asynchronous).

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Dictionary with num_sessions and sessions list
        """
        try:
            return await self.http.aget("sdk/datasets/items/sessions", {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            })
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to list item sessions: {e}")
                return {"num_sessions": 0, "sessions": []}
            raise


class DatasetSchemasResource:
    """``client.datasets.schemas`` — dataset schema CRUD (LUC-918).

    A dataset schema is an org-scoped typed definition of a dataset item's ``input``
    shape (required for dataset generation). Org-scoped: a bound key reaches them
    all. Data-bearing — no production swallow. Each method has an async sibling.
    """

    def __init__(self, http: HttpClient):
        self.http = http

    def list(
        self, *, ordering: Optional[str] = None, page_size: Optional[int] = None
    ) -> Iterator[DatasetSchema]:
        """Lazily iterate the org's dataset schemas, newest first. ``ordering``
        accepts ``id`` (± prefix)."""
        base = self._params(ordering, page_size)
        return paginate(lambda c: self._page_get(base, c), model=DatasetSchema)

    def alist(
        self, *, ordering: Optional[str] = None, page_size: Optional[int] = None
    ) -> AsyncIterator[DatasetSchema]:
        """Async sibling of ``list``."""
        base = self._params(ordering, page_size)
        return apaginate(lambda c: self._apage_get(base, c), model=DatasetSchema)

    def list_page(
        self, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of dataset schemas."""
        return CursorPage.from_body(
            self._page_get(self._params(ordering, page_size), cursor), model=DatasetSchema
        )

    async def alist_page(
        self, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``list_page``."""
        body = await self._apage_get(self._params(ordering, page_size), cursor)
        return CursorPage.from_body(body, model=DatasetSchema)

    def get(self, schema_id: str) -> DatasetSchema:
        """Read one schema by id (raises ``NotFoundError`` if absent/cross-org)."""
        return DatasetSchema.from_dict(self.http.get(f"{_SCHEMAS}/{schema_id}"))

    async def aget(self, schema_id: str) -> DatasetSchema:
        """Async sibling of ``get``."""
        return DatasetSchema.from_dict(await self.http.aget(f"{_SCHEMAS}/{schema_id}"))

    def create(
        self, name: str, *, fields: List[Dict[str, Any]], description: Optional[str] = None
    ) -> DatasetSchema:
        """Create a dataset schema in the key's org (POST /sdk/v2/datasets/schemas;
        needs ``dataset-schema:write``). ``fields`` is a non-empty list of typed
        field defs — each ``{key, type, description?, required?, options?,
        children?}`` where ``type`` is ``string`` / ``number`` / ``categorical`` /
        ``object`` (``categorical`` needs ``options``; ``object`` needs
        ``children``). ``name`` is unique per org — a duplicate → ``ConflictError``;
        a malformed field → ``ValidationError``."""
        return DatasetSchema.from_dict(
            self.http.post(_SCHEMAS, self._create_body(name, fields, description))
        )

    async def acreate(
        self, name: str, *, fields: List[Dict[str, Any]], description: Optional[str] = None
    ) -> DatasetSchema:
        """Async sibling of ``create``."""
        return DatasetSchema.from_dict(
            await self.http.apost(_SCHEMAS, self._create_body(name, fields, description))
        )

    def update(
        self, schema_id: str, *, name: Optional[str] = None,
        description: Optional[str] = None, fields: Optional[List[Dict[str, Any]]] = None,
    ) -> DatasetSchema:
        """Partially update a schema (PUT /sdk/v2/datasets/schemas/{id}; needs
        ``dataset-schema:write``). Send only the fields to change. A rename
        collision → ``ConflictError``."""
        return DatasetSchema.from_dict(
            self.http.put(f"{_SCHEMAS}/{schema_id}", self._update_body(name, description, fields))
        )

    async def aupdate(
        self, schema_id: str, *, name: Optional[str] = None,
        description: Optional[str] = None, fields: Optional[List[Dict[str, Any]]] = None,
    ) -> DatasetSchema:
        """Async sibling of ``update``."""
        return DatasetSchema.from_dict(
            await self.http.aput(f"{_SCHEMAS}/{schema_id}", self._update_body(name, description, fields))
        )

    def delete(self, schema_id: str) -> None:
        """Delete a schema (DELETE /sdk/v2/datasets/schemas/{id}; needs
        ``dataset-schema:delete``). Datasets/generation runs pointing at it are
        un-linked (their schema pointer is set null), not deleted. A missing /
        cross-org id → ``NotFoundError``."""
        self.http.delete(f"{_SCHEMAS}/{schema_id}")

    async def adelete(self, schema_id: str) -> None:
        """Async sibling of ``delete``."""
        await self.http.adelete(f"{_SCHEMAS}/{schema_id}")

    # ---- internals ----

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
        return self.http.get(_SCHEMAS, params or None)

    async def _apage_get(self, base: Optional[Dict[str, Any]], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base or {})
        if cursor:
            params["cursor"] = cursor
        return await self.http.aget(_SCHEMAS, params or None)

    @staticmethod
    def _create_body(
        name: str, fields: List[Dict[str, Any]], description: Optional[str]
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name, "fields": fields}
        if description is not None:
            body["description"] = description
        return body

    @staticmethod
    def _update_body(
        name: Optional[str], description: Optional[str], fields: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if fields is not None:
            body["fields"] = fields
        return body


class DatasetFixturesResource:
    """``client.datasets.fixtures`` — author base fixtures from JSON rows (LUC-918).

    A fixture is the hydrated mock data (a DuckDB blob) for one ``(dataset,
    resource)`` pair. ``create`` builds one from explicit rows. Data-bearing — no
    production swallow. Has an async sibling.
    """

    def __init__(self, http: HttpClient):
        self.http = http

    def create(
        self, *, resource_id: str, dataset_id: str, tables: List[Dict[str, Any]]
    ) -> Fixture:
        """Author a base fixture from JSON rows (POST /sdk/v2/fixtures; needs
        ``fixture:write``). ``tables`` is a list of ``{"name": <table>, "rows":
        [<row dict>, ...]}`` validated against the resource's ``spec``. Exactly one
        fixture may exist per ``(dataset, resource)`` — a duplicate →
        ``ConflictError``; a row/spec mismatch → ``ValidationError``. Unknown
        resource or dataset → ``NotFoundError``."""
        return Fixture.from_dict(self.http.post(_FIXTURES, self._body(resource_id, dataset_id, tables)))

    async def acreate(
        self, *, resource_id: str, dataset_id: str, tables: List[Dict[str, Any]]
    ) -> Fixture:
        """Async sibling of ``create``."""
        return Fixture.from_dict(
            await self.http.apost(_FIXTURES, self._body(resource_id, dataset_id, tables))
        )

    @staticmethod
    def _body(resource_id: str, dataset_id: str, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"resource_id": resource_id, "dataset_id": dataset_id, "tables": tables}
