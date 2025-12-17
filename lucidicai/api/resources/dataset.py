"""Dataset resource API operations."""
import logging
from typing import Any, Dict, List, Optional

from ..client import HttpClient

logger = logging.getLogger("Lucidic")


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
