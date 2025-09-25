"""Dataset resource API operations."""
from typing import Any, Dict, List, Optional

from ..client import HttpClient


class DatasetResource:
    """Handle dataset-related API operations."""

    def __init__(self, http: HttpClient):
        """Initialize dataset resource.

        Args:
            http: HTTP client instance
        """
        self.http = http

    def list_datasets(self, agent_id=None):
        """List all datasets for agent.

        Args:
            agent_id: Optional agent ID to filter by

        Returns:
            Dictionary with num_datasets and datasets list
        """
        params = {}
        if agent_id:
            params["agent_id"] = agent_id
        return self.http.get("sdk/datasets", params)

    def create_dataset(self, name, description=None, tags=None,
                      suggested_flag_config=None, agent_id=None):
        """Create new dataset.

        Args:
            name: Dataset name (must be unique per agent)
            description: Optional description
            tags: Optional list of tags
            suggested_flag_config: Optional flag configuration
            agent_id: Optional agent ID

        Returns:
            Dictionary with dataset_id
        """
        data = {"name": name}
        if description is not None:
            data["description"] = description
        if tags is not None:
            data["tags"] = tags
        if suggested_flag_config is not None:
            data["suggested_flag_config"] = suggested_flag_config
        if agent_id is not None:
            data["agent_id"] = agent_id
        return self.http.post("sdk/datasets/create", data)

    def get_dataset(self, dataset_id):
        """Get dataset with all items - uses existing endpoint.

        Args:
            dataset_id: Dataset UUID

        Returns:
            Full dataset data including all items
        """
        return self.http.get("getdataset", {"dataset_id": dataset_id})

    def update_dataset(self, dataset_id, **kwargs):
        """Update dataset metadata.

        Args:
            dataset_id: Dataset UUID
            **kwargs: Fields to update (name, description, tags, suggested_flag_config)

        Returns:
            Updated dataset data
        """
        data = {"dataset_id": dataset_id}
        data.update(kwargs)
        return self.http.put("sdk/datasets/update", data)

    def delete_dataset(self, dataset_id):
        """Delete dataset and all items.

        Args:
            dataset_id: Dataset UUID

        Returns:
            Success message
        """
        return self.http.delete("sdk/datasets/delete", {"dataset_id": dataset_id})

    def create_item(self, dataset_id, name, input_data,
                   expected_output=None, description=None,
                   tags=None, metadata=None, flag_overrides=None):
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
        data = {
            "dataset_id": dataset_id,
            "name": name,
            "input": input_data
        }

        # Add optional fields if provided
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

    def get_item(self, dataset_id, item_id):
        """Get specific dataset item.

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Dataset item data
        """
        return self.http.get("sdk/datasets/items/get", {
            "dataset_id": dataset_id,
            "datasetitem_id": item_id
        })

    def update_item(self, dataset_id, item_id, **kwargs):
        """Update dataset item.

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID
            **kwargs: Fields to update

        Returns:
            Updated item data
        """
        data = {
            "dataset_id": dataset_id,
            "datasetitem_id": item_id
        }
        data.update(kwargs)
        return self.http.put("sdk/datasets/items/update", data)

    def delete_item(self, dataset_id, item_id):
        """Delete dataset item.

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Success message
        """
        return self.http.delete("sdk/datasets/items/delete", {
            "dataset_id": dataset_id,
            "datasetitem_id": item_id
        })

    def list_item_sessions(self, dataset_id, item_id):
        """List all sessions for a dataset item.

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Dictionary with num_sessions and sessions list
        """
        return self.http.get("sdk/datasets/items/sessions", {
            "dataset_id": dataset_id,
            "datasetitem_id": item_id
        })