import os
import logging
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

from .client import Client
from .errors import APIKeyVerificationError

logger = logging.getLogger("Lucidic")


def get_dataset(
    dataset_id: str,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get a dataset by ID with all its items.
    
    Args:
        dataset_id: The ID of the dataset to retrieve (required).
        api_key: API key for authentication. If not provided, will use the LUCIDIC_API_KEY environment variable.
        agent_id: Agent ID. If not provided, will use the LUCIDIC_AGENT_ID environment variable.
    
    Returns:
        A dictionary containing the dataset information including:
        - dataset_id: The dataset ID
        - name: Dataset name
        - description: Dataset description
        - tags: List of tags
        - created_at: Creation timestamp
        - updated_at: Last update timestamp
        - num_items: Number of items in the dataset
        - items: List of dataset items
    
    Raises:
        APIKeyVerificationError: If API key or agent ID is missing or invalid.
        ValueError: If dataset_id is not provided.
    """
    return # no op for now
    load_dotenv()
    
    # Validation
    if not dataset_id:
        raise ValueError("Dataset ID is required")
    
    # Get credentials
    if api_key is None:
        api_key = os.getenv("LUCIDIC_API_KEY", None)
        if api_key is None:
            raise APIKeyVerificationError(
                "Make sure to either pass your API key into get_dataset() or set the LUCIDIC_API_KEY environment variable."
            )
    
    if agent_id is None:
        agent_id = os.getenv("LUCIDIC_AGENT_ID", None)
        if agent_id is None:
            raise APIKeyVerificationError(
                "Lucidic agent ID not specified. Make sure to either pass your agent ID into get_dataset() or set the LUCIDIC_AGENT_ID environment variable."
            )
    
    # Get current client or create a new one
    client = Client()
    # If not yet initialized or still the NullClient -> create a real client
    if not getattr(client, 'initialized', False):
        client = Client(api_key=api_key, agent_id=agent_id)
    else:
        # Already initialized, check if we need to update credentials
        if api_key is not None and agent_id is not None and (api_key != client.api_key or agent_id != client.agent_id):
            client.set_api_key(api_key)
            client.agent_id = agent_id
    
    # Make request to get dataset
    response = client.make_request(
        'getdataset',
        'GET',
        {'dataset_id': dataset_id}
    )
    
    logger.info(f"Retrieved dataset {dataset_id} with {response.get('num_items', 0)} items")
    return response


def get_dataset_items(
    dataset_id: str,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to get just the items from a dataset.
    
    Args:
        dataset_id: The ID of the dataset to retrieve items from (required).
        api_key: API key for authentication. If not provided, will use the LUCIDIC_API_KEY environment variable.
        agent_id: Agent ID. If not provided, will use the LUCIDIC_AGENT_ID environment variable.
    
    Returns:
        A list of dataset items, where each item contains:
        - dataset_item_id: The item ID
        - name: Item name
        - description: Item description
        - tags: List of tags
        - input: Input data for the item
        - expected_output: Expected output data
        - metadata: Additional metadata
        - created_at: Creation timestamp
    
    Raises:
        APIKeyVerificationError: If API key or agent ID is missing or invalid.
        ValueError: If dataset_id is not provided.
    """
    return # no op for now
    dataset = get_dataset(dataset_id, api_key, agent_id)
    return dataset.get('items', [])