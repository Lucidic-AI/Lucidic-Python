"""Serialization utilities for event payloads."""
import json
from typing import Any
from collections.abc import Iterable


def serialize_value(value: Any) -> Any:
    """Serialize a value to JSON-compatible format.

    Recursively converts complex objects to JSON-serializable types.

    Args:
        value: Any value to serialize

    Returns:
        JSON-compatible representation of the value
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [serialize_value(v) for v in value]
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)
