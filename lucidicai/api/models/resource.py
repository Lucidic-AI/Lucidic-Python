"""Typed response model for client.resources (LUC-917)."""
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import APIModel


@dataclass
class Resource(APIModel):
    """A SQL-substrate Resource — the org-scoped definition of an external service
    the client mocks at runtime (``type`` ``SQL`` / ``API`` / ``CUSTOM``). Authored
    once per org and attached to Datasets and Tools; ``spec`` is the parsed schema
    that backs fixture hydration + runtime mocking. Org-scoped, so a bound key still
    sees all of the org's resources.

    ``name`` and ``tool_name`` are each unique per org. ``handler_type`` is derived
    from ``type`` by the backend (read-only). The write-only ``ddl`` create input
    is parsed into ``spec`` server-side and never returned.
    """

    resource_id: str
    type: Optional[str] = None
    dialect: Optional[str] = None
    tool_name: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    spec: Optional[Dict[str, Any]] = None
    handler_type: Optional[str] = None
    hydration_config: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
