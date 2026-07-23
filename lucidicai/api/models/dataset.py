"""Typed response models for client.datasets v2 (LUC-918).

Distinct from the SDK's grandfathered gen-3 dataset helpers (which return raw
dicts). These model the v2 schemas / items / fixtures surface.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import APIModel


@dataclass
class DatasetSchema(APIModel):
    """A reusable, org-scoped typed definition of the shape of a dataset item's
    ``input`` — a required input to dataset generation. ``fields`` is a list of
    typed field defs (``{key, type, description?, required?, options?, children?}``);
    ``type`` is one of ``string`` / ``number`` / ``categorical`` / ``object``.
    ``name`` is unique per org.
    """

    schema_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    fields: List[Any] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class DatasetItem(APIModel):
    """One test case in a dataset — an ``input`` (matching the dataset's schema)
    and its ``expected_output``, plus tags/metadata. Returned by
    ``client.datasets.items(dataset_id)``."""

    datasetitem_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    tags: List[Any] = field(default_factory=list)
    input: Optional[Dict[str, Any]] = None
    expected_output: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    flag_overrides: Optional[Any] = None
    created_at: Optional[str] = None


@dataclass
class Fixture(APIModel):
    """A hydrated mock-data fixture (a DuckDB blob in object storage) authored for
    one ``(dataset, resource)`` pair. The create response returns its id, storage
    key, byte size, and status (``COMPLETED`` for a row-authored fixture)."""

    fixture_id: str
    blob_key: Optional[str] = None
    byte_size: Optional[int] = None
    status: Optional[str] = None
