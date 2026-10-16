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


# Terminal generation statuses — the run is done (successfully or not) and polling
# should stop. Everything else (queued / planning / hydrating_fixtures / generating
# / ...) is ongoing. Mirrors the backend's DATASET_GEN_TERMINAL_STATUSES.
_TERMINAL_GENERATION_STATUSES = frozenset({"completed", "failed"})


@dataclass
class DatasetGenerationRun(APIModel):
    """A dataset-generation run — the async pipeline that fills a Dataset from a
    schema + the experiment's trace taxonomy. The trigger / retry response populates
    ``run_id`` / ``dataset_id`` / ``status`` / ``created_at``; the status endpoint
    additionally reports progress (``items_generated`` / ``total_target``),
    ``error_message`` (set when failed), and the source experiment / schema.

    ``status`` is ongoing (``queued`` / ``planning`` / ``hydrating_fixtures`` /
    ``generating`` / ...) or terminal (``completed`` = success, ``failed`` = read
    ``error_message``). Use ``is_terminal`` / ``succeeded`` rather than comparing
    the raw string.
    """

    run_id: str
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    status: Optional[str] = None
    items_generated: Optional[int] = None
    total_target: Optional[int] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    schema_id: Optional[str] = None
    schema_name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

    @property
    def is_terminal(self) -> bool:
        """True once the run has finished (``completed`` or ``failed``)."""
        return self.status in _TERMINAL_GENERATION_STATUSES

    @property
    def succeeded(self) -> bool:
        """True only for a ``completed`` run (a ``failed`` run is terminal but not
        successful — read ``error_message``)."""
        return self.status == "completed"
