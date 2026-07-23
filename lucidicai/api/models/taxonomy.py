"""Typed response models for client.experiments.taxonomy (LUC-922)."""
from dataclasses import dataclass
from typing import Any, Optional

from .base import APIModel


@dataclass
class TaxonomyRun(APIModel):
    """A trace-taxonomy run for an experiment. The trigger / ongoing shape carries
    ``run_id`` / ``version`` / ``status`` / ``created_at``; the completed read
    (``taxonomy.get``) adds ``completed_at`` and the ``taxonomy`` dimensions output.
    ``status`` is ongoing (``queued`` / ``sampling`` / ``discovering_dimensions`` /
    ... ) or terminal (``completed`` / ``failed``)."""

    run_id: str
    version: Optional[int] = None
    status: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    taxonomy: Optional[Any] = None


@dataclass
class TaxonomyStatus(APIModel):
    """The poll target for taxonomy generation (``GET .../taxonomy/status``): the
    experiment's in-flight run (``ongoing``, null when none) and its latest completed
    taxonomy (``latest_completed``, null until one finishes). Terminal once no run is
    ongoing — a run either completes (into ``latest_completed``) or fails (clears
    ``ongoing`` without updating ``latest_completed``)."""

    ongoing: Optional[TaxonomyRun] = None
    latest_completed: Optional[TaxonomyRun] = None
    evaluating_session_count: Optional[int] = None

    @classmethod
    def from_dict(cls, data):
        obj = super().from_dict(data)
        # The two nested run objects arrive as raw dicts — type them.
        for attr in ("ongoing", "latest_completed"):
            value = getattr(obj, attr, None)
            if isinstance(value, dict):
                setattr(obj, attr, TaxonomyRun.from_dict(value))
        return obj

    @property
    def is_terminal(self) -> bool:
        """True once no taxonomy run is in progress (the triggered run finished —
        successfully into ``latest_completed`` or by failing out of ``ongoing``)."""
        return self.ongoing is None
