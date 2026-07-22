"""Typed response model for client.experiments reads (LUC-908)."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import APIModel


@dataclass
class Experiment(APIModel):
    """An experiment — a group of sessions with aggregated metrics/insights.

    ``client.experiments.list()`` returns the light shape (through
    ``eval_metrics`` / ``num_sessions`` / ``tags``); ``get()`` additionally
    populates the detail metrics + failure groups (the fields below default to
    ``None`` / ``[]`` in a list result).
    """

    experiment_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    num_sessions: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    eval_metrics: Optional[Any] = None

    # detail-only (get) — aggregated analytics; kept as raw dicts/lists.
    agent_id: Optional[str] = None
    eval_metrics_by_tag: Optional[Dict[str, Any]] = None
    time_data: Optional[Dict[str, Any]] = None
    time_data_by_tag: Optional[Dict[str, Any]] = None
    cost_data: Optional[Dict[str, Any]] = None
    cost_data_by_tag: Optional[Dict[str, Any]] = None
    num_events_data: Optional[Dict[str, Any]] = None
    num_events_data_by_tag: Optional[Dict[str, Any]] = None
    event_failure_groups: List[Dict[str, Any]] = field(default_factory=list)
    analytics_session_count: Optional[int] = None
