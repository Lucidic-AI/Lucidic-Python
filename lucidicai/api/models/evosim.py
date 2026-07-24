"""Typed response models for client.evosims run management (LUC-923)."""
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .base import APIModel


# A run's terminal statuses, for polling. NOTE this INCLUDES ``CANCELED`` — unlike
# the backend's internal ``_TERMINAL_EVOSIM_STATUSES`` constant, which omits it (that
# one exists only so the cancel path won't relabel an already-finished run). A
# null/absent status means "iteration not yet materialized" — keep polling, not
# terminal.
_TERMINAL_EVOSIM_STATUSES = frozenset({"SUCCEEDED", "PARTIAL_SUCCESS", "FAILED", "CANCELED"})


@dataclass
class EvoSimIteration(APIModel):
    """One optimization cycle of an EvoSim run, with its own rolled-up ``status``."""

    evosimiteration_id: str
    evosim_id: Optional[str] = None
    iteration: Optional[int] = None
    status: Optional[str] = None
    failure_reason: Optional[str] = None
    is_finished: Optional[bool] = None
    hard_stop: Optional[Any] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class EvoSim(APIModel):
    """An EvoSim run (an evolutionary optimization of an agent). ``list`` returns the
    light shape; ``get`` additionally rolls up the run ``status`` / ``failure_reason``,
    the derived ``checkpoint_id`` (the checkpoint the run produced — null until one is
    published, and always null for a FAILED / no-ready run), and its ``iterations``.

    ``status`` is ``RUNNING`` (ongoing) / ``SUCCEEDED`` / ``PARTIAL_SUCCESS`` /
    ``FAILED`` / ``CANCELED`` (terminal), or ``None`` before the first iteration
    exists. Use ``is_terminal`` rather than comparing the raw string.
    """

    evosim_id: str
    agent_id: Optional[str] = None
    experiment_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    max_iterations: Optional[int] = None
    hard_stop_seconds_per_iteration: Optional[int] = None
    max_session_concurrency: Optional[int] = None
    webhook_url: Optional[str] = None
    temporal_workflow_id: Optional[str] = None
    temporal_run_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    # detail-only (get): the rolled-up run status + produced checkpoint + iterations.
    status: Optional[str] = None
    failure_reason: Optional[str] = None
    checkpoint_id: Optional[str] = None
    iterations: List[EvoSimIteration] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data):
        obj = super().from_dict(data)
        if obj.iterations and isinstance(obj.iterations[0], dict):
            setattr(obj, "iterations", EvoSimIteration.from_list(obj.iterations))
        return obj

    @property
    def is_terminal(self) -> bool:
        """True once the run has finished (``SUCCEEDED`` / ``PARTIAL_SUCCESS`` /
        ``FAILED`` / ``CANCELED``). A ``None`` status (no iteration yet) is not
        terminal — keep polling."""
        return self.status in _TERMINAL_EVOSIM_STATUSES


@dataclass
class TrainingModuleInstance(APIModel):
    """One Training Module Instance (TMI) — a module's run within an EvoSim iteration.
    ``status`` is ``BUILDING`` (ongoing) / ``READY`` / ``FAILED`` / ``ARCHIVED``."""

    tmi_id: str
    module_key: Optional[str] = None
    module_artifact_key: Optional[str] = None
    created_by: Optional[str] = None
    status: Optional[str] = None
    session_id: Optional[str] = None
    created_at: Optional[str] = None
