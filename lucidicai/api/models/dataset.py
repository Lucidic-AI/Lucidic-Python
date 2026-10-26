"""Typed response models for client.datasets v2 (LUC-918).

Distinct from the SDK's grandfathered gen-3 dataset helpers (which return raw
dicts). These model the v2 schemas / items / fixtures surface, generation runs
(LUC-921) and eval runs (LUC-1195).
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
    ``client.datasets.items(dataset_id)``. ``case_evaluator_id`` is the evaluator
    the case points at (set with ``client.datasets.set_case_evaluator``), or
    ``None``."""

    datasetitem_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    tags: List[Any] = field(default_factory=list)
    input: Optional[Dict[str, Any]] = None
    expected_output: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    flag_overrides: Optional[Any] = None
    created_at: Optional[str] = None
    case_evaluator_id: Optional[str] = None


@dataclass
class CaseGrader(APIModel):
    """The evaluator that will grade a case, as the cases list reports it.
    ``graded_by`` is the step of the grading order that picks it
    (``case_evaluator`` / ``agent_reported`` / ``fallback``); ``scope`` is the
    outcome rows' ``grader_scope`` (``case`` / ``agent`` / ``dataset`` / ``world``).
    ``pass_threshold`` is the mark a number result must reach (``None`` for a
    boolean evaluator). ``criteria`` is the LLM rubric's criteria — ``None`` for a
    non-LLM evaluator, or when the API key lacks ``evaluator:read``."""

    evaluator_id: str
    name: Optional[str] = None
    type: Optional[str] = None
    result_type: Optional[str] = None
    scope: Optional[str] = None
    graded_by: Optional[str] = None
    pass_threshold: Optional[float] = None
    criteria: Optional[List[Any]] = None


@dataclass
class DatasetCase(DatasetItem):
    """A case with what will grade it under the grading order, from the current
    links (``client.datasets.cases``): every ``DatasetItem`` field plus

    - ``scored``: False only for a World's entry item (a way in, not a case);
    - ``graded_by`` / ``grader``: the first step that applies and its evaluator, or
      ``None`` when nothing will grade the case (it reads as ungraded);
    - ``if_not_reported``: for an ``agent_reported`` case, what grades it if the
      agent reports nothing (the Dataset's fallback), or ``None`` — then a silent
      agent leaves the case ungraded.
    """

    scored: bool = True
    graded_by: Optional[str] = None
    grader: Optional[CaseGrader] = None
    if_not_reported: Optional[CaseGrader] = None

    @classmethod
    def from_dict(cls, data):
        obj = super().from_dict(data)
        for key in ("grader", "if_not_reported"):
            value = getattr(obj, key)
            if isinstance(value, dict):
                setattr(obj, key, CaseGrader.from_dict(value))
        return obj


@dataclass
class CaseEvaluatorUpdate(APIModel):
    """The result of pointing cases at a case evaluator, or unlinking them
    (``client.datasets.set_case_evaluator`` / ``remove_case_evaluator``).
    ``evaluator_id`` is the evaluator the cases now point at (``None`` after an
    unlink); ``created`` is True when a spec made a new evaluator rather than reusing
    one; ``item_count`` counts the distinct ids sent and ``changed`` the cases whose
    link actually moved, so a repeated call reads ``changed == 0``."""

    evaluator_id: Optional[str] = None
    created: bool = False
    item_count: int = 0
    changed: int = 0


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


# Terminal eval-run statuses. ``running`` / ``grading`` are ongoing. Mirrors the
# backend's RUN_* constants (api/services/eval_run_outcomes.py).
_TERMINAL_RUN_STATUSES = frozenset({"complete", "failed", "canceled"})


@dataclass
class EvalRun(APIModel):
    """An eval run — one start of a Dataset's cases (all, or the chosen ones) sent to
    the agent ``repeats`` times, then graded and frozen. Either a ``baseline`` or a
    ``candidate`` (one try of an improved version). Returned by
    ``client.datasets.start_run`` and ``runs``.

    The start response carries the run's ids and settings with ``status="running"``;
    a ``runs`` row adds ``created_at`` and, once complete, the compact frozen
    ``summary`` (pass rates per repeat, buckets; ``None`` until then). The run's
    sessions belong to ``experiment_id``; cancel a run with
    ``client.evosims.cancel(run.evosim_id)``.

    ``status`` is ``running`` / ``grading`` (ongoing) or ``complete`` / ``failed`` /
    ``canceled`` (terminal). Use ``is_terminal`` / ``is_complete`` rather than
    comparing the raw string.
    """

    eval_run_id: str
    experiment_id: Optional[str] = None
    evosim_id: Optional[str] = None
    tmi_id: Optional[str] = None
    dataset_id: Optional[str] = None  # None on runs started before Dataset runs existed
    world_id: Optional[str] = None  # set only when the Dataset is a World's
    role: Optional[str] = None
    repeats: Optional[int] = None
    checkpoint_id: Optional[str] = None
    window: Optional[int] = None
    repeat_deadline_seconds: Optional[int] = None
    case_count: Optional[int] = None
    session_count: Optional[int] = None
    status: Optional[str] = None
    # runs() rows only
    created_at: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None

    @property
    def is_terminal(self) -> bool:
        """True once the run has finished (``complete`` / ``failed`` / ``canceled``)."""
        return self.status in _TERMINAL_RUN_STATUSES

    @property
    def is_complete(self) -> bool:
        """True only for a graded and frozen run. Says nothing about its pass rate —
        read ``summary`` for that."""
        return self.status == "complete"
