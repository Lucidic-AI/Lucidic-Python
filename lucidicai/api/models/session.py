"""Typed response models for client.sessions v2 reads (LUC-907)."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import APIModel


@dataclass
class Event(APIModel):
    """One event in a session's trace.

    The list/trace (preview) shape omits ``payload``; the single-event detail
    (``client.sessions.event``) includes ``payload`` and, for an offloaded
    payload requested with ``raw=True``, a short-lived presigned ``blob_url``.
    ``children`` is populated by ``SessionTrace.tree()`` — not sent by the
    backend.
    """

    event_id: str
    session_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    type: Optional[str] = None
    created_at: Optional[str] = None
    occurred_at: Optional[str] = None
    duration: Optional[float] = None
    cost: Optional[float] = None
    summary: Optional[str] = None
    function_name: Optional[str] = None
    model_name: Optional[str] = None
    provider: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    num_time_travels: Optional[int] = None
    has_blob: bool = False
    payload: Optional[Any] = None       # detail only
    blob_url: Optional[str] = None      # raw=True + offloaded payload
    children: List["Event"] = field(default_factory=list, repr=False)  # from tree()


@dataclass
class Session(APIModel):
    """A session — a full run/trace of an agent execution. Returned by
    ``client.sessions.list()`` and as the ``.session`` of a detail (``get``)."""

    session_id: str
    custom_session_id: Optional[str] = None
    name: Optional[str] = None
    start_time: Optional[str] = None
    duration: Optional[float] = None
    is_finished: bool = False
    production_monitoring: bool = False
    task: Optional[str] = None
    cost: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    eval_previews: List[Dict[str, Any]] = field(default_factory=list)
    num_events: Optional[int] = None
    status: Optional[str] = None
    datasetitem_id: Optional[str] = None


@dataclass
class SessionTrace(APIModel):
    """A session detail: session meta + its flat, ``occurred_at``-ordered event
    trace. Rebuild the parent/child tree with ``tree()`` (or group with
    ``children_by_parent()``)."""

    session: Session
    events: List[Event] = field(default_factory=list)
    num_events: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionTrace":
        data = data if isinstance(data, dict) else {}
        obj = cls(
            session=Session.from_dict(data.get("session") or {}),
            events=Event.from_list(data.get("events") or []),
            num_events=data.get("num_events"),
        )
        object.__setattr__(obj, "_extra", {
            k: v for k, v in data.items()
            if k not in ("session", "events", "num_events")
        })
        return obj

    def children_by_parent(self) -> Dict[Optional[str], List[Event]]:
        """Group events by parent. Root events — no parent, a parent absent from
        this event set (e.g. a purged or truncated-out parent), or a
        self-reference — are bucketed under the ``None`` key so nothing is
        silently dropped. Preserves the backend's ``occurred_at`` ordering.
        """
        known = {ev.event_id for ev in self.events}
        by_parent: Dict[Optional[str], List[Event]] = {}
        for ev in self.events:
            pid = ev.parent_event_id
            if pid is None or pid == ev.event_id or pid not in known:
                pid = None  # true root, orphan, or self-cycle -> treat as root
            by_parent.setdefault(pid, []).append(ev)
        return by_parent

    def tree(self) -> List[Event]:
        """Rebuild the trace tree from ``parent_event_id``: returns the root
        events, each with its ``children`` linked. Orphan events (parent not in
        the returned set) surface as roots rather than vanishing."""
        by_parent = self.children_by_parent()
        for ev in self.events:
            ev.children = by_parent.get(ev.event_id, [])
        return by_parent.get(None, [])


@dataclass
class EvalResult(APIModel):
    """One session-level evaluator result."""

    eval_id: str
    evaluator_id: Optional[str] = None
    evaluator_name: Optional[str] = None
    evaluator_type: Optional[str] = None
    result: Optional[Any] = None
    result_type: Optional[str] = None
    description: Optional[str] = None
    is_pending: bool = False
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    submitted_by: Optional[str] = None
    evaluated_at: Optional[str] = None
    last_updated: Optional[str] = None
    created_at: Optional[str] = None
    session_id: Optional[str] = None
    criteria_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EventEval(APIModel):
    """One event-level eval mapping (``NewEventEval``)."""

    eval_id: str
    event_id: Optional[str] = None
    session_id: Optional[str] = None
    criteria_name: Optional[str] = None
    result: Optional[Any] = None
    description: Optional[str] = None
    lucidic_created: bool = False
    evaluator_failed: bool = False
    criteria_failed: bool = False
    last_updated: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class SessionEvaluatorResults(APIModel):
    """A session's eval scores: session-level ``evals`` + event-level
    ``event_evals``. Returned by ``client.sessions.evaluator_results``."""

    evals: List[EvalResult] = field(default_factory=list)
    event_evals: List[EventEval] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionEvaluatorResults":
        data = data if isinstance(data, dict) else {}
        obj = cls(
            evals=EvalResult.from_list(data.get("evals") or []),
            event_evals=EventEval.from_list(data.get("event_evals") or []),
        )
        object.__setattr__(obj, "_extra", {
            k: v for k, v in data.items() if k not in ("evals", "event_evals")
        })
        return obj
