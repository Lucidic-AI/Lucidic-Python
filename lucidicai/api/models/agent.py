"""Typed response models for client.agents (LUC-906)."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import APIModel


@dataclass
class Agent(APIModel):
    """An agent — the top-level entity every SDK call hangs off.

    Returned by ``client.agents.list()`` / ``.get()``. ``agent_id`` is the id an
    SDK-first caller needs to bootstrap (``LucidicAI(agent_id=...)``).
    """

    agent_id: str
    name: Optional[str] = None
    icon: Optional[str] = None
    project_id: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class CatalogTool(APIModel):
    """One tool in an agent's tool catalog, incl. denormalized usage counters
    (``call_count`` / ``last_called_at``, bumped at event ingestion)."""

    tool_id: str
    name: Optional[str] = None
    tier: Optional[str] = None
    signature: Optional[Dict[str, Any]] = None
    docstring: Optional[str] = None
    return_shape: Optional[Any] = None
    source_hash: Optional[str] = None
    drift_acknowledged_hash: Optional[str] = None
    has_drift: bool = False
    impl_body: Optional[str] = None
    impl_entry_point: Optional[str] = None
    last_seen_at: Optional[str] = None
    discovered_at: Optional[str] = None
    call_count: int = 0
    last_called_at: Optional[str] = None
    resources: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentToolCatalog(APIModel):
    """An agent's tool catalog: its tools (typed) plus the deduped, agent-wide
    reachable Resource set.

    ``resources`` is kept as raw dicts — an auxiliary section most callers
    don't need typed. ``tools`` are converted to ``CatalogTool`` on construction
    (the nested-model pattern C1+ resources reuse).
    """

    agent_id: str
    agent_name: Optional[str] = None
    tools: List[CatalogTool] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentToolCatalog":
        obj = super().from_dict(data)
        # Convert the nested tool dicts to typed CatalogTool models (read the
        # raw list from `data`, not obj.tools, which super() left as dicts).
        # `or []` guards against the key being absent OR present-but-null.
        raw_tools = (data.get("tools") or []) if isinstance(data, dict) else []
        object.__setattr__(obj, "tools", CatalogTool.from_list(raw_tools))
        return obj
