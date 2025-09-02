"""Typed Event model for the Lucidic API"""
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime


class EventType(Enum):
    LLM_GENERATION = "llm_generation"
    FUNCTION_CALL = "function_call"
    ERROR_TRACEBACK = "error_traceback"
    GENERIC = "generic"


class Event:
    """Client-side representation of an Event returned by the backend.

    Note: This object is a thin data container; creation and updates are
    performed by the Client. This model reflects the new typed event schema.
    """

    def __init__(self, event_data: Dict[str, Any], client):
        # Identifiers
        self.event_id: Optional[str] = event_data.get("event_id")
        self.session_id: Optional[str] = event_data.get("session_id")

        # Hierarchy and timing
        self.type: EventType = EventType(event_data.get("type", "generic"))
        self.parent_event_id: Optional[str] = event_data.get("parent_event_id")
        self.created_at: Optional[str] = event_data.get("created_at")
        occurred_at_val = event_data.get("occurred_at")
        # Store occurred_at as datetime if provided in ISO format
        if isinstance(occurred_at_val, str):
            try:
                self.occurred_at: Optional[datetime] = datetime.fromisoformat(occurred_at_val.replace("Z", "+00:00"))
            except Exception:
                self.occurred_at = None
        elif isinstance(occurred_at_val, datetime):
            self.occurred_at = occurred_at_val
        else:
            self.occurred_at = None
        self.duration: Optional[float] = event_data.get("duration")

        # Tags/metadata
        self.tags: List[str] = event_data.get("tags", []) or []
        self.metadata: Dict[str, Any] = event_data.get("metadata", {}) or {}

        # Typed payload
        self.payload: Dict[str, Any] = event_data.get("payload", {}) or {}

        # Local client reference
        self._client = client
        # Completion flag (kept for legacy-like usage; not authoritative)
        self.is_finished: bool = event_data.get("is_finished", False)
