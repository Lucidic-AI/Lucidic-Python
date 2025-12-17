"""Event resource API operations."""
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..client import HttpClient

logger = logging.getLogger("Lucidic")


def _truncate_id(id_str: Optional[str]) -> str:
    """Truncate ID for logging."""
    if not id_str:
        return "None"
    return f"{id_str[:8]}..." if len(id_str) > 8 else id_str


class EventResource:
    """Handle event-related API operations."""

    def __init__(self, http: HttpClient, production: bool = False):
        """Initialize event resource.

        Args:
            http: HTTP client instance
            production: Whether to suppress errors in production mode
        """
        self.http = http
        self._production = production

    # ==================== High-Level Event Methods ====================

    def create(
        self,
        type: str = "generic",
        event_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Create a new event.

        Args:
            type: Event type (e.g., "llm_generation", "function_call", "error_traceback", "generic")
            event_id: Optional client event ID (auto-generated if not provided)
            session_id: Optional session ID (uses current context if not provided)
            **kwargs: Event-specific fields

        Returns:
            Event ID (client-generated or provided UUID)

        Example:
            event_id = client.events.create(
                type="custom_event",
                data={"key": "value"}
            )
        """
        from ...sdk.context import current_session_id, current_parent_event_id
        from ...sdk.event_builder import EventBuilder

        # Generate event ID if not provided
        client_event_id = event_id or str(uuid.uuid4())

        # Get session from context if not provided
        if not session_id:
            session_id = current_session_id.get(None)

        if not session_id:
            logger.debug("[EventResource] No active session for create()")
            return client_event_id

        # Get parent event ID from context
        parent_event_id = None
        try:
            parent_event_id = current_parent_event_id.get(None)
        except Exception:
            pass

        # Build event request
        params = {
            "type": type,
            "event_id": client_event_id,
            "parent_event_id": parent_event_id,
            "session_id": session_id,
            "occurred_at": kwargs.pop("occurred_at", None)
            or datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }

        try:
            event_request = EventBuilder.build(params)
            self.create_event(event_request)
            logger.debug(f"[EventResource] Created event {client_event_id[:8]}...")
        except Exception as e:
            if self._production:
                logger.error(f"[EventResource] Failed to create event: {e}")
            else:
                raise

        return client_event_id

    async def acreate(
        self,
        type: str = "generic",
        event_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Create a new event (async version).

        See create() for full documentation.
        """
        from ...sdk.context import current_session_id, current_parent_event_id
        from ...sdk.event_builder import EventBuilder

        client_event_id = event_id or str(uuid.uuid4())

        if not session_id:
            session_id = current_session_id.get(None)

        if not session_id:
            logger.debug("[EventResource] No active session for acreate()")
            return client_event_id

        parent_event_id = None
        try:
            parent_event_id = current_parent_event_id.get(None)
        except Exception:
            pass

        params = {
            "type": type,
            "event_id": client_event_id,
            "parent_event_id": parent_event_id,
            "session_id": session_id,
            "occurred_at": kwargs.pop("occurred_at", None)
            or datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }

        try:
            event_request = EventBuilder.build(params)
            await self.acreate_event(event_request)
            logger.debug(f"[EventResource] Created async event {client_event_id[:8]}...")
        except Exception as e:
            if self._production:
                logger.error(f"[EventResource] Failed to create async event: {e}")
            else:
                raise

        return client_event_id

    def emit(
        self,
        type: str = "generic",
        event_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Fire-and-forget event creation that returns instantly.

        This function returns immediately with an event ID, while the actual
        event creation happens in a background thread. Perfect for hot path
        telemetry where latency is critical.

        Args:
            type: Event type (e.g., "llm_generation", "function_call", "generic")
            event_id: Optional client event ID (auto-generated if not provided)
            session_id: Optional session ID (uses current context if not provided)
            **kwargs: Event-specific fields

        Returns:
            Event ID (client-generated or provided UUID) - returned immediately

        Example:
            client.events.emit(type="log", message="Something happened")
        """
        from ...sdk.context import current_session_id, current_parent_event_id

        # Pre-generate event ID for instant return
        client_event_id = event_id or str(uuid.uuid4())

        # Capture context variables BEFORE creating the thread
        captured_parent_id = kwargs.get("parent_event_id")
        if captured_parent_id is None:
            try:
                captured_parent_id = current_parent_event_id.get(None)
            except Exception:
                pass

        # Get session from context if not provided
        captured_session_id = session_id
        if not captured_session_id:
            captured_session_id = current_session_id.get(None)

        if not captured_session_id:
            logger.debug("[EventResource] No active session for emit()")
            return client_event_id

        # Capture all data for background thread
        captured_kwargs = dict(kwargs)
        captured_kwargs["parent_event_id"] = captured_parent_id

        def _background_create():
            try:
                self.create(
                    type=type,
                    event_id=client_event_id,
                    session_id=captured_session_id,
                    **captured_kwargs,
                )
            except Exception as e:
                logger.debug(f"[EventResource] Background emit() failed: {e}")

        # Start background thread
        thread = threading.Thread(target=_background_create, daemon=True)
        thread.start()

        return client_event_id

    def create_error(
        self,
        error: Any,
        parent_event_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Create an error traceback event.

        Convenience method for creating error events with proper traceback information.

        Args:
            error: The error message or exception object
            parent_event_id: Optional parent event ID for nesting
            **kwargs: Additional event parameters

        Returns:
            Event ID of the created error event

        Example:
            try:
                risky_operation()
            except Exception as e:
                client.events.create_error(e)
        """
        import traceback as tb

        if isinstance(error, Exception):
            error_str = str(error)
            traceback_str = tb.format_exc()
        else:
            error_str = str(error)
            traceback_str = kwargs.pop("traceback", "")

        return self.create(
            type="error_traceback",
            error=error_str,
            traceback=traceback_str,
            parent_event_id=parent_event_id,
            **kwargs,
        )

    async def acreate_error(
        self,
        error: Any,
        parent_event_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Create an error traceback event (async version).

        See create_error() for full documentation.
        """
        import traceback as tb

        if isinstance(error, Exception):
            error_str = str(error)
            traceback_str = tb.format_exc()
        else:
            error_str = str(error)
            traceback_str = kwargs.pop("traceback", "")

        return await self.acreate(
            type="error_traceback",
            error=error_str,
            traceback=traceback_str,
            parent_event_id=parent_event_id,
            **kwargs,
        )

    # ==================== Low-Level HTTP Methods ====================

    def create_event(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new event via API.

        Args:
            params: Event parameters including:
                - client_event_id: Client-generated event ID
                - session_id: Session ID
                - type: Event type
                - occurred_at: When the event occurred
                - payload: Event payload
                - etc.

        Returns:
            API response with optional blob_url for large payloads
        """
        event_id = params.get("client_event_id")
        session_id = params.get("session_id")
        event_type = params.get("type")
        parent_id = params.get("client_parent_event_id")
        logger.debug(
            f"[Event] create_event() called - "
            f"event_id={_truncate_id(event_id)}, session_id={_truncate_id(session_id)}, "
            f"type={event_type!r}, parent_id={_truncate_id(parent_id)}"
        )

        response = self.http.post("events", params)

        resp_event_id = response.get("event_id") if response else None
        logger.debug(
            f"[Event] create_event() response - "
            f"event_id={_truncate_id(resp_event_id)}, response_keys={list(response.keys()) if response else 'None'}"
        )
        return response

    async def acreate_event(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new event via API (asynchronous).

        Args:
            params: Event parameters

        Returns:
            API response with optional blob_url for large payloads
        """
        event_id = params.get("client_event_id")
        session_id = params.get("session_id")
        event_type = params.get("type")
        parent_id = params.get("client_parent_event_id")
        logger.debug(
            f"[Event] acreate_event() called - "
            f"event_id={_truncate_id(event_id)}, session_id={_truncate_id(session_id)}, "
            f"type={event_type!r}, parent_id={_truncate_id(parent_id)}"
        )

        response = await self.http.apost("events", params)

        resp_event_id = response.get("event_id") if response else None
        logger.debug(
            f"[Event] acreate_event() response - "
            f"event_id={_truncate_id(resp_event_id)}, response_keys={list(response.keys()) if response else 'None'}"
        )
        return response

    def get(self, event_id: str) -> Dict[str, Any]:
        """Get an event by ID.

        Args:
            event_id: Event ID

        Returns:
            Event data
        """
        return self.http.get(f"events/{event_id}")

    def update(self, event_id: str, **updates) -> Dict[str, Any]:
        """Update an existing event.

        Args:
            event_id: Event ID
            **updates: Fields to update

        Returns:
            Updated event data
        """
        return self.http.put(f"events/{event_id}", updates)

    def list(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List events with optional filters.

        Args:
            session_id: Filter by session ID
            event_type: Filter by event type
            limit: Maximum number of events to return
            offset: Pagination offset

        Returns:
            List of events and pagination info
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset
        }

        if session_id:
            params["session_id"] = session_id

        if event_type:
            params["type"] = event_type

        return self.http.get("events", params)

    async def aget(self, event_id: str) -> Dict[str, Any]:
        """Get an event by ID (asynchronous).

        Args:
            event_id: Event ID

        Returns:
            Event data
        """
        return await self.http.aget(f"events/{event_id}")

    async def aupdate(self, event_id: str, **updates) -> Dict[str, Any]:
        """Update an existing event (asynchronous).

        Args:
            event_id: Event ID
            **updates: Fields to update

        Returns:
            Updated event data
        """
        return await self.http.aput(f"events/{event_id}", updates)

    async def alist(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List events with optional filters (asynchronous).

        Args:
            session_id: Filter by session ID
            event_type: Filter by event type
            limit: Maximum number of events to return
            offset: Pagination offset

        Returns:
            List of events and pagination info
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset
        }

        if session_id:
            params["session_id"] = session_id

        if event_type:
            params["type"] = event_type

        return await self.http.aget("events", params)
