"""Event resource API operations."""
from typing import Any, Dict, Optional
from datetime import datetime

from ..client import HttpClient


class EventResource:
    """Handle event-related API operations."""
    
    def __init__(self, http: HttpClient):
        """Initialize event resource.
        
        Args:
            http: HTTP client instance
        """
        self.http = http
    
    def create_event(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new event.
        
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
        return self.http.post("events", params)
    
    def get_event(self, event_id: str) -> Dict[str, Any]:
        """Get an event by ID.
        
        Args:
            event_id: Event ID
            
        Returns:
            Event data
        """
        return self.http.get(f"events/{event_id}")
    
    def update_event(self, event_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing event.
        
        Args:
            event_id: Event ID
            updates: Fields to update
            
        Returns:
            Updated event data
        """
        return self.http.put(f"events/{event_id}", updates)
    
    def list_events(
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
        params = {
            "limit": limit,
            "offset": offset
        }
        
        if session_id:
            params["session_id"] = session_id
        
        if event_type:
            params["type"] = event_type
        
        return self.http.get("events", params)