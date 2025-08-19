"""Event API resource handler"""

from typing import Optional, List, Dict, Any
from lucidicai.client.http_client import HttpClient


class EventResource:
    """Handles event-related API operations"""
    
    def __init__(self, http_client: HttpClient):
        """Initialize event resource with HTTP client
        
        Args:
            http_client: HTTP client instance for API requests
        """
        self.http = http_client
    
    def init_event(
        self,
        session_id: str,
        step_id: Optional[str] = None,
        description: Optional[str] = None,
        result: Optional[str] = None,
        cost_added: Optional[float] = None,
        model: Optional[str] = None,
        duration: Optional[float] = None,
        nscreenshots: Optional[int] = None,
        function_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Initialize a new event
        
        Args:
            session_id: Session ID for the event
            step_id: Optional step ID to link event to
            description: Event description
            result: Event result
            cost_added: Cost added by this event
            model: Model used
            duration: Duration of event
            nscreenshots: Number of screenshots
            function_name: Name of function that triggered event
            arguments: Function arguments
            
        Returns:
            Dict with event_id and other event data
        """
        request_data = {
            "session_id": session_id,
            "step_id": step_id,
            "description": description,
            "result": result,
            "cost_added": cost_added,
            "model": model,
            "duration": duration,
            "nscreenshots": nscreenshots,
            "function_name": function_name,
            "arguments": arguments,
        }
        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}
        return self.http.post("initevent", request_data)
    
    def update_event(
        self,
        event_id: str,
        description: Optional[str] = None,
        result: Optional[str] = None,
        cost_added: Optional[float] = None,
        model: Optional[str] = None,
        is_finished: Optional[bool] = None,
        duration: Optional[float] = None,
        function_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update event information
        
        Args:
            event_id: Event ID to update
            description: Event description
            result: Event result
            cost_added: Cost added by this event
            model: Model used
            is_finished: Whether event is finished
            duration: Duration of event
            function_name: Name of function
            arguments: Function arguments
            
        Returns:
            Updated event data
        """
        request_data = {
            "event_id": event_id,
            "description": description,
            "result": result,
            "cost_added": cost_added,
            "model": model,
            "is_finished": is_finished,
            "duration": duration,
            "function_name": function_name,
            "arguments": arguments,
        }
        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}
        return self.http.put("updateevent", request_data)