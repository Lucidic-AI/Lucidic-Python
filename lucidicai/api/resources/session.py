"""Session resource API operations."""
from typing import Any, Dict, List, Optional

from ..client import HttpClient


class SessionResource:
    """Handle session-related API operations."""
    
    def __init__(self, http: HttpClient):
        """Initialize session resource.
        
        Args:
            http: HTTP client instance
        """
        self.http = http
    
    def create_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new session.
        
        Args:
            params: Session parameters including:
                - session_name: Name of the session
                - agent_id: Agent ID
                - task: Optional task description
                - tags: Optional tags
                - etc.
                
        Returns:
            Created session data with session_id
        """
        return self.http.post("initsession", params)
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get a session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data
        """
        return self.http.get(f"sessions/{session_id}")
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing session.
        
        Args:
            session_id: Session ID
            updates: Fields to update (task, is_finished, etc.)
            
        Returns:
            Updated session data
        """
        # Add session_id to the updates payload
        updates["session_id"] = session_id
        return self.http.put("updatesession", updates)
    
    def end_session(
        self,
        session_id: str,
        is_successful: Optional[bool] = None,
        is_successful_reason: Optional[str] = None,
        session_eval: Optional[float] = None,
        session_eval_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """End a session.
        
        Args:
            session_id: Session ID
            is_successful: Whether session was successful
            is_successful_reason: Reason for success or failure
            session_eval: Session evaluation score
            session_eval_reason: Reason for evaluation
            
        Returns:
            Final session data
        """
        updates = {
            "is_finished": True
        }
        
        if is_successful is not None:
            updates["is_successful"] = is_successful
        
        if session_eval is not None:
            updates["session_eval"] = session_eval
        
        if session_eval_reason is not None:
            updates["session_eval_reason"] = session_eval_reason

        if is_successful_reason is not None:
            updates["is_successful_reason"] = is_successful_reason
        
        return self.update_session(session_id, updates)
    
    def list_sessions(
        self,
        agent_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List sessions with optional filters.
        
        Args:
            agent_id: Filter by agent ID
            experiment_id: Filter by experiment ID
            limit: Maximum number of sessions
            offset: Pagination offset
            
        Returns:
            List of sessions and pagination info
        """
        params = {
            "limit": limit,
            "offset": offset
        }
        
        if agent_id:
            params["agent_id"] = agent_id
        
        if experiment_id:
            params["experiment_id"] = experiment_id
        
        return self.http.get("sessions", params)