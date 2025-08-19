"""Session API resource handler"""

from typing import Optional, List, Dict, Any
from lucidicai.client.http_client import HttpClient


class SessionResource:
    """Handles session-related API operations"""
    
    def __init__(self, http_client: HttpClient):
        """Initialize session resource with HTTP client
        
        Args:
            http_client: HTTP client instance for API requests
        """
        self.http = http_client
    
    def init_session(
        self,
        agent_id: str,
        session_name: Optional[str] = None,
        session_id: Optional[str] = None,
        task: Optional[str] = None,
        mass_sim_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        rubrics: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        production_monitoring: bool = False
    ) -> Dict[str, Any]:
        """Initialize a new session
        
        Args:
            agent_id: Agent ID
            session_name: Optional session name
            session_id: Optional custom session ID
            task: Optional task description
            mass_sim_id: Optional mass simulation ID
            experiment_id: Optional experiment ID
            rubrics: Optional list of rubrics
            tags: Optional list of tags
            production_monitoring: Whether to enable production monitoring
            
        Returns:
            Dict with session_id and other session data
        """
        request_data = {
            "agent_id": agent_id,
            "session_name": session_name,
            "session_id": session_id,
            "task": task,
            "mass_sim_id": mass_sim_id,
            "experiment_id": experiment_id,
            "rubrics": rubrics,
            "tags": tags,
            "production_monitoring": production_monitoring,
        }
        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}
        return self.http.post("initsession", request_data)
    
    def continue_session(self, session_id: str) -> Dict[str, Any]:
        """Continue an existing session
        
        Args:
            session_id: Session ID to continue
            
        Returns:
            Dict with session data
        """
        return self.http.post("continuesession", {"session_id": session_id})
    
    def update_session(
        self,
        session_id: str,
        task: Optional[str] = None,
        is_finished: Optional[bool] = None,
        is_successful: Optional[bool] = None,
        is_successful_reason: Optional[str] = None,
        session_eval: Optional[float] = None,
        session_eval_reason: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Update session information
        
        Args:
            session_id: Session ID to update
            task: Optional task description
            is_finished: Whether session is finished
            is_successful: Whether session was successful
            is_successful_reason: Reason for success/failure
            session_eval: Session evaluation score
            session_eval_reason: Session evaluation reason
            tags: Optional tags to add
            
        Returns:
            Updated session data
        """
        request_data = {
            "session_id": session_id,
            "task": task,
            "is_finished": is_finished,
            "is_successful": is_successful,
            "is_successful_reason": is_successful_reason,
            "session_eval": session_eval,
            "session_eval_reason": session_eval_reason,
            "tags": tags,
        }
        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}
        return self.http.put("updatesession", request_data)