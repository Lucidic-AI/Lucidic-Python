"""Step API resource handler"""

from typing import Optional, Dict, Any
from lucidicai.client.http_client import HttpClient


class StepResource:
    """Handles step-related API operations"""
    
    def __init__(self, http_client: HttpClient):
        """Initialize step resource with HTTP client
        
        Args:
            http_client: HTTP client instance for API requests
        """
        self.http = http_client
    
    def init_step(self, session_id: str) -> Dict[str, Any]:
        """Initialize a new step in a session
        
        Args:
            session_id: Session ID to create step in
            
        Returns:
            Dict with step_id and other step data
        """
        return self.http.post("initstep", {"session_id": session_id})
    
    def update_step(
        self,
        step_id: str,
        state: Optional[str] = None,
        action: Optional[str] = None,
        goal: Optional[str] = None,
        eval_score: Optional[float] = None,
        eval_description: Optional[str] = None,
        is_finished: Optional[bool] = None,
        has_screenshot: Optional[bool] = None,
        has_gif: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Update step information
        
        Args:
            step_id: Step ID to update
            state: Optional state description
            action: Optional action description
            goal: Optional goal description
            eval_score: Optional evaluation score
            eval_description: Optional evaluation description
            is_finished: Whether step is finished
            has_screenshot: Whether step has screenshot
            has_gif: Whether step has GIF
            
        Returns:
            Updated step data
        """
        request_data = {
            "step_id": step_id,
            "state": state,
            "action": action,
            "goal": goal,
            "eval_score": eval_score,
            "eval_description": eval_description,
            "is_finished": is_finished,
            "has_screenshot": has_screenshot,
            "has_gif": has_gif,
        }
        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}
        return self.http.put("updatestep", request_data)