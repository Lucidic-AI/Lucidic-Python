import io
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

from PIL import Image
import requests

from .action import Action
from .errors import InvalidOperationError
from .state import State

if TYPE_CHECKING:
    from .event import Event


class Step:
    """Represents a step within a session"""
    def __init__(
        self, session_id: str, 
        state: Optional[str] = None, 
        action: Optional[str] = None, 
        goal: Optional[str] = None
    ):
        self.session_id = session_id
        self.step_id: Optional[str] = None
        self.end_time: Optional[str] = None
        self.base_url = "https://analytics.lucidic.ai/api"
        self.goal = goal
        self.state = State(state or "state not provided")
        self.action = Action(action) if action else None
        self.event_history: List[Event] = []
        self.is_successful = None
        self.is_finished = None
        self.eval_score = None
        self.eval_description = None
        self.cost_added = 0.0  
        self.start_time = datetime.now().isoformat()
        self.screenshot = None
        self.init_step()

    def init_step(self) -> bool:
        """Initialize the step with the API"""
        from .client import Client
        request_data = {
            "session_id": self.session_id,
            "current_time": datetime.now().isoformat(),
            "goal": self.goal,
            "action": str(self.action) if self.action else None,
            "state": str(self.state)
        }
        data = Client().make_request('initstep', 'POST', request_data)
        self.step_id = data["step_id"]
        return True

    def update_step(
        self, 
        goal: Optional[str] = None, 
        action: Optional[Action] = None,
        state: Optional[State] = None, 
        is_successful: Optional[bool] = None, 
        eval_score: Optional[float] = None,
        eval_description: Optional[float] = None,
        is_finished: Optional[bool] = None, 
        cost_added: Optional[float] = None,
        model: Optional[str] = None
    ) -> bool:
        from .client import Client
        update_attrs = {k: v for k, v in locals().items() 
                       if k != 'self' and v is not None}
        self.__dict__.update(update_attrs)
        request_data = {
            "step_id": self.step_id,
            "current_time": datetime.now().isoformat(),
            "goal": self.goal,
            "action": str(self.action) if self.action else None,
            "state": str(self.state),
            "is_successful": self.is_successful,
            "eval_score": self.eval_score,
            "eval_description": self.eval_description,
            "is_finished": self.is_finished,
            "cost_added": self.cost_added,
            "has_screenshot": True if self.screenshot is not None else False
        }
        Client().make_request('updatestep', 'PUT', request_data)
        return True

    def set_state(self, state: str) -> None:
        """Set the state for this step"""
        if self.is_finished:
            raise InvalidOperationError("Cannot modify finished step")
        self.state = State(state)
        self.update_step(state=self.state)

    def set_action(self, action: str) -> None:
        """Set the action for this step"""
        if self.is_finished:
            raise InvalidOperationError("Cannot modify finished step")
        self.action = Action(action)
        self.update_step(action=self.action)
        
    def create_event(
        self, 
        description: Optional[str] = None, 
        result: Optional[str] = None
    ) -> 'Event':
        from .event import Event  # Import moved inside method
        
        if not self.step_id:
            raise InvalidOperationError("Step ID not set. Call init_step first.")
            
        if self.is_finished:
            raise InvalidOperationError("Cannot create event in finished step")
        if any(not event.is_finished for event in self.event_history):
            raise InvalidOperationError("Cannot create new event while previous event is unfinished")
            
        event = Event(
            session_id=self.session_id,
            step_id=self.step_id,
            description=description,
            result=result
        )
        
        self.event_history.append(event)
        return event
    
    def end_step(
        self, 
        is_successful: bool, 
        eval_score: Optional[float] = None,
        eval_description: Optional[float] = None,
        final_state: Optional[str] = None,
        final_action: Optional[str] = None, 
        cost_added: Optional[float] = None, 
        model: Optional[str] = None, 
        screenshot = None
    ) -> bool:
        """Finish the step and mark it as successful or failed"""
        # Check for unfinished events
        if any(not event.is_finished for event in self.event_history):
            raise ValueError("Cannot finish step with unfinished events")
            
        self.end_time = datetime.now().isoformat()
        
        if final_state:
            self.state = State(final_state)  # Set directly to avoid update_step call
        if final_action:
            self.action = Action(final_action)  # Set directly to avoid update_step call
            
        # Update total cost from events if not provided
        if cost_added is None:
            self.cost_added = sum(
                event.cost_added for event in self.event_history 
                if event.cost_added is not None
            )
        else:
            self.cost_added = cost_added
            
        self.is_finished = True
        self.is_successful = is_successful

        self.screenshot = screenshot
            
        return self.update_step(
            is_finished=True,
            is_successful=is_successful,
            eval_score=eval_score,
            eval_description=eval_description,
            cost_added=self.cost_added,
            model=model
        )
