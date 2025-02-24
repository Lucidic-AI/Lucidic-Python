"""Step management for the Lucidic API"""
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
import requests
from .errors import handle_session_response
from .action import Action
from .state import State

if TYPE_CHECKING:
    from .event import Event

class Step:
    """Represents a step within a session"""
    def __init__(self, session_id: str, api_key: str, state: Optional[str] = None, 
                 action: Optional[str] = None, goal: Optional[str] = None):
        self.session_id = session_id
        self.api_key = api_key
        self.step_id: Optional[str] = None
        self.end_time: Optional[str] = None
        self.base_url = "https://analytics.lucidic.ai/api"
        
        self.goal = goal
        self.state = State(state or "state not provided")
        self.action = Action(action) if action else None
        self.event_history: List[Event] = []
        self.is_successful = None
        self.is_finished = None
        self.cost_added = 0.0  # Initialize cost to 0
        self.model = None
        self.start_time = datetime.now().isoformat()
        
        self.init_step()

    def init_step(self) -> bool:
        """Initialize the step with the API"""
        request_data = {
            "session_id": self.session_id,
            "current_time": datetime.now().isoformat(),
            "goal": self.goal,
            "action": str(self.action) if self.action else None,
            "state": str(self.state)
        }
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        
        response = requests.post(
            f"{self.base_url}/initstep",
            headers=headers,
            json=request_data
        )
        
        data = handle_session_response(response, required_fields=["step_id"])
        self.step_id = data["step_id"]
        return True

    def update_step(self, goal: Optional[str] = None, action: Optional[Action] = None,
                   state: Optional[State] = None, is_successful: Optional[bool] = None, 
                   is_finished: Optional[bool] = None, cost_added: Optional[float] = None, 
                   model: Optional[str] = None) -> bool:
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
            "is_finished": self.is_finished,
            "cost_added": self.cost_added,
            "model": self.model
        }
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        
        response = requests.put(
            f"{self.base_url}/updatestep",
            headers=headers,
            json=request_data
        )
        
        handle_session_response(response)
        return True

    def set_state(self, state: str) -> None:
        """Set the state for this step"""
        if self.is_finished:
            raise ValueError("Cannot modify state of finished step")
        self.state = State(state)
        self.update_step(state=self.state)

    def set_action(self, action: str) -> None:
        """Set the action for this step"""
        if self.is_finished:
            raise ValueError("Cannot modify action of finished step")
        self.action = Action(action)
        self.update_step(action=self.action)
        
    def create_event(self, description: Optional[str] = None, result: Optional[str] = None) -> 'Event':
        """Create a new event within this step"""
        from .event import Event  # Import moved inside method
        
        if not self.step_id:
            raise ValueError("Step ID not set. Call init_step first.")
            
        if self.is_finished:
            raise ValueError("Cannot create event in finished step")
            
        # Check if there are any unfinished events
        if any(not event.is_finished for event in self.event_history):
            raise ValueError("Cannot create new event while previous event is unfinished")
            
        event = Event(
            session_id=self.session_id,
            step_id=self.step_id,
            api_key=self.api_key,
            description=description,
            result=result
        )
        
        if event.init_event(description, result):
            self.event_history.append(event)
            return event
        raise ValueError("Failed to initialize event")
    
    def finish_step(self, is_successful: bool, final_state: Optional[str] = None,
                   final_action: Optional[str] = None, cost_added: Optional[float] = None, 
                   model: Optional[str] = None) -> bool:
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
            
        return self.update_step(
            is_finished=True,
            is_successful=is_successful,
            cost_added=self.cost_added,
            model=model
        )

    def print_event_history(self):
        """Print out the descriptions and results of all events in the step"""
        print(f"\nStep Event History (Goal: {self.goal}):")
        print("-" * 50)
        for i, event in enumerate(self.event_history):
            print(f"Event {i}:")
            print(f"Description: {event.description}")
            print(f"Result: {event.result}")
            print(f"Status: {'Successful' if event.is_successful else 'Failed' if event.is_successful is False else 'Unknown'}")
            print("-" * 50)
