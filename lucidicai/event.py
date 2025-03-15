"""Event management for the Lucidic API"""
from datetime import datetime
from typing import Optional

from .errors import InvalidOperationError

class Event:
    def __init__(
        self, 
        session_id: str, 
        step_id: str, 
        description: Optional[str] = None, 
        result: Optional[str] = None
    ):
        self.session_id = session_id
        self.step_id = step_id
        self.event_id: Optional[str] = None
        self.description = description
        self.result = result
        self.is_finished = False
        self.is_successful = None
        self.cost_added = None
        self.model = None
        self.base_url = "https://analytics.lucidic.ai/api"
        self.init_event(description, result)

    def init_event(
        self, 
        description: Optional[str] = None, 
        result: Optional[str] = None
    ) -> bool:
        from .client import Client
        self.start_time = datetime.now().isoformat()
        request_data = {
            "session_id": self.session_id,
            "step_id": self.step_id,
            "current_time": datetime.now().isoformat(),
            "description": description,
            "result": result
        }
        data = Client().make_request('initevent', 'POST', request_data)
        self.event_id = data["event_id"]
        return True

    def update_event(self, 
        description: Optional[str] = None, 
        result: Optional[str] = None,
        is_successful: Optional[bool] = None, 
        is_finished: Optional[bool] = None,
        cost_added: Optional[float] = None, 
        model: Optional[str] = None
    ) -> bool:
        from .client import Client
        update_attrs = {k: v for k, v in locals().items() 
                       if k != 'self' and v is not None}
        self.__dict__.update(update_attrs)
        request_data = {
            "event_id": self.event_id,
            "current_time": datetime.now().isoformat(),
            "description": self.description,
            "result": self.result,
            "is_successful": self.is_successful,
            "is_finished": self.is_finished,
            "cost_added": self.cost_added,
            "model": self.model
        }
        Client().make_request('updateevent', 'PUT', request_data)
        return True

    def end_event(
        self, 
        is_successful: bool, 
        cost_added: Optional[float] = None,
        model: Optional[str] = None, 
        result: Optional[str] = None,
        description: Optional[str] = None, 
    ) -> bool:
        if self.is_finished:
            raise InvalidOperationError("Event is already finished")
        
        self.end_time = datetime.now().isoformat()
        return self.update_event(
            is_successful=is_successful,
            is_finished=True,
            cost_added=cost_added,
            model=model,
            result=result,
            description=description
        )
