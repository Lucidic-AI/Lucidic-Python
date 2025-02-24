"""Event management for the Lucidic API"""
from datetime import datetime
from typing import Optional
import requests
from .errors import handle_session_response
from .step import Step

class Event:
    def __init__(self, session_id: str, step_id: str, api_key: str,
                 description: Optional[str] = None, result: Optional[str] = None):
        self.session_id = session_id
        self.step_id = step_id
        self.api_key = api_key
        self.event_id: Optional[str] = None
        self.description = description
        self.result = result
        self.is_finished = False
        self.is_successful = None
        self.cost_added = None
        self.model = None
        self.base_url = "https://analytics.lucidic.ai/api"
        
        self.init_event(description, result)

    def init_event(self, description: Optional[str] = None, result: Optional[str] = None) -> bool:
        self.start_time = datetime.now().isoformat()
        request_data = {
            "session_id": self.session_id,
            "step_id": self.step_id,
            "current_time": datetime.now().isoformat(),
            "description": description,
            "result": result
        }
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        
        response = requests.post(
            f"{self.base_url}/initevent",
            headers=headers,
            json=request_data
        )
        
        data = handle_session_response(response, required_fields=["event_id"])
        self.event_id = data["event_id"]
        return True

    def update_event(self, description: Optional[str] = None, result: Optional[str] = None,
                   is_successful: Optional[bool] = None, is_finished: Optional[bool] = None,
                   cost_added: Optional[float] = None, model: Optional[str] = None) -> bool:
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
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        
        response = requests.put(
            f"{self.base_url}/updateevent",
            headers=headers,
            json=request_data
        )
        
        handle_session_response(response)
        return True

    def finish_event(self, is_successful: bool, cost_added: Optional[float] = None,
                    model: Optional[str] = None, result: Optional[str] = None) -> bool:
        if self.is_finished:
            raise ValueError("Event is already finished")
            
        self.is_finished = True
        self.is_successful = is_successful
        self.cost_added = cost_added
        self.model = model
        self.result = result
        self.end_time = datetime.now().isoformat()
        
        request_data = {
            "event_id": self.event_id,
            "current_time": datetime.now().isoformat(),
            "is_successful": is_successful,
            "cost_added": cost_added,
            "model": model,
            "result": result
        }
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        
        response = requests.put(
            f"{self.base_url}/updateevent",
            headers=headers,
            json=request_data
        )
        
        handle_session_response(response)
        return True
