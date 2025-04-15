"""Event management for the Lucidic API"""
from .errors import InvalidOperationError

class Event:
    def __init__(
        self, 
        session_id: str, 
        step_id: str, 
        **kwargs
    ):
        self.session_id = session_id
        self.step_id = step_id
        self.event_id = None
        self.description = None
        self.result = None
        self.is_finished = False
        self.cost_added = None
        self.model = None
        self.screenshots = []
        self.init_event()
        self.update_event(**kwargs)


    def init_event(self) -> None:
        from .client import Client
        request_data = {
            "step_id": self.step_id,
            # TODO: get rid of these in backend API interface?
            # "description": description,
            # "result": result
        }
        data = Client().make_request('initevent', 'POST', request_data)
        self.event_id = data["event_id"]

    def update_event(self, **kwargs) -> None:
        from .client import Client
        if 'screenshots' in kwargs and kwargs['screenshots'] is not None:
            self.screenshots += kwargs['screenshots']
        if 'is_finished' in kwargs:
            if self.is_finished:
                raise InvalidOperationError("Event is already finished")
        update_attrs = {k: v for k, v in kwargs.items() if v is not None}
        self.__dict__.update(update_attrs)
        request_data = {
            "event_id": self.event_id,
            "description": self.description,
            "result": self.result,
            "is_finished": self.is_finished, 
            "cost_added": self.cost_added,
            "model": self.model, 
            "nscreenshots": len(self.screenshots)
        }
        Client().make_request('updateevent', 'PUT', request_data)
