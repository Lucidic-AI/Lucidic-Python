import base64
import io
from typing import List, Optional

from PIL import Image

from .errors import InvalidOperationError, LucidicNotInitializedError
from .image_upload import get_presigned_url, upload_image_to_s3
from .step import Step
from .event import Event

class Session:
    def __init__(
        self, 
        agent_id: str, 
        session_name: Optional[str] = "", 
        session_id: Optional[str] = None,
        mass_sim_id: Optional[str] = None, 
        task: Optional[str] = None,
        rubrics: Optional[list] = None,
        tags: Optional[list] = None
    ):
        self.agent_id = agent_id
        self.session_name = session_name
        self.mass_sim_id = mass_sim_id
        self.task = task
        self.session_id = None
        self.step_history = dict()
        self._active_step: Optional[str] = None  # Rename to latest_step
        self.event_history = dict()
        self.latest_event = None
        self.is_finished = False
        self.rubrics = rubrics
        self.is_successful = None
        self.is_successful_reason = None
        self.session_eval = None
        self.session_eval_reason = None
        self.has_gif = None
        self.tags = tags
        if session_id is None:  # The kwarg, not the attribute
            self.init_session()
        else:
            self.continue_session(session_id)

    def init_session(self) -> None:
        from .client import Client
        request_data = {
            "agent_id": self.agent_id,
            "session_name": self.session_name,
            "task": self.task,
            "mass_sim_id": self.mass_sim_id,
            "rubrics": self.rubrics,
            "tags": self.tags
        }
        data = Client().make_request('initsession', 'POST', request_data)
        self.session_id = data["session_id"]
    
    def continue_session(self, session_id: str) -> None:
        from .client import Client
        self.session_id = session_id
        data = Client().make_request('continuesession', 'POST', {"session_id": session_id})
        self.session_id = data["session_id"]
        self.session_name = data["session_name"]
        return self.session_id

    @property   
    def active_step(self) -> Optional[Step]:
        return self._active_step
    
    def update_session(
        self, 
        **kwargs
    ) -> None:
        from .client import Client
        update_attrs = {k: v for k, v in kwargs.items() 
                       if k != 'self' and v is not None}
        self.__dict__.update(update_attrs)
        request_data = {
            "session_id": self.session_id,
            "is_finished": self.is_finished,
            "task": self.task,
            "is_successful": self.is_successful,
            "is_successful_reason": self.is_successful_reason,
            "session_eval": self.session_eval,
            "session_eval_reason": self.session_eval_reason,
            "tags": self.tags
        }
        Client().make_request('updatesession', 'PUT', request_data)

    def create_step(self, **kwargs) -> Step:
        if not self.session_id:
            raise LucidicNotInitializedError()
        step = Step(session_id=self.session_id)
        self.step_history[step.step_id] = step
        self._active_step = step
        return step.step_id

    def update_step(self, **kwargs) -> None:
        if 'step_id' in kwargs and kwargs['step_id'] is not None:
            if kwargs['step_id'] not in self.step_history:
                raise InvalidOperationError("Step ID not found in session history")
            self.step_history[kwargs['step_id']].update_step(**kwargs)
        else:
            if not self._active_step:
                raise InvalidOperationError("No active step to update")
            self._active_step.update_step(**kwargs)


    def create_event(self, **kwargs):
        step_id = self._active_step.step_id
        if 'step_id' in kwargs and kwargs['step_id'] is not None:
            step_id = kwargs['step_id']
        kwargs.pop('step_id', None)
        event = Event(
            session_id=self.session_id,
            step_id=step_id,
            **kwargs
        )
        self.event_history[event.event_id] = event
        self._active_event = event
        return event.event_id

    def update_event(self, **kwargs):
        if 'event_id' in kwargs and kwargs['event_id'] is not None:
            if kwargs['event_id'] not in self.event_history:
                raise InvalidOperationError("Event ID not found in session history")
            self.event_history[kwargs['event_id']].update_event(**kwargs)
        else:
            if not self._active_event:
                raise InvalidOperationError("No active event to update")
            self._active_event.update_event(**kwargs)

            