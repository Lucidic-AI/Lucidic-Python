import io
from typing import List, TYPE_CHECKING

from PIL import Image

from .action import Action
from .errors import InvalidOperationError
from .image_upload import get_presigned_url, upload_image_to_s3
from .state import State

if TYPE_CHECKING:
    from .event import Event


class Step:
    """Represents a step within a session"""
    def __init__(self, session_id: str, **kwargs):
        self.session_id = session_id
        self.step_id = None
        self.goal = None
        self.state = None
        self.action = None
        self.event_history: List[Event] = []
        self.is_successful = None
        self.is_finished = False
        self.eval_score = None
        self.eval_description = None
        self.cost = 0.0  
        self.screenshot = None
        self.init_step()
        self.update_step(**kwargs)

    def init_step(self) -> None:
        """Initialize the step with the API"""
        from .client import Client
        request_data = {
            "session_id": self.session_id,
            # TODO: Remove following from init_step backend API 
            # "goal": self.goal,
            # "action": str(self.action) if self.action else None,
            # "state": str(self.state)
        }
        data = Client().make_request('initstep', 'POST', request_data)
        self.step_id = data["step_id"]

    def update_step(self, **kwargs) -> None:
        from .client import Client
        update_attrs = {k: v for k, v in kwargs.items() if v is not None}
        self.__dict__.update(update_attrs)
        if ('screenshot' in kwargs and kwargs['screenshot'] is not None) or ('screenshot_path' in kwargs and kwargs['screenshot_path'] is not None):
            self.try_upload_screenshot(**kwargs)
        if 'state' in kwargs:
            self.state = State(kwargs['state'])
        if 'action' in kwargs:
            self.action = Action(kwargs['action'])
        if 'is_finished' in kwargs:
            self.cost = sum(event.cost_added for event in self.event_history if event.cost_added is not None)
        request_data = {
            "step_id": self.step_id,
            "goal": self.goal,
            "action": str(self.action) if self.action else None,
            "state": str(self.state),
            "is_successful": self.is_successful,
            "eval_score": self.eval_score,
            "eval_description": self.eval_description,
            "is_finished": self.is_finished,
            "cost_added": self.cost,
            "has_screenshot": True if self.screenshot is not None else False
        }
        Client().make_request('updatestep', 'PUT', request_data)

    def try_upload_screenshot(self, **kwargs) -> None:
        from .client import Client
        if 'screenshot_path' in kwargs and kwargs['screenshot_path']:
            screenshot_path = kwargs['screenshot_path']
            img = Image.open(screenshot_path)
            img = img.convert("RGB") 
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")  # Save to BytesIO buffer
            img_byte = buffered.getvalue()
            screenshot = base64.b64encode(img_byte).decode('utf-8')
        elif 'screenshot' in kwargs and kwargs['screenshot'] is not None:
            screenshot = kwargs['screenshot']
        presigned_url, bucket_name, object_key = get_presigned_url(Client().agent_id, step_id=self.step_id)
        upload_image_to_s3(presigned_url, screenshot, "JPEG")
        
    def create_event(self, **kwargs) -> 'Event':
        from .event import Event  # Import moved inside method
        
        if not self.step_id:
            raise InvalidOperationError("Step ID not set. Call init_step first.")
            
        if self.is_finished:
            raise InvalidOperationError("Cannot create event in finished step")
        if self.event_history and not self.event_history[-1].is_finished:
            raise InvalidOperationError("Cannot create new event while previous event is unfinished")
            
        event = Event(
            session_id=self.session_id,
            step_id=self.step_id,
            **kwargs
        )
        
        self.event_history.append(event)
        return event

