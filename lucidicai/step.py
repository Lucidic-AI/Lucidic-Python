import base64
import io
from typing import List, TYPE_CHECKING

from PIL import Image

from .errors import InvalidOperationError
from .image_upload import get_presigned_url, screenshot_path_to_jpeg, upload_image_to_s3

if TYPE_CHECKING:
    from .event import Event


class Step:
    """Represents a step within a session"""
    def __init__(self, session_id: str, **kwargs):
        self.session_id = session_id
        self.step_id = None
        self.is_finished = False
        self.init_step(**kwargs)

    def init_step(self, **kwargs) -> None:
        """Initialize the step with the API"""
        from .client import Client
        data = Client().make_request('initstep', 'POST', {"session_id": self.session_id})
        self.step_id = data["step_id"]
        self.update_step(**kwargs)
    
    def update_step(self, **kwargs) -> None:
        """Update the step with the API"""
        from .client import Client
        request_data = self.build_request_data(**kwargs)
        Client().make_request('updatestep', 'PUT', request_data)
        
        # Update local state
        if 'is_finished' in kwargs:
            self.is_finished = kwargs['is_finished']

    def build_request_data(self, **kwargs):
        from .client import Client
        screenshot = kwargs['screenshot'] if 'screenshot' in kwargs else None
        if 'screenshot_path' in kwargs and kwargs['screenshot_path'] is not None:
            screenshot = screenshot_path_to_jpeg(kwargs['screenshot_path'])
        if screenshot is not None:
            presigned_url, bucket_name, object_key = get_presigned_url(Client().agent_id, step_id=self.step_id)
            if screenshot.startswith("data:image"):
                screenshot = screenshot[screenshot.find(",") + 1:]
            upload_image_to_s3(presigned_url, screenshot, "JPEG")
        request_data = {
            "step_id": self.step_id,
            "goal": Client().mask(kwargs['goal']) if 'goal' in kwargs else None,
            "action": Client().mask(kwargs['action']) if 'action' in kwargs else None,
            "state": Client().mask(kwargs['state']) if 'state' in kwargs else None,
            "eval_score": kwargs['eval_score'] if 'eval_score' in kwargs else None,
            "eval_description": Client().mask(kwargs['eval_description']) if 'eval_description' in kwargs else None,
            "is_finished": kwargs['is_finished'] if 'is_finished' in kwargs else None,
            "has_screenshot": True if screenshot else None
        }
        return request_data

