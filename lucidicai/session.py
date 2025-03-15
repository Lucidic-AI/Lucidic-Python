import base64
import io
from datetime import datetime
from typing import Optional, List

import requests
from PIL import Image

from .action import Action
from .errors import InvalidOperationError, LucidicNotInitializedError
from .event import Event
from .state import State
from .step import Step


class Session:
    def __init__(
        self, 
        agent_id: str, 
        session_name: str, 
        mass_sim_id: Optional[str] = None, 
        task: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.session_name = session_name
        self.mass_sim_id = mass_sim_id
        self.task = task
        self.session_id = None
        self.step_history: List[Step] = []
        self._active_step: Optional[Step] = None
        self.base_url = "https://analytics.lucidic.ai/api"
        self.starttime = datetime.now().isoformat()
        self.is_finished = False
        self.is_successful = None
        self.init_session()

    def init_session(self) -> bool:
        from .client import Client
        request_data = {
            "agent_id": self.agent_id,
            "session_name": self.session_name,
            "current_time": datetime.now().isoformat(),
            "task": self.task,
            "mass_sim_id": self.mass_sim_id
        }
        data = Client().make_request('initsession', 'POST', request_data)
        self.session_id = data["session_id"]
        return True

    @property   
    def active_step(self) -> Optional[Step]:
        return self._active_step
    
    def update_session(
        self, 
        is_finished: Optional[bool] = None,
        is_successful: Optional[bool] = None, 
        task: Optional[str] = None, 
        has_gif=False
    ) -> bool:
        from .client import Client
        update_attrs = {k: v for k, v in locals().items() 
                       if k != 'self' and v is not None}
        self.__dict__.update(update_attrs)
        request_data = {
            "session_id": self.session_id,
            "current_time": datetime.now().isoformat(),
            "is_finished": self.is_finished,
            "is_successful": self.is_successful, 
            "task": self.task,
            "has_gif": has_gif,
        }
        Client().make_request('updatesession', 'PUT', request_data)
        return True

    def create_step(
        self, 
        state: Optional[str] = None, 
        action: Optional[str] = None, 
        goal: Optional[str] = None
    ) -> Step:
        if not self.session_id:
            raise LucidicNotInitializedError()
            
        if self._active_step:
            raise InvalidOperationError("Cannot create new step while another step is active. Please finish current step first.")
        
        state = state or "state not provided"
        
        step = Step(
            session_id=self.session_id,
            state=state,
            action=action,
            goal=goal
        )
        self._active_step = step
        self.step_history.append(step)
        return step

    def update_step(
        self, 
        is_successful: Optional[bool] = None, 
        eval_score: Optional[float] = None,
        eval_description: Optional[float] = None,
        state: Optional[str] = None, 
        action: Optional[str] = None, 
        goal: Optional[str] = None, 
        is_finished: Optional[bool] = None, 
        cost_added: Optional[float] = None,
        model: Optional[str] = None
    ) -> None:
        if not self._active_step:
            raise InvalidOperationError("Cannot update step without active step")
        self._active_step.update_step(
            is_successful=is_successful,
            eval_score=eval_score,
            eval_description=eval_description,
            state=state,
            action=action,
            goal=goal,
            is_finished=is_finished,
            cost_added=cost_added,
            model=model
        )

    def end_step(
        self, 
        is_successful: bool, 
        eval_score: Optional[float] = None,
        eval_description: Optional[float] = None,
        state: Optional[str] = None, 
        action: Optional[str] = None, 
        screenshot = None, 
        screenshot_path=None
    ) -> None:
        if not self._active_step:
            raise InvalidOperationError("Cannot end step without active step")
        if screenshot_path is not None:
            img = Image.open(screenshot_path)
            img = img.convert("RGB") 
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")  # Save to BytesIO buffer
            img_byte = buffered.getvalue()
            screenshot = base64.b64encode(img_byte).decode('utf-8')
        if screenshot is not None:
            presigned_url, bucket_name, object_key = self.get_presigned_url(step_id=self._active_step.step_id)
            self.upload_image_to_s3(presigned_url, screenshot, "JPEG")
        self._active_step.end_step(
            is_successful=is_successful,
            eval_score=eval_score,
            eval_description=eval_description,
            final_state=state,
            final_action=action,
            screenshot=screenshot,
        )
        self._active_step = None

    def create_event(self, description: Optional[str] = None, result: Optional[str] = None) -> Event:
        if not self._active_step:
            print("[Lucidic] Attempted to create event without active step, ignoring...")
            return None
        
        return self._active_step.create_event(description=description, result=result)

    def end_event(self, is_successful: bool, cost_added: Optional[float] = None, 
                 model: Optional[str] = None, result: Optional[str] = None) -> None:
        if not self._active_step:
            raise ValueError("No active step to end event in")
        if not self._active_step.event_history:
            raise ValueError("No events exist in the current step")
        latest_event = self._active_step.event_history[-1]
        if latest_event.is_finished:
            raise ValueError("Latest event is already finished")
        latest_event.end_event(
            is_successful=is_successful, 
            cost_added=cost_added, 
            model=model,
            result=result 
        )
        self._active_step.cost_added = sum(
            event.cost_added for event in self._active_step.event_history 
            if event.cost_added is not None
        )

    def end_session(self, is_successful: bool) -> bool:
        if self._active_step:
            print("[Warning] Ending Lucidic session while current step is unfinished...")
        self.is_finished = True
        self.is_successful = is_successful
        images_b64 = []
        for step in self.step_history:
            if step.screenshot is not None:
                images_b64.append(step.screenshot)
        has_gif = False
        if images_b64:
            images = [Image.open(io.BytesIO(base64.b64decode(b64))) for b64 in images_b64]
            gif_buffer = io.BytesIO()
            images[0].save(gif_buffer, format="GIF", save_all=True, append_images=images[1:], duration=2000, loop=0)
            gif_buffer.seek(0)
            presigned_url, bucket_name, object_key = self.get_presigned_url(session_id=self.session_id)
            self.upload_image_to_s3(presigned_url, gif_buffer, "GIF")
            has_gif = True
        return self.update_session(is_finished=True, is_successful=is_successful, has_gif=has_gif)

    def get_presigned_url(self, step_id=None, session_id=None):
        from .client import Client
        request_data = {
            "agent_id": self.agent_id,
        }
        if step_id is not None:
            request_data["step_id"] = step_id
        if session_id is not None:
            request_data["session_id"] = session_id
        response = Client().make_request('getpresigneduploadurl', 'GET', request_data)
        return response['presigned_url'], response['bucket_name'], response['object_key']

    def upload_image_to_s3(self, url, image, format):
        if format == "JPEG":
            image_stream = io.BytesIO(base64.b64decode(image))
            image_stream.seek(0)
            pil_image = Image.open(image_stream)
            image_obj = io.BytesIO()
            pil_image.save(image_obj, format="JPEG")
            image_obj.seek(0)
            content_type = "image/jpeg"
        elif format == "GIF":
            image_obj = image
            content_type = "image/gif"
        upload_response = requests.put(
            url,
            data=image_obj.getvalue(),
            headers={"Content-Type": content_type}
        )
        upload_response.raise_for_status()
