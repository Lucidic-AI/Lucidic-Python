import base64
import io
from typing import List, Optional

from PIL import Image

from .errors import InvalidOperationError, LucidicNotInitializedError
from .image_upload import get_presigned_url, upload_image_to_s3
from .step import Step

class Session:
    def __init__(
        self, 
        agent_id: str, 
        session_name: str, 
        mass_sim_id: Optional[str] = None, 
        task: Optional[str] = None,
        rubrics: Optional[list] = None
    ):
        self.agent_id = agent_id
        self.session_name = session_name
        self.mass_sim_id = mass_sim_id
        self.task = task
        self.session_id = None
        self.step_history: List[Step] = []
        self._active_step: Optional[Step] = None
        self.base_url = "https://analytics.lucidic.ai/api"
        self.is_finished = False
        self.rubrics = rubrics
        self.is_successful = None
        self.is_successful_reason = None
        self.user_eval = None
        self.user_eval_reason = None
        self.init_session()

    def init_session(self) -> None:
        from .client import Client
        request_data = {
            "agent_id": self.agent_id,
            "session_name": self.session_name,
            "task": self.task,
            "mass_sim_id": self.mass_sim_id,
            "rubrics": self.rubrics
        }
        data = Client().make_request('initsession', 'POST', request_data)
        self.session_id = data["session_id"]

    @property   
    def active_step(self) -> Optional[Step]:
        return self._active_step
    
    def update_session(
        self, 
        is_finished: Optional[bool] = None,
        task: Optional[str] = None, 
        has_gif=False,
        is_successful: Optional[bool] = None,
        is_successful_reason: Optional[str] = None,
        user_eval: Optional[str] = None,
        user_eval_reason: Optional[str] = None
    ) -> None:
        from .client import Client
        update_attrs = {k: v for k, v in locals().items() 
                       if k != 'self' and v is not None}
        self.__dict__.update(update_attrs)
        request_data = {
            "session_id": self.session_id,
            "is_finished": self.is_finished,
            "task": self.task,
            "has_gif": has_gif,
            "is_successful": self.is_successful,
            "is_successful_reason": self.is_successful_reason,
            "user_eval": self.user_eval,
            "user_eval_reason": self.user_eval_reason
        }
        Client().make_request('updatesession', 'PUT', request_data)

    def create_step(self, **kwargs) -> Step:
        if not self.session_id:
            raise LucidicNotInitializedError()
            
        if self._active_step:
            raise InvalidOperationError("Cannot create new step while another step is active. Please finish current step first.")
                
        step = Step(session_id=self.session_id, **kwargs)
        self._active_step = step
        self.step_history.append(step)
        return step

    def update_step(self, **kwargs) -> None:
        if not self._active_step:
            raise InvalidOperationError("Cannot update step without active step")
        self._active_step.update_step(**kwargs)
        if 'is_finished' in kwargs and kwargs['is_finished']:
            self._active_step = None

    def end_session(self, is_finished: bool = True) -> bool:
        if self._active_step:
            print("[Warning] Ending Lucidic session while current step is unfinished...")
        self.is_finished = is_finished and True
        images_b64 = []
        events_b64 = [] # (event_id, nth screenshot, b64)
        for step in self.step_history:
            if step.screenshot is not None:
                images_b64.append(step.screenshot)
            for event in step.event_history:
                for j, event_screenshot in enumerate(event.screenshots):
                    events_b64.append((event.event_id, j, event_screenshot))
        has_gif = False
        if images_b64:
            images = [Image.open(io.BytesIO(base64.b64decode(b64))) for b64 in images_b64]
            gif_buffer = io.BytesIO()
            images[0].save(gif_buffer, format="GIF", save_all=True, append_images=images[1:], duration=2000, loop=0)
            gif_buffer.seek(0)
            presigned_url, bucket_name, object_key = get_presigned_url(self.agent_id, session_id=self.session_id)
            upload_image_to_s3(presigned_url, gif_buffer, "GIF")
            has_gif = True
        for event_id, nthscreenshot, event_b64 in events_b64:
            presigned_url, bucket_name, object_key = get_presigned_url(self.agent_id, session_id=self.session_id, event_id=event_id, nthscreenshot=nthscreenshot)
            upload_image_to_s3(presigned_url, event_b64, "JPEG")
        return self.update_session(is_finished=True, has_gif=has_gif)