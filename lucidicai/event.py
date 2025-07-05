"""Event management for the Lucidic API"""
from .errors import InvalidOperationError
from .image_upload import get_presigned_url, upload_image_to_s3

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
        self.screenshots = []
        self.is_finished = False
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
            for i in range(len(kwargs['screenshots'])):
                presigned_url, bucket_name, object_key = get_presigned_url(Client().agent_id, session_id=self.session_id, event_id=self.event_id, nthscreenshot=i + len(self.screenshots))
                upload_image_to_s3(presigned_url, kwargs['screenshots'][i], "JPEG")
                self.screenshots.append(kwargs['screenshots'][i])
        if 'is_finished' in kwargs:
            self.is_finished = kwargs['is_finished']
        request_data = {
            "event_id": self.event_id,
            "description": Client().mask(kwargs.get("description", None)),
            "result": Client().mask(kwargs.get("result", None)),
            "is_finished": self.is_finished, 
            "cost_added": kwargs.get("cost_added", None),
            "model": kwargs.get("model", None), 
            "nscreenshots": len(self.screenshots)
        }
        Client().make_request('updateevent', 'PUT', request_data)
