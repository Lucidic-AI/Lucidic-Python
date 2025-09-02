from typing import Optional

from .errors import InvalidOperationError, LucidicNotInitializedError


class Session:
    def __init__(
        self, 
        agent_id: str,
        session_id = None,
        **kwargs
    ):
        self.agent_id = agent_id
        self.session_id = session_id
        self.event_history = []  # List[Event]
        self.latest_event = None
        self.is_finished = False
        self.is_successful = None
        self.is_successful_reason = None
        self.session_eval = None
        self.session_eval_reason = None
        self.has_gif = None
        
    def update_session(
        self, 
        **kwargs
    ) -> None:
        from .client import Client
        request_data = {
            "session_id": self.session_id,
            "is_finished": kwargs.get("is_finished", None),
            "task": kwargs.get("task", None),
            "is_successful": kwargs.get("is_successful", None),
            "is_successful_reason": Client().mask(kwargs.get("is_successful_reason", None)),
            "session_eval": kwargs.get("session_eval", None),
            "session_eval_reason": Client().mask(kwargs.get("session_eval_reason", None)),
            "tags": kwargs.get("tags", None)
        }
        Client().make_request('updatesession', 'PUT', request_data)

    def create_event(self, type: str = "generic", **kwargs) -> str:
        """Proxy to client.create_event bound to this session."""
        if not self.session_id:
            raise LucidicNotInitializedError()
        from .client import Client
        kwargs = dict(kwargs)
        kwargs['session_id'] = self.session_id
        event_id = Client().create_event(type=type, **kwargs)
        return event_id

            