from typing import Optional
from .base_providers import BaseProvider
from lucidicai.session import Session, Step
from lucidicai.singleton import singleton

@singleton
class OpenAIHandler(BaseProvider):
    def __init__(self, client):
        super().__init__(client)
        self._provider_name = "OpenAI"
        self.original_create = None

    def handle_response(self, response, kwargs, session: Optional[Session] = None):
        if not session:
            return response

        from openai import Stream
        if isinstance(response, Stream):
            return self._handle_stream_response(response, kwargs, session)
        return self._handle_regular_response(response, kwargs, session)

    def _handle_stream_response(self, response, kwargs, session):
        accumulated_response = ""

        # Create initial step
        step = Step(
            session=session,
            goal=str(kwargs.get('messages', '')),
            action="Processing..."
        )

        def generate():
            nonlocal accumulated_response
            for chunk in response:
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content
                    if content:
                        accumulated_response += content
                yield chunk

            # Pass accumulated_response as action in finish_step
            step.update_step(
                is_successful=True,
                cost=None,
                model=kwargs.get('model'),
                action=accumulated_response
            )

        return generate()

    def _handle_regular_response(self, response, kwargs, session):
        # Create initial step
        step = Step(
            session=session,
            goal=str(kwargs.get('messages', '')),
            action="Processing..."
        )

        # Process response
        response_text = (response.choices[0].message.content
                        if hasattr(response, 'choices') and response.choices
                        else str(response))

        token_count = None
        if hasattr(response, 'usage'):
            token_count = (getattr(response.usage, 'total_tokens', None) or
                          (getattr(response.usage, 'prompt_tokens', 0) +
                           getattr(response.usage, 'completion_tokens', 0)))

        # Pass response_text as action in finish_step
        step.update_step(
            is_successful=True,
            cost=token_count,
            model=response.model if hasattr(response, 'model') else kwargs.get('model'),
            action=response_text
        )

        return response

    def override(self):
        from openai.resources.chat import completions
        self.original_create = completions.Completions.create
        
        def patched_function(*args, **kwargs):
            session = kwargs.pop("session", self.client.session) if "session" in kwargs else self.client.session
            result = self.original_create(*args, **kwargs)
            return self.handle_response(result, kwargs, session=session)
            
        completions.Completions.create = patched_function

    def undo_override(self):
        if self.original_create:
            from openai.resources.chat import completions
            completions.Completions.create = self.original_create
            self.original_create = None