"""OpenAI provider handler for the Lucidic API"""
from typing import Optional

from .base_providers import BaseProvider
from lucidicai.singleton import singleton

@singleton
class OpenAIHandler(BaseProvider):
    def __init__(self, client):
        super().__init__(client)
        self._provider_name = "OpenAI"
        self.original_create = None
        self.pricing = {
            'gpt-4o': {'input': 2.5, 'output': 10.0},
            'gpt-4o-mini': {'input': 0.15, 'output': 0.6},
            'gpt-3.5-turbo': {'input': 0.5, 'output': 1.5},
            'o1': {'input': 15.0, 'output': 60.0},
            'o3-mini': {'input': 1.1, 'output': 4.4}
        }

    def _calculate_cost(self, model: str, usage) -> Optional[float]:
        if not usage:
            return None

        base_model = model.split('-20')[0] if '-20' in model else model
        
        for model_prefix, prices in self.pricing.items():
            if base_model.startswith(model_prefix):
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)
                
                input_cost = (input_tokens * prices['input']) / 1_000_000
                output_cost = (output_tokens * prices['output']) / 1_000_000
                
                return input_cost + output_cost
        
        return None

    def _format_messages(self, messages):
        if not messages:
            return "No messages provided"
        
        if isinstance(messages, list):
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    out = []
                    for content_piece in content:
                        if content_piece.get('type') == 'text':
                            out.append(content_piece)
                    return out
            content = messages[-1].get('content', '')
            out = []
            for content_piece in content:
                if content_piece.get('type') == 'text':
                    out.append(content_piece)
            return out
            
        return str(messages)

    def handle_response(self, response, kwargs, step = None):
        if not step:
            return response
            
        if not step.event_history:
            return response

        from openai import Stream
        if isinstance(response, Stream):
            return self._handle_stream_response(response, kwargs, step)
        return self._handle_regular_response(response, kwargs, step)

    def _handle_stream_response(self, response, kwargs, step):
        accumulated_response = ""

        def generate():
            nonlocal accumulated_response
            try:
                for chunk in response:
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            accumulated_response += delta.content
                    yield chunk
                
                step.update_event(
                    is_finished=True,
                    is_successful=True,
                    cost_added=None,
                    model=kwargs.get('model'),
                    result=accumulated_response
                )
            except Exception as e:
                step.update_event(
                    is_finished=True,
                    is_successful=False,
                    cost_added=None,
                    model=kwargs.get('model'),
                    result=f"Error during streaming: {str(e)}"
                )
                raise

        return generate()

    def _handle_regular_response(self, response, kwargs, step):
        try:
            response_text = (response.choices[0].message.content 
                           if hasattr(response, 'choices') and response.choices 
                           else str(response))

            cost = None
            if hasattr(response, 'usage'):
                model = response.model if hasattr(response, 'model') else kwargs.get('model')
                cost = self._calculate_cost(model, response.usage)

            step.update_event(
                is_finished=True,
                is_successful=True,
                cost_added=cost,
                model=response.model if hasattr(response, 'model') else kwargs.get('model'),
                result=response_text
            )

            return response

        except Exception as e:
            step.update_event(
                is_finished=True,
                is_successful=False,
                cost_added=None,
                model=kwargs.get('model'),
                result=f"Error processing response: {str(e)}"
            )
            raise

    def override(self):
        from openai.resources.chat import completions
        self.original_create = completions.Completions.create
        
        def patched_function(*args, **kwargs):
            step = kwargs.pop("step", self.client.session.active_step) if "step" in kwargs else self.client.session.active_step
            # Create event before API call
            if step:
                description = self._format_messages(kwargs.get('messages', ''))
                step.create_event(
                    description=description,
                    result="Waiting for response..."
                )
            
            # Make API call
            result = self.original_create(*args, **kwargs)
            return self.handle_response(result, kwargs, step=step)
        
        completions.Completions.create = patched_function

    def undo_override(self):
        if self.original_create:
            from openai.resources.chat import completions
            completions.Completions.create = self.original_create
            self.original_create = None