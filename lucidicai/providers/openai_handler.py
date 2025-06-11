"""OpenAI provider handler for the Lucidic API"""
from typing import Optional

from .base_providers import BaseProvider
from lucidicai.model_pricing import calculate_cost
from lucidicai.singleton import singleton

@singleton
class OpenAIHandler(BaseProvider):
    def __init__(self, client):
        super().__init__(client)
        self._provider_name = "OpenAI"
        self.original_create = None
        self.original_parse = None

    def _format_messages(self, messages):
        if not messages:
            return "No messages provided"
        
        if isinstance(messages, list):
            for msg in messages:
                content = msg.get('content', '')
                out = []
                images = []
                if isinstance(content, list):
                    for content_piece in content:
                        if content_piece.get('type') == 'text':
                            out.append(content_piece)
                        elif content_piece.get('type') == 'image_url':
                            image_str = content_piece.get('image_url').get('url')
                            images.append(image_str[image_str.find(',') + 1:])
                        elif content_piece.get('type') == 'output_text':
                            out.append(content_piece)
                elif isinstance(content, str):
                    out.append(content)
                return out, images
        
        return str(messages)

    def handle_response(self, response, kwargs, event = None):
        if not event:
            return response

        from openai import Stream
        if isinstance(response, Stream):
            return self._handle_stream_response(response, kwargs, event)
        return self._handle_regular_response(response, kwargs, event)

    def _handle_stream_response(self, response, kwargs, event):
        accumulated_response = ""

        def generate():
            nonlocal accumulated_response
            try:
                for chunk in response:
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            accumulated_response += delta.content
                    
                    # Handle final chunk with usage information
                    if hasattr(chunk, 'usage') and chunk.usage:
                        cost = None
                        model = kwargs.get('model')
                        cost = calculate_cost(model, dict(chunk.usage))
                        
                        event.update_event(
                            is_finished=True,
                            is_successful=True,
                            cost_added=cost,
                            model=model,
                            result=accumulated_response
                        )
                    
                    yield chunk
                    
            except Exception as e:
                event.update_event(
                    is_finished=True,
                    is_successful=False,
                    cost_added=None,
                    model=kwargs.get('model'),
                    result=f"Error during streaming: {str(e)}"
                )
                raise

        return generate()

    def _handle_regular_response(self, response, kwargs, event):
        try:
            # Handle structured output response (beta.chat.completions.parse)
            if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0].message, 'parsed'):
                response_text = str(response.choices[0].message.parsed)
            else:
                # Handle regular response
                response_text = (response.choices[0].message.content 
                               if hasattr(response, 'choices') and response.choices 
                               else str(response))

            cost = None
            if hasattr(response, 'usage'):
                model = response.model if hasattr(response, 'model') else kwargs.get('model')
                cost = calculate_cost(model, dict(response.usage))

            event.update_event(
                is_finished=True,
                is_successful=True,
                cost_added=cost,
                model=response.model if hasattr(response, 'model') else kwargs.get('model'),
                result=response_text, 
                
            )

            return response

        except Exception as e:
            event.update_event(
                is_finished=True,
                is_successful=False,
                cost_added=None,
                model=kwargs.get('model'),
                result=f"Error processing response: {str(e)}"
            )
            raise

    def override(self):
        from openai.resources.chat import completions
        from openai.resources.beta.chat import completions as beta_completions
        
        # Store original methods
        self.original_create = completions.Completions.create
        self.original_parse = beta_completions.Completions.parse
        
        def patched_create_function(*args, **kwargs):
            step = kwargs.pop("step", self.client.session.active_step) if "step" in kwargs else self.client.session.active_step
            
            # Add stream_options for usage tracking if streaming is enabled
            if kwargs.get('stream', False) and 'stream_options' not in kwargs:
                kwargs['stream_options'] = {"include_usage": True}
            
            # Create event before API call
            if step:
                description, images = self._format_messages(kwargs.get('messages', ''))
                event = step.create_event(
                    description=description,
                    result="Waiting for response...",
                    screenshots=images
                )
                
            
            # Make API call
            result = self.original_create(*args, **kwargs)
            return self.handle_response(result, kwargs, event=event)
        
        def patched_parse_function(*args, **kwargs):
            step = kwargs.pop("step", self.client.session.active_step) if "step" in kwargs else self.client.session.active_step
            
            # Create event before API call
            if step:
                description, images = self._format_messages(kwargs.get('messages', ''))
                # Add info about structured output format
                response_format = kwargs.get('response_format')
                if response_format:
                    description += f"\n[Structured Output: {response_format.__name__}]"
                
                event = step.create_event(
                    description=description,
                    result="Waiting for structured response...",
                    screenshots=images
                )
                
            
            # Make API call
            result = self.original_parse(*args, **kwargs)
            return self.handle_response(result, kwargs, event=event)
        
        # Apply patches
        completions.Completions.create = patched_create_function
        beta_completions.Completions.parse = patched_parse_function

    def undo_override(self):
        if self.original_create:
            from openai.resources.chat import completions
            completions.Completions.create = self.original_create
            self.original_create = None
            
        if self.original_parse:
            from openai.resources.beta.chat import completions as beta_completions
            beta_completions.Completions.parse = self.original_parse
            self.original_parse = None