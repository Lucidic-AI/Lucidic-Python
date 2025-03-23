from typing import Optional

from .base_providers import BaseProvider
from lucidicai.singleton import singleton

@singleton
class AnthropicHandler(BaseProvider):
    def __init__(self, client):
        super().__init__(client)
        self._provider_name = "Anthropic"
        self.original_create = None
        self.original_create_async = None
    
    def handle_response(self, response, kwargs, step = None):
        """Handle responses for Anthropic"""
        import anthropic
        from anthropic import AsyncStream, Stream
        
        # For streaming responses
        if isinstance(response, (AsyncStream, Stream)):
            input_messages = kwargs.get('messages', '')
            event = step.create_event(
                description=str(input_messages),
                result=None
            )
            
            accumulated_response = ""
            
            # For synchronous streams
            if isinstance(response, Stream):
                for chunk in response:
                    try:
                        if chunk.type == "message_start":
                            continue
                        elif chunk.type == "content_block_start":
                            if chunk.content_block.type == "text":
                                accumulated_response += chunk.content_block.text
                        elif chunk.type == "content_block_delta":
                            if chunk.delta.type == "text_delta":
                                accumulated_response += chunk.delta.text
                    except Exception as e:
                        event.update_event(
                            is_finished=True,
                            result=accumulated_response,
                            is_successful=False,
                            cost_added=None,
                            model=kwargs.get('model')
                        )
                        raise
            
            # Update final response before finishing
            event.update_event(result=accumulated_response)
            event.finish_event(
                is_successful=True,
                cost_added=None,  # Streaming doesn't provide token count
                model=kwargs.get('model')
            )
            
            return response
            
        # For non-streaming responses
        try:
            input_messages = kwargs.get('messages', '')
            event = step.create_event(
                description=str(input_messages),
                result=None
            )
            
            # Extract and update response text
            if hasattr(response, 'content'):
                response_text = response.content[0].text
            else:
                response_text = str(response)
            
            event.update_event(result=response_text)
            
            # Calculate token count if available
            token_count = None
            if hasattr(response, 'usage'):
                token_count = response.usage.input_tokens + response.usage.output_tokens
            
            # Finish event with metadata
            event.finish_event(
                is_successful=True,
                cost_added=token_count,
                model=response.model if hasattr(response, 'model') else kwargs.get('model')
            )
            
        except Exception as e:
            if step:
                # Try to update with error message if we fail
                event.update_event(
                    is_finished=True,
                    result=f"anthropic Error: {str(e)}",
                    is_successful=False,
                    cost_added=None,
                    model=kwargs.get('model')
                )
            raise
        
        return response

    def override(self):
        import anthropic
        
        # Store original methods
        self.original_create = anthropic.messages.Messages.create
        
        def patched_function(*args, **kwargs):
            step = kwargs.pop("step", self.client.session.active_step) if "step" in kwargs else self.client.session.active_step
            if not step:
                return self.original_create(*args, **kwargs)
            
            result = self.original_create(*args, **kwargs)
            return self.handle_response(result, kwargs, step=step)
            
        # Override the methods
        anthropic.messages.Messages.create = patched_function

    def undo_override(self):
        if self.original_create:
            import anthropic
            anthropic.messages.Messages.create = self.original_create
            self.original_create = None