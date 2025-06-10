"""Pydantic AI provider handler for the Lucidic API"""
from typing import Any, Dict, Optional

from .base_providers import BaseProvider
from lucidicai.model_pricing import calculate_cost
from lucidicai.singleton import singleton


@singleton
class PydanticAIHandler(BaseProvider):
    """Handler for tracking PydanticAI model interactions with Lucidic"""
    
    def __init__(self, client):
        super().__init__(client)
        self._provider_name = "PydanticAI"
        self._original_anthropic_request = None
        self._original_anthropic_request_stream = None
        self._original_openai_request = None
        self._original_openai_request_stream = None

    def handle_response(self, response, kwargs, event=None):
        """Handle responses from Pydantic AI models"""
        if not event:
            return response
        return response

    def _format_messages(self, messages):
        """Format messages for event description"""
        if not messages:
            return "No messages provided"
        
        # Extract text content from messages
        formatted_messages = []
        for message in messages:
            if hasattr(message, 'content'):
                if isinstance(message.content, str):
                    formatted_messages.append(message.content)
                elif isinstance(message.content, list):
                    # Handle structured content
                    for item in message.content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            formatted_messages.append(item.get('text', ''))
            elif isinstance(message, dict):
                formatted_messages.append(str(message.get('content', message)))
            else:
                formatted_messages.append(str(message))
        
        return ' | '.join(formatted_messages[:3])  # Limit to first 3 messages

    def _handle_response(self, response, event, messages, model_settings):
        """Handle non-streaming response"""
        if not event:
            return response
        
        try:
            # Extract response text and usage information
            response_text = self._extract_response_text(response)
            usage_info = self._extract_usage_info(response)
            model_name = self._extract_model_name(response, model_settings)
            
            # Calculate cost if usage info is available
            cost = None
            if usage_info and model_name:
                cost = calculate_cost(model_name, usage_info)
            
            event.update_event(
                is_finished=True,
                is_successful=True,
                cost_added=cost,
                model=model_name,
                result=response_text
            )
            
        except Exception as e:
            event.update_event(
                is_finished=True,
                is_successful=False,
                result=f"Error processing response: {str(e)}"
            )
        
        return response

    def _wrap_stream(self, original_stream, event, messages, model_settings):
        """Wrap streaming response to track accumulation"""
        
        class StreamWrapper:
            def __init__(self, original_stream, event, handler, model_settings):
                self._original_stream = original_stream
                self._event = event
                self._handler = handler
                self._model_settings = model_settings
                self._accumulated_text = ""
                self._iterator = None
                
            def __aiter__(self):
                """Return an async iterator that properly implements the protocol"""
                return AsyncStreamIterator(self._original_stream, self._event, self._handler, self._model_settings)
                
            def stream_text(self, delta=True):
                """Return the wrapped stream iterator for compatibility with PydanticAI"""
                return AsyncStreamIterator(self._original_stream, self._event, self._handler, self._model_settings)
                
            def stream(self):
                """Deprecated compatibility method - use stream_text instead"""
                return self.stream_text(delta=True)
                
            # Delegate other methods to the original stream
            def __getattr__(self, name):
                return getattr(self._original_stream, name)
                
        class AsyncStreamIterator:
            def __init__(self, original_stream, event, handler, model_settings):
                self._original_stream = original_stream
                self._event = event
                self._handler = handler
                self._model_settings = model_settings
                self._accumulated_text = ""
                self._original_iterator = None
                
            def __aiter__(self):
                """Return self to implement async iterator protocol"""
                return self
                
            async def __anext__(self):
                """Implement async iterator protocol"""
                if self._original_iterator is None:
                    # Initialize the original iterator - StreamedResponse is directly iterable
                    self._original_iterator = self._original_stream.__aiter__()
                    
                try:
                    # Get the next chunk from original iterator
                    chunk = await self._original_iterator.__anext__()
                    
                    # Extract text content from the StreamedResponse chunk
                    chunk_text = self._handler._extract_chunk_text(chunk)
                    if chunk_text:
                        self._accumulated_text += chunk_text
                    
                    return chunk
                    
                except StopAsyncIteration:
                    # Stream is done, update the event with accumulated text
                    if self._event and not self._event.is_finished:
                        model_name = self._handler._extract_model_name(None, self._model_settings)
                        # Try to get usage info from the original stream
                        usage_info = None
                        if hasattr(self._original_stream, 'usage'):
                            usage_info = self._handler._extract_usage_info(self._original_stream)
                        
                        cost = None
                        if usage_info and model_name:
                            cost = calculate_cost(model_name, usage_info)
                        
                        final_result = self._accumulated_text or "No content streamed"
                        
                        self._event.update_event(
                            is_finished=True,
                            is_successful=True,
                            model=model_name,
                            cost_added=cost,
                            result=final_result
                        )
                    
                    # Re-raise StopAsyncIteration to end iteration
                    raise
                    
                except Exception as e:
                    # Handle errors
                    if self._event and not self._event.is_finished:
                        self._event.update_event(
                            is_finished=True,
                            is_successful=False,
                            result=f"Error during streaming: {str(e)}"
                        )
                    raise
        
        return StreamWrapper(original_stream, event, self, model_settings)

    def _extract_response_text(self, response):
        """Extract text content from response"""
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'content'):
            if isinstance(response.content, str):
                return response.content
            elif isinstance(response.content, list):
                # Extract text from structured content
                text_parts = []
                for item in response.content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                return ' '.join(text_parts)
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content
        
        return str(response)

    def _extract_chunk_text(self, chunk):
        """Extract text from PydanticAI StreamedResponse chunk"""
        if not chunk:
            return ""
        
        # Try direct string check first
        if isinstance(chunk, str):
            return chunk
        
        # PydanticAI StreamedResponse chunks can have different formats
        # Try various attributes that might contain the text content
        
        # Check for delta content (common in streaming responses)
        if hasattr(chunk, 'delta') and chunk.delta:
            # PydanticAI uses content_delta attribute
            if hasattr(chunk.delta, 'content_delta') and chunk.delta.content_delta:
                return chunk.delta.content_delta
            elif hasattr(chunk.delta, 'content') and chunk.delta.content:
                return chunk.delta.content
            elif hasattr(chunk.delta, 'text') and chunk.delta.text:
                return chunk.delta.text
        
        # Check for direct text content
        if hasattr(chunk, 'text') and chunk.text:
            return chunk.text
        elif hasattr(chunk, 'content') and chunk.content:
            return chunk.content
        
        # Check for choices (OpenAI style)
        if hasattr(chunk, 'choices') and chunk.choices:
            for choice in chunk.choices:
                if hasattr(choice, 'delta') and choice.delta:
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        return choice.delta.content
                elif hasattr(choice, 'text') and choice.text:
                    return choice.text
        
        return ""

    def _extract_usage_info(self, response_or_chunk):
        """Extract usage information from response or chunk and normalize token keys"""
        if not response_or_chunk:
            return None
        
        # Common usage patterns
        usage_attrs = ['usage', 'usage_metadata', 'token_usage']
        
        for attr in usage_attrs:
            if hasattr(response_or_chunk, attr):
                usage = getattr(response_or_chunk, attr)
                if usage:
                    # Convert to dict format expected by calculate_cost
                    if hasattr(usage, '__dict__'):
                        usage_dict = usage.__dict__
                    elif isinstance(usage, dict):
                        usage_dict = usage
                    else:
                        continue
                    
                    # Normalize token keys for PydanticAI format
                    normalized_usage = {}
                    
                    # Map PydanticAI token keys to standard format
                    if 'request_tokens' in usage_dict:
                        normalized_usage['prompt_tokens'] = usage_dict['request_tokens']
                        normalized_usage['input_tokens'] = usage_dict['request_tokens']
                    
                    if 'response_tokens' in usage_dict:
                        normalized_usage['completion_tokens'] = usage_dict['response_tokens']
                        normalized_usage['output_tokens'] = usage_dict['response_tokens']
                    
                    # Copy other standard keys if they exist
                    for key in ['prompt_tokens', 'completion_tokens', 'input_tokens', 'output_tokens', 'total_tokens']:
                        if key in usage_dict:
                            normalized_usage[key] = usage_dict[key]
                    
                    # Copy all original keys for completeness
                    normalized_usage.update(usage_dict)
                    
                    return normalized_usage
        
        return None

    def _extract_model_name(self, response_or_chunk, model_settings=None):
        """Extract model name from response or settings"""
        # Try to get from response first
        if response_or_chunk and hasattr(response_or_chunk, 'model'):
            return response_or_chunk.model
        
        # Try from model settings
        if model_settings and isinstance(model_settings, dict):
            return model_settings.get('model')
        
        # Default fallback for PydanticAI models
        return "claude-3-5-sonnet-20241022"

    def _is_final_chunk(self, chunk):
        """Check if this is the final chunk with usage information"""
        return hasattr(chunk, 'usage') and chunk.usage is not None

    def _wrap_request(self, model_instance, messages, model_settings, model_request_parameters, original_method):
        """Wrap regular request method to track LLM calls"""
        # Create event before API call
        event = None
        from lucidicai.client import Client
        step = Client().session.active_step if Client().session else None
        
        if step:
            description = self._format_messages(messages)
            event = step.create_event(
                description=description,
                result="Waiting for response..."
            )
        
        async def async_wrapper():
            try:
                # Make the original API call
                response = await original_method(model_instance, messages, model_settings, model_request_parameters)
                
                # Handle the response
                return self._handle_response(response, event, messages, model_settings)
                
            except Exception as e:
                if event:
                    event.update_event(
                        is_finished=True,
                        is_successful=False,
                        result=f"Error during request: {str(e)}"
                    )
                raise
        
        return async_wrapper()
    
    def _wrap_request_stream_context_manager(self, model_instance, messages, model_settings, model_request_parameters, original_method):
        """Return an async context manager for streaming requests"""
        # Create event before API call
        event = None
        from lucidicai.client import Client
        step = Client().session.active_step if Client().session else None
        
        if step:
            description = self._format_messages(messages)
            event = step.create_event(
                description=description,
                result="Streaming response..."
            )
        
        class WrappedStreamContextManager:
            def __init__(self, original_method, model_instance, messages, model_settings, model_request_parameters, event, handler):
                self.original_method = original_method
                self.model_instance = model_instance
                self.messages = messages
                self.model_settings = model_settings
                self.model_request_parameters = model_request_parameters
                self.event = event
                self.handler = handler
                self.original_context_manager = None
            
            async def __aenter__(self):
                try:
                    # Get the original context manager (don't await it, it's already a context manager)
                    self.original_context_manager = self.original_method(
                        self.model_instance, self.messages, self.model_settings, self.model_request_parameters
                    )
                    
                    # Enter the original context manager to get the stream
                    original_stream = await self.original_context_manager.__aenter__()
                    
                    # Wrap the stream to capture the actual streamed content
                    return self.handler._wrap_stream(original_stream, self.event, self.messages, self.model_settings)
                    
                except Exception as e:
                    if self.event:
                        self.event.update_event(
                            is_finished=True,
                            is_successful=False,
                            result=f"Error during streaming: {str(e)}"
                        )
                    raise
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                # The stream wrapper handles event finalization, so we just delegate
                if self.original_context_manager:
                    return await self.original_context_manager.__aexit__(exc_type, exc_val, exc_tb)
        
        return WrappedStreamContextManager(original_method, model_instance, messages, model_settings, model_request_parameters, event, self)

    async def _wrap_request_stream(self, model_instance, messages, model_settings, model_request_parameters, original_method):
        """Wrap streaming request method"""
        # Create event before API call
        event = None
        from lucidicai.client import Client
        step = Client().session.active_step if Client().session else None
        
        if step:
            description = self._format_messages(messages)
            event = step.create_event(
                description=description,
                result="Streaming response..."
            )
        
        try:
            # Get the original stream
            original_stream = await original_method(model_instance, messages, model_settings, model_request_parameters)
            
            # Return wrapped stream
            return self._wrap_stream(original_stream, event, messages, model_settings)
            
        except Exception as e:
            if event:
                event.update_event(
                    is_finished=True,
                    is_successful=False,
                    result=f"Error during streaming: {str(e)}"
                )
            raise

    def wrap_agent(self, agent):
        """
        Wrap a PydanticAI agent to enable tracking.
        
        Note: This method is deprecated as we now use monkey-patching.
        The handler automatically tracks all PydanticAI model calls when override() is called.
        """
        # No longer needed due to monkey-patching approach
        return agent

    def override(self):
        """
        Override PydanticAI model methods to enable automatic tracking.
        
        This method uses monkey-patching to intercept calls to PydanticAI's 
        AnthropicModel and OpenAIModel request methods, allowing Lucidic to
        track all LLM interactions automatically.
        """
        try:
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.models.openai import OpenAIModel
            
            # Store original methods for restoration later
            self._original_anthropic_request = AnthropicModel.request
            self._original_anthropic_request_stream = AnthropicModel.request_stream
            self._original_openai_request = OpenAIModel.request
            self._original_openai_request_stream = OpenAIModel.request_stream
            
            # Create patched methods for Anthropic models
            def patched_anthropic_request(model_instance, messages, model_settings=None, model_request_parameters=None):
                return self._wrap_request(model_instance, messages, model_settings, model_request_parameters, self._original_anthropic_request)
            
            def patched_anthropic_request_stream(model_instance, messages, model_settings=None, model_request_parameters=None):
                return self._wrap_request_stream_context_manager(model_instance, messages, model_settings, model_request_parameters, self._original_anthropic_request_stream)
            
            # Create patched methods for OpenAI models
            def patched_openai_request(model_instance, messages, model_settings=None, model_request_parameters=None):
                return self._wrap_request(model_instance, messages, model_settings, model_request_parameters, self._original_openai_request)
            
            def patched_openai_request_stream(model_instance, messages, model_settings=None, model_request_parameters=None):
                return self._wrap_request_stream_context_manager(model_instance, messages, model_settings, model_request_parameters, self._original_openai_request_stream)
            
            # Apply the patches
            AnthropicModel.request = patched_anthropic_request
            AnthropicModel.request_stream = patched_anthropic_request_stream
            OpenAIModel.request = patched_openai_request
            OpenAIModel.request_stream = patched_openai_request_stream
            
        except ImportError:
            # PydanticAI models not available, skip patching
            pass

    def undo_override(self):
        """
        Restore original PydanticAI model methods.
        
        This method restores the original, unpatched methods to their
        respective model classes, effectively disabling Lucidic tracking.
        """
        try:
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.models.openai import OpenAIModel
            
            # Restore original methods if they were previously stored
            if hasattr(self, '_original_anthropic_request'):
                AnthropicModel.request = self._original_anthropic_request
                AnthropicModel.request_stream = self._original_anthropic_request_stream
                OpenAIModel.request = self._original_openai_request
                OpenAIModel.request_stream = self._original_openai_request_stream
                
        except ImportError:
            # PydanticAI models not available, nothing to restore
            pass