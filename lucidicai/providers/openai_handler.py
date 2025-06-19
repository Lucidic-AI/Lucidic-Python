"""OpenAI provider handler for the Lucidic API"""
from typing import Optional, Dict, Any, Callable, Union
import logging
import asyncio
from functools import wraps

from .base_providers import BaseProvider
from lucidicai.client import Client
from lucidicai.model_pricing import calculate_cost
from lucidicai.singleton import singleton

logger = logging.getLogger("Lucidic")

# Constants for messages
WAITING_RESPONSE = "Waiting for response..."
WAITING_STRUCTURED_RESPONSE = "Waiting for structured response..."
RESPONSE_RECEIVED = "Response received"
OPENAI_AGENTS_REQUEST = "OpenAI Agents SDK Request"
NO_ACTIVE_STEP = "No active step, skipping tracking"

@singleton
class OpenAIHandler(BaseProvider):
    """Handler for OpenAI API integration with Lucidic tracking"""
    
    def __init__(self):
        super().__init__()
        self.original_methods = {}
        self._provider_name = "OpenAI"
    
    def _wrap_api_call(
        self, 
        original_method: Callable,
        method_name: str,
        format_description: Callable[[Dict[str, Any]], tuple[str, list]],
        extract_response: Callable[[Any, Dict[str, Any]], str],
        is_async: bool = False
    ) -> Callable:
        """Generic wrapper for OpenAI API calls to reduce duplication
        
        Args:
            original_method: The original OpenAI method to wrap
            method_name: Name of the method for logging
            format_description: Function to format the event description
            extract_response: Function to extract response text
            is_async: Whether this is an async method
        """
        if is_async:
            @wraps(original_method)
            async def async_wrapper(*args, **kwargs):
                logger.info(f"[OpenAI Handler] Intercepted {method_name}")
                
                try:
                    session = Client().session
                    if session is None or session.active_step is None:
                        logger.info(f"[OpenAI Handler] {NO_ACTIVE_STEP}")
                        return await original_method(*args, **kwargs)
                    
                    # Add stream options if needed
                    if method_name.startswith("chat.completions") and kwargs.get('stream', False) and 'stream_options' not in kwargs:
                        kwargs['stream_options'] = {"include_usage": True}
                    
                    # Create event
                    description, images = format_description(kwargs)
                    event_id = session.create_event(
                        description=description,
                        result=WAITING_STRUCTURED_RESPONSE if "parse" in method_name else WAITING_RESPONSE,
                        screenshots=images if images else None,
                        model=kwargs.get('model', 'unknown')
                    )
                    
                    # Make API call
                    result = await original_method(*args, **kwargs)
                    
                    # Handle response
                    return self.handle_response(result, kwargs)
                    
                except Exception as e:
                    logger.error(f"Error in {method_name}: {str(e)}")
                    # Update event with error if we have one
                    try:
                        if 'event_id' in locals() and event_id:
                            session.update_event(
                                event_id=event_id,
                                is_finished=True,
                                is_successful=False,
                                result=f"Error: {str(e)}"
                            )
                    except Exception as update_error:
                        logger.debug(f"Failed to update event on error: {str(update_error)}")
                    raise
                    
            return async_wrapper
        else:
            @wraps(original_method)
            def sync_wrapper(*args, **kwargs):
                logger.info(f"[OpenAI Handler] Intercepted {method_name}")
                
                try:
                    session = Client().session
                    if session is None or session.active_step is None:
                        logger.info(f"[OpenAI Handler] {NO_ACTIVE_STEP}")
                        return original_method(*args, **kwargs)
                    
                    # Add stream options if needed
                    if method_name.startswith("chat.completions") and kwargs.get('stream', False) and 'stream_options' not in kwargs:
                        kwargs['stream_options'] = {"include_usage": True}
                    
                    # Create event
                    description, images = format_description(kwargs)
                    event_id = session.create_event(
                        description=description,
                        result=WAITING_STRUCTURED_RESPONSE if "parse" in method_name else WAITING_RESPONSE,
                        screenshots=images if images else None,
                        model=kwargs.get('model', 'unknown')
                    )
                    
                    # Make API call
                    result = original_method(*args, **kwargs)
                    
                    # Handle response
                    return self.handle_response(result, kwargs)
                    
                except Exception as e:
                    logger.error(f"Error in {method_name}: {str(e)}")
                    # Update event with error if we have one
                    try:
                        if 'event_id' in locals() and event_id:
                            session.update_event(
                                event_id=event_id,
                                is_finished=True,
                                is_successful=False,
                                result=f"Error: {str(e)}"
                            )
                    except Exception as update_error:
                        logger.debug(f"Failed to update event on error: {str(update_error)}")
                    raise
                    
            return sync_wrapper
    
    def _format_messages(self, messages: Any) -> tuple[str, list]:
        """Format messages for event description"""
        description = "Model request"
        images = []
        
        if not messages:
            return description, images
            
        if isinstance(messages, str):
            return messages, images
            
        # Handle message list
        if isinstance(messages, list):
            content_parts = []
            for message in messages:
                if isinstance(message, dict):
                    role = message.get('role', 'unknown')
                    content = message.get('content', '')
                    
                    if isinstance(content, str):
                        content_parts.append(f"{role}: {content}")
                    elif isinstance(content, list):
                        # Handle multimodal content
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict):
                                if item.get('type') == 'text':
                                    text_parts.append(item.get('text', ''))
                                elif item.get('type') == 'image_url':
                                    image_url = item.get('image_url', {})
                                    if isinstance(image_url, dict) and 'url' in image_url:
                                        images.append(image_url['url'])
                        
                        if text_parts:
                            content_parts.append(f"{role}: {' '.join(text_parts)}")
            
            description = '\n'.join(content_parts) if content_parts else "Model request"
            
        return description, images
    
    def _format_responses_description(self, kwargs: Dict[str, Any]) -> tuple[str, list]:
        """Format description for responses API calls"""
        input_messages = kwargs.get('input', [])
        if isinstance(input_messages, list) and input_messages:
            return str(input_messages), []
        return OPENAI_AGENTS_REQUEST, []
    
    def _extract_responses_text(self, result: Any) -> str:
        """Extract text from responses API result"""
        if not hasattr(result, 'output') or not result.output:
            return RESPONSE_RECEIVED
            
        if isinstance(result.output, list) and len(result.output) > 0:
            first_msg = result.output[0]
            if hasattr(first_msg, 'content'):
                content = first_msg.content
                if isinstance(content, list) and len(content) > 0:
                    # Handle ResponseOutputText objects
                    content_item = content[0]
                    if hasattr(content_item, 'text'):
                        return content_item.text
                    return str(content_item)
                return str(content)
        
        return RESPONSE_RECEIVED
    
    def override(self):
        """Override OpenAI methods with tracking versions"""
        try:
            # Import all required modules
            from openai.resources.chat import completions
            from openai.resources.beta.chat import completions as beta_completions
            from openai.resources.chat.completions import AsyncCompletions
            from openai.resources.beta.chat.completions import AsyncCompletions as BetaAsyncCompletions
            
            # Store original methods
            self.original_methods = {
                'chat.completions.create': completions.Completions.create,
                'beta.chat.completions.parse': beta_completions.Completions.parse,
                'async.chat.completions.create': AsyncCompletions.create,
                'async.beta.chat.completions.parse': BetaAsyncCompletions.parse
            }
            
            # Try to import responses API (may not exist in all versions)
            try:
                from openai.resources import responses
                from openai.resources.responses import AsyncResponses
                self.original_methods['responses.create'] = responses.Responses.create
                self.original_methods['async.responses.create'] = AsyncResponses.create
            except ImportError:
                logger.debug("Responses API not available in this OpenAI version")
            
            # Apply patches for chat completions
            completions.Completions.create = self._wrap_api_call(
                self.original_methods['chat.completions.create'],
                'chat.completions.create',
                lambda kwargs: self._format_messages(kwargs.get('messages', '')),
                lambda result, kwargs: "",  # Response handled by handle_response
                is_async=False
            )
            
            AsyncCompletions.create = self._wrap_api_call(
                self.original_methods['async.chat.completions.create'],
                'async chat.completions.create',
                lambda kwargs: self._format_messages(kwargs.get('messages', '')),
                lambda result, kwargs: "",
                is_async=True
            )
            
            # Apply patches for beta completions (structured output)
            def format_parse_description(kwargs):
                description, images = self._format_messages(kwargs.get('messages', ''))
                response_format = kwargs.get('response_format')
                if response_format:
                    description += f"\n[Structured Output: {response_format.__name__}]"
                return description, images
            
            beta_completions.Completions.parse = self._wrap_api_call(
                self.original_methods['beta.chat.completions.parse'],
                'beta.chat.completions.parse',
                format_parse_description,
                lambda result, kwargs: "",
                is_async=False
            )
            
            BetaAsyncCompletions.parse = self._wrap_api_call(
                self.original_methods['async.beta.chat.completions.parse'],
                'async beta.chat.completions.parse',
                format_parse_description,
                lambda result, kwargs: "",
                is_async=True
            )
            
            # Apply patches for responses API if available
            if 'responses.create' in self.original_methods:
                from openai.resources import responses
                from openai.resources.responses import AsyncResponses
                
                # Create specialized wrappers for responses API
                responses.Responses.create = self._create_responses_wrapper(
                    self.original_methods['responses.create'],
                    is_async=False
                )
                
                AsyncResponses.create = self._create_responses_wrapper(
                    self.original_methods['async.responses.create'],
                    is_async=True
                )
                
        except Exception as e:
            logger.error(f"Failed to override OpenAI methods: {str(e)}")
            raise
    
    def _create_responses_wrapper(self, original_method: Callable, is_async: bool = False) -> Callable:
        """Create a specialized wrapper for responses API"""
        if is_async:
            @wraps(original_method)
            async def async_wrapper(*args, **kwargs):
                logger.info("[OpenAI Handler] Intercepted async responses.create call")
                
                try:
                    session = Client().session
                    if session is None or session.active_step is None:
                        logger.info(f"[OpenAI Handler] {NO_ACTIVE_STEP}")
                        return await original_method(*args, **kwargs)
                    
                    # Create event
                    description, _ = self._format_responses_description(kwargs)
                    event_id = session.create_event(
                        description=description,
                        result=WAITING_RESPONSE,
                        model=kwargs.get('model', 'unknown')
                    )
                    
                    # Make API call
                    result = await original_method(*args, **kwargs)
                    
                    # Update event
                    response_text = self._extract_responses_text(result)
                    session.update_event(
                        event_id=event_id,
                        is_finished=True,
                        is_successful=True,
                        result=response_text,
                        model=kwargs.get('model', 'unknown')
                    )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in async responses.create: {str(e)}")
                    if 'event_id' in locals() and event_id and 'session' in locals():
                        try:
                            session.update_event(
                                event_id=event_id,
                                is_finished=True,
                                is_successful=False,
                                result=f"Error: {str(e)}"
                            )
                        except:
                            pass
                    raise
                    
            return async_wrapper
        else:
            @wraps(original_method)
            def sync_wrapper(*args, **kwargs):
                logger.info("[OpenAI Handler] Intercepted responses.create call")
                
                try:
                    session = Client().session
                    if session is None or session.active_step is None:
                        logger.info(f"[OpenAI Handler] {NO_ACTIVE_STEP}")
                        return original_method(*args, **kwargs)
                    
                    # Create event
                    description, _ = self._format_responses_description(kwargs)
                    event_id = session.create_event(
                        description=description,
                        result=WAITING_RESPONSE,
                        model=kwargs.get('model', 'unknown')
                    )
                    
                    # Make API call
                    result = original_method(*args, **kwargs)
                    
                    # Update event
                    response_text = self._extract_responses_text(result)
                    session.update_event(
                        event_id=event_id,
                        is_finished=True,
                        is_successful=True,
                        result=response_text,
                        model=kwargs.get('model', 'unknown')
                    )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in responses.create: {str(e)}")
                    if 'event_id' in locals() and event_id and 'session' in locals():
                        try:
                            session.update_event(
                                event_id=event_id,
                                is_finished=True,
                                is_successful=False,
                                result=f"Error: {str(e)}"
                            )
                        except:
                            pass
                    raise
                    
            return sync_wrapper

    def _is_using_anthropic_base_url(self, args, kwargs):
        """Check if we're using Anthropic base URL by inspecting the client
        
        This is more robust than string inspection but still not ideal.
        Consider adding explicit provider detection in the future.
        """
        # Check if first arg is a client instance
        if args and hasattr(args[0], '_base_url'):
            base_url = str(args[0]._base_url)
            return 'anthropic' in base_url.lower()
        
        # Check kwargs for client
        client = kwargs.get('client')
        if client and hasattr(client, '_base_url'):
            base_url = str(client._base_url)
            return 'anthropic' in base_url.lower()
            
        return False

    def handle_response(self, response, kwargs, session: Optional = None):
        """Handle the response from OpenAI API calls"""
        try:
            # Handle Anthropic responses
            if self._is_using_anthropic_base_url([], kwargs):
                return self._handle_anthropic_response(response, kwargs)
            
            # Handle streaming responses
            if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
                return self._handle_streaming_response(response, kwargs)
            
            # Handle standard responses
            return self._handle_standard_response(response, kwargs)
            
        except Exception as e:
            logger.error(f"Error handling response: {str(e)}")
            # Don't re-raise, just return the response
            return response
    
    def _handle_anthropic_response(self, response, kwargs):
        """Handle Anthropic-style responses"""
        # Let AnthropicHandler deal with it
        return response

    def _handle_streaming_response(self, response, kwargs):
        """Handle streaming responses with proper event updates"""
        from lucidicai.streaming import StreamingResponseWrapper
        return StreamingResponseWrapper(response, session=Client().session, kwargs=kwargs)

    def _handle_standard_response(self, response, kwargs):
        """Handle standard (non-streaming) responses"""
        try:
            session = Client().session
            if not session:
                return response
                
            # Extract content based on response type
            if hasattr(response, 'parsed'):
                # Beta parse response
                result = f"[Structured Output]\n{response.parsed}"
            elif hasattr(response, 'choices') and response.choices:
                # Standard completion
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    result = choice.message.content or "No content"
                else:
                    result = str(choice)
            else:
                result = str(response)
            
            # Calculate cost if possible
            cost = None
            if hasattr(response, 'usage') and response.usage:
                cost = calculate_cost(
                    kwargs.get('model', 'unknown'),
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            
            # Update the event
            session.update_event(
                is_finished=True,
                is_successful=True,
                result=result,
                cost_added=cost,
                model=kwargs.get('model', response.model if hasattr(response, 'model') else 'unknown')
            )
            
        except Exception as e:
            logger.error(f"Error updating event: {str(e)}")
        
        return response

    def undo_override(self):
        """Restore the original OpenAI methods"""
        try:
            # Restore chat completions
            if 'chat.completions.create' in self.original_methods:
                from openai.resources.chat import completions
                completions.Completions.create = self.original_methods['chat.completions.create']
                
            if 'beta.chat.completions.parse' in self.original_methods:
                from openai.resources.beta.chat import completions as beta_completions
                beta_completions.Completions.parse = self.original_methods['beta.chat.completions.parse']
                
            if 'async.chat.completions.create' in self.original_methods:
                from openai.resources.chat.completions import AsyncCompletions
                AsyncCompletions.create = self.original_methods['async.chat.completions.create']
                
            if 'async.beta.chat.completions.parse' in self.original_methods:
                from openai.resources.beta.chat.completions import AsyncCompletions as BetaAsyncCompletions
                BetaAsyncCompletions.parse = self.original_methods['async.beta.chat.completions.parse']
            
            # Restore responses API if it was patched
            if 'responses.create' in self.original_methods:
                from openai.resources import responses
                responses.Responses.create = self.original_methods['responses.create']
                
            if 'async.responses.create' in self.original_methods:
                from openai.resources.responses import AsyncResponses
                AsyncResponses.create = self.original_methods['async.responses.create']
            
            # Clear stored methods
            self.original_methods.clear()
            
        except Exception as e:
            logger.error(f"Error restoring OpenAI methods: {str(e)}")