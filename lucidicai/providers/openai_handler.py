"""OpenAI provider handler for the Lucidic API"""
from typing import Optional, Dict, Any, Callable, Union
import logging
import asyncio
from functools import wraps

from .base_providers import BaseProvider
from lucidicai.client import Client
from lucidicai.model_pricing import calculate_cost
from lucidicai.singleton import singleton
import lucidicai as lai

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
    
    # ========== Helper Methods ==========
    
    def _create_event_for_call(self, session, method_name: str, kwargs: Dict[str, Any], format_description: Callable) -> Optional[str]:
        """Create an event for an API call"""
        try:
            description, images = format_description(kwargs)
            
            # For streaming responses, don't set an initial result
            initial_result = None
            if not kwargs.get('stream', False):
                initial_result = WAITING_STRUCTURED_RESPONSE if "parse" in method_name else WAITING_RESPONSE
            
            return session.create_event(
                description=description,
                result=initial_result,
                screenshots=images if images else None,
                model=kwargs.get('model', 'unknown')
            )
        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            return None
    
    def _update_event_on_error(self, session, event_id: str, error: Exception) -> None:
        """Update event with error information"""
        try:
            if event_id and session:
                session.update_event(
                    event_id=event_id,
                    is_finished=True,
                    is_successful=False,
                    result=f"Error: {str(error)}"
                )
        except Exception as e:
            logger.debug(f"Failed to update event on error: {e}")
    
    def _prepare_streaming_kwargs(self, method_name: str, kwargs: Dict[str, Any]) -> None:
        """Prepare kwargs for streaming requests"""
        if method_name.startswith("chat.completions") and kwargs.get('stream', False):
            if 'stream_options' not in kwargs:
                kwargs['stream_options'] = {"include_usage": True}
    
    # ========== Wrapper Methods ==========
    
    def _wrap_api_call(
        self, 
        original_method: Callable,
        method_name: str,
        format_description: Callable[[Dict[str, Any]], tuple[str, list]],
        extract_response: Callable[[Any, Dict[str, Any]], str],
        is_async: bool = False
    ) -> Callable:
        """Generic wrapper for OpenAI API calls to reduce duplication"""
        
        async def _execute_async_call(original_method, args, kwargs, session, event_id):
            """Execute async API call with proper error handling"""
            try:
                # Create a copy of kwargs for the API call
                api_kwargs = kwargs.copy()
                
                # For streaming, pass the event_id through our internal kwargs
                if kwargs.get('stream', False):
                    kwargs['_event_id'] = event_id
                    
                result = await original_method(*args, **api_kwargs)
                return self.handle_response(result, kwargs)
            except Exception as e:
                self._update_event_on_error(session, event_id, e)
                raise
        
        def _execute_sync_call(original_method, args, kwargs, session, event_id):
            """Execute sync API call with proper error handling"""
            try:
                # Create a copy of kwargs for the API call
                api_kwargs = kwargs.copy()
                
                # For streaming, pass the event_id through our internal kwargs
                if kwargs.get('stream', False):
                    kwargs['_event_id'] = event_id
                
                result = original_method(*args, **api_kwargs)
                return self.handle_response(result, kwargs)
            except Exception as e:
                self._update_event_on_error(session, event_id, e)
                raise
        
        if is_async:
            @wraps(original_method)
            async def async_wrapper(*args, **kwargs):
                logger.info(f"[OpenAI Handler] Intercepted {method_name}")
                
                session = Client().session
                if session is None:
                    logger.info(f"[OpenAI Handler] No session, skipping tracking")
                    return await original_method(*args, **kwargs)

                
                # Prepare kwargs
                self._prepare_streaming_kwargs(method_name, kwargs)
                
                # Create event
                event_id = self._create_event_for_call(session, method_name, kwargs, format_description)
                
                # Execute call
                return await _execute_async_call(original_method, args, kwargs, session, event_id)
                    
            return async_wrapper
        else:
            @wraps(original_method)
            def sync_wrapper(*args, **kwargs):
                logger.info(f"[OpenAI Handler] Intercepted {method_name}")
                
                session = Client().session
                if session is None:
                    logger.info(f"[OpenAI Handler] No session, skipping tracking")
                    return original_method(*args, **kwargs)
                
                # Prepare kwargs
                self._prepare_streaming_kwargs(method_name, kwargs)
                
                # Create event
                event_id = self._create_event_for_call(session, method_name, kwargs, format_description)
                
                # Execute call
                return _execute_sync_call(original_method, args, kwargs, session, event_id)
                    
            return sync_wrapper
    
    # ========== Message Formatting Methods ==========
    
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
    
    # ========== Main Override Methods ==========
    
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
    
    # ========== Responses API Methods ==========
    
    def _get_agent_name_from_input(self, input_messages: list) -> Optional[str]:
        """Extract agent name from input messages"""
        if not isinstance(input_messages, list):
            return None
            
        # Look for agent information in messages
        for msg in input_messages:
            if isinstance(msg, dict):
                # Check for function call outputs containing agent info
                if msg.get('type') == 'function_call_output' and msg.get('output'):
                    try:
                        import json
                        output = json.loads(msg['output'])
                        if 'assistant' in output:
                            agent_name = output['assistant']
                            logger.debug(f"[OpenAI Handler] Detected agent from output: {agent_name}")
                            return agent_name
                    except:
                        pass
                
                # Legacy check for sender field (kept for compatibility)
                if msg.get('sender') == 'Agent':
                    agent_name = msg.get('name') or msg.get('agent_name')
                    if agent_name:
                        logger.debug(f"[OpenAI Handler] Detected agent from sender: {agent_name}")
                        return agent_name
        
        return None
    
    def _get_step_id_for_agent(self, agent_name: str) -> Optional[str]:
        """Get step ID for a specific agent from OpenAI Agents handler"""
        try:
            from lucidicai.providers.openai_agents_handler import OpenAIAgentsHandler
            agents_handler = OpenAIAgentsHandler()
            if hasattr(agents_handler, '_active_steps') and agent_name in agents_handler._active_steps:
                step_info = agents_handler._active_steps[agent_name]
                if not step_info.get('ended', False):
                    logger.info(f"[OpenAI Handler] Found step for {agent_name}: {step_info['step_id']}")
                    return step_info['step_id']
        except Exception as e:
            logger.debug(f"[OpenAI Handler] Could not find agent step: {e}")
        return None
    
    def _create_responses_wrapper(self, original_method: Callable, is_async: bool = False) -> Callable:
        """Create a specialized wrapper for responses API"""
        
        async def _handle_async_responses_call(original_method, args, kwargs):
            """Handle async responses API call"""
            session = Client().session
            if session is None:
                logger.info(f"[OpenAI Handler] No session, skipping tracking")
                return await original_method(*args, **kwargs)
            
            # Check for agent context
            agent_name = self._get_agent_name_from_input(kwargs.get('input', []))
            
            # Create event
            description, _ = self._format_responses_description(kwargs)
            if agent_name:
                description = f"[{agent_name}] {description}"
            
            # Check if we're in OpenAI Agents SDK mode
            try:
                from lucidicai.providers.openai_agents_handler import OpenAIAgentsHandler
                agents_handler = OpenAIAgentsHandler()
                
                # Only do this if the agents handler is actually instrumented and has active steps
                if (hasattr(agents_handler, '_is_instrumented') and agents_handler._is_instrumented and
                    hasattr(agents_handler, '_active_steps') and agents_handler._active_steps):
                    
                    # Check if this agent is different from current active agents
                    current_active_agents = [name for name, info in agents_handler._active_steps.items() 
                                           if not info.get('ended', False)]
                    
                    if agent_name and agent_name not in current_active_agents and current_active_agents:
                        # This is a handoff! End the previous step and create a new one
                        for prev_agent in current_active_agents:
                            prev_step_info = agents_handler._active_steps[prev_agent]
                            if not prev_step_info.get('ended', False):
                                # End the previous step
                                lai.end_step(
                                    step_id=prev_step_info['step_id'],
                                    state=f"Handoff from {prev_agent} to {agent_name}",
                                    action="Initiated handoff",
                                    goal=f"Transfer control to {agent_name}"
                                )
                                prev_step_info['ended'] = True
                        
                        # Create new step for this agent
                        new_step_id = lai.create_step(
                            state=f"Handoff: {agent_name}",
                            action=f"Continuing from previous agent",
                            goal="Process request"
                        )
                        
                        if new_step_id:
                            agents_handler._active_steps[agent_name] = {
                                'step_id': new_step_id,
                                'ended': False
                            }
                            logger.info(f"[OpenAI Handler] Created handoff step for {agent_name}: {new_step_id}")
            except ImportError:
                # OpenAI Agents SDK not available, skip this logic
                pass
            
            # Build event kwargs
            event_kwargs = {
                'description': description,
                'result': WAITING_RESPONSE,
                'model': kwargs.get('model', 'unknown')
            }
            
            # Try to find specific step for this agent
            if agent_name:
                step_id = self._get_step_id_for_agent(agent_name)
                if step_id:
                    event_kwargs['step_id'] = step_id
            
            event_id = session.create_event(**event_kwargs)
            
            try:
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
                self._update_event_on_error(session, event_id, e)
                raise
        
        def _handle_sync_responses_call(original_method, args, kwargs):
            """Handle sync responses API call"""
            session = Client().session
            if session is None:
                logger.info(f"[OpenAI Handler] No session, skipping tracking")
                return original_method(*args, **kwargs)
            
            # Check for agent context
            agent_name = self._get_agent_name_from_input(kwargs.get('input', []))
            
            # Create event
            description, _ = self._format_responses_description(kwargs)
            if agent_name:
                description = f"[{agent_name}] {description}"
            
            # Check if we're in OpenAI Agents SDK mode by looking for active agents handler
            try:
                from lucidicai.providers.openai_agents_handler import OpenAIAgentsHandler
                agents_handler = OpenAIAgentsHandler()
                
                # Only do this if the agents handler is actually instrumented and has active steps
                if (hasattr(agents_handler, '_is_instrumented') and agents_handler._is_instrumented and
                    hasattr(agents_handler, '_active_steps') and agents_handler._active_steps):
                    
                    # Check if this agent is different from current active agents
                    current_active_agents = [name for name, info in agents_handler._active_steps.items() 
                                           if not info.get('ended', False)]
                    
                    if agent_name and agent_name not in current_active_agents and current_active_agents:
                        # This is a handoff! End the previous step and create a new one
                        for prev_agent in current_active_agents:
                            prev_step_info = agents_handler._active_steps[prev_agent]
                            if not prev_step_info.get('ended', False):
                                # End the previous step
                                lai.end_step(
                                    step_id=prev_step_info['step_id'],
                                    state=f"Handoff from {prev_agent} to {agent_name}",
                                    action="Initiated handoff",
                                    goal=f"Transfer control to {agent_name}"
                                )
                                prev_step_info['ended'] = True
                        
                        # Create new step for this agent
                        new_step_id = lai.create_step(
                            state=f"Handoff: {agent_name}",
                            action=f"Continuing from previous agent",
                            goal="Process request"
                        )
                        
                        if new_step_id:
                            agents_handler._active_steps[agent_name] = {
                                'step_id': new_step_id,
                                'ended': False
                            }
                            logger.info(f"[OpenAI Handler] Created handoff step for {agent_name}: {new_step_id}")
            except ImportError:
                # OpenAI Agents SDK not available, skip this logic
                pass
            
            # Build event kwargs
            event_kwargs = {
                'description': description,
                'result': WAITING_RESPONSE,
                'model': kwargs.get('model', 'unknown')
            }
            
            # Try to find specific step for this agent
            if agent_name:
                step_id = self._get_step_id_for_agent(agent_name)
                if step_id:
                    event_kwargs['step_id'] = step_id
            
            event_id = session.create_event(**event_kwargs)
            
            try:
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
                self._update_event_on_error(session, event_id, e)
                raise
        
        if is_async:
            @wraps(original_method)
            async def async_wrapper(*args, **kwargs):
                logger.info("[OpenAI Handler] Intercepted async responses.create call")
                return await _handle_async_responses_call(original_method, args, kwargs)
            return async_wrapper
        else:
            @wraps(original_method)
            def sync_wrapper(*args, **kwargs):
                logger.info("[OpenAI Handler] Intercepted responses.create call")
                return _handle_sync_responses_call(original_method, args, kwargs)
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

    # ========== Response Handling Methods ==========
    
    def handle_response(self, response, kwargs, session: Optional = None):
        """Handle the response from OpenAI API calls"""
        try:
            # Handle Anthropic responses
            if self._is_using_anthropic_base_url([], kwargs):
                return self._handle_anthropic_response(response, kwargs)
            
            # Check if this is a Stream object
            from openai import Stream
            from openai import AsyncStream
            
            if isinstance(response, (Stream, AsyncStream)):
                logger.debug(f"[OpenAI Handler] Detected Stream response")
                return self._handle_streaming_response(response, kwargs)
            
            # Handle streaming responses by checking kwargs
            if kwargs.get('stream', False):
                logger.debug(f"[OpenAI Handler] Detected streaming from kwargs")
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
                usage_info = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                cost = calculate_cost(
                    kwargs.get('model', 'unknown'),
                    usage_info
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

    # ========== Cleanup Methods ==========
    
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