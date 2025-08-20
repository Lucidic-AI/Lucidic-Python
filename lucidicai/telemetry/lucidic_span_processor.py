"""Custom span processor for real-time Lucidic event handling

Updated to stamp spans with the correct session id from async-safe
context, and to create events for that session without mutating the
global client session.
"""
import os
import logging
import json
from typing import Optional, Dict, Any
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import Span, SpanProcessor
from opentelemetry.trace import StatusCode
from opentelemetry.semconv_ai import SpanAttributes

from lucidicai.client import Client
from lucidicai.model_pricing import calculate_cost
from lucidicai.context import current_session_id
from .utils.image_storage import get_stored_images, clear_stored_images, get_image_by_placeholder
from .utils.text_storage import get_stored_text, clear_stored_texts

logger = logging.getLogger("Lucidic")
DEBUG = os.getenv("LUCIDIC_DEBUG", "False") == "True"
VERBOSE = os.getenv("LUCIDIC_VERBOSE", "False") == "True"


class LucidicSpanProcessor(SpanProcessor):
    """
    Real-time span processor that creates Lucidic events as spans start
    and updates them as spans end, maintaining the current SDK behavior
    """
    
    def __init__(self):
        self.span_to_event = {}  # Map span_id to event_id
        self.span_contexts = {}  # Store span start data
        
    def on_start(self, span: Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """Called when a span is started - create Lucidic event immediately"""
        try:
            if DEBUG:
                logger.info(f"[SpanProcessor] on_start called for span: {span.name}")
                # logger.info(f"[SpanProcessor] Span attributes at start: {dict(span.attributes or {})}")
                
            # Stamp session id from contextvars if available
            try:
                sid = current_session_id.get(None)
                if sid:
                    span.set_attribute('lucidic.session_id', sid)
            except Exception:
                pass

            client = Client()
            # Only process LLM spans
            if not self._is_llm_span(span):
                if DEBUG:
                    logger.info(f"[SpanProcessor] Skipping non-LLM span: {span.name}")
                return
            
            # Store span info for processing - we'll create the event on_end when all attributes are available
            span_id = span.get_span_context().span_id
            self.span_contexts[span_id] = {
                'start_time': span.start_time,
                'name': span.name,
                'attributes': dict(span.attributes or {}),
                'span': span
            }
            
            if DEBUG:
                logger.info(f"[SpanProcessor] Stored span {span_id} for later processing")
                
        except Exception as e:
            logger.error(f"Error in on_start: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
    
    def on_end(self, span: Span) -> None:
        """Called when a span ends - create and complete the Lucidic event"""
        try:
            span_id = span.get_span_context().span_id
            
            if DEBUG:
                logger.info(f"[SpanProcessor] on_end called for span: {span.name}")
                # logger.info(f"[SpanProcessor] Span attributes at end: {dict(span.attributes or {})}")
                # logger.info(f"[SpanProcessor] Tracked span contexts: {list(self.span_contexts.keys())}")
                """
                # Log any attributes that might contain message data
                attrs = dict(span.attributes or {})
                for key, value in attrs.items():
                    if 'message' in key.lower() or 'prompt' in key.lower() or 'content' in key.lower():
                        logger.info(f"[SpanProcessor] Found potential message attr: {key} = {value[:200] if isinstance(value, str) else value}")
                """
            
            # Check if we have context for this span
            if span_id not in self.span_contexts:
                if DEBUG:
                    logger.warning(f"[SpanProcessor] No context found for span {span_id}")
                return
                
            client = Client()
            span_context = self.span_contexts.pop(span_id, {})
            
            # Create event with all the attributes now available
            event_id = self._create_event_from_span_end(span, client)
            
            if DEBUG:
                logger.info(f"[SpanProcessor] Created and completed event {event_id} for span {span_id}")
            
            # Clear thread-local images and texts after processing
            clear_stored_images()
            clear_stored_texts()
            
        except Exception as e:
            logger.error(f"Error in on_end: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
    
    def _is_llm_span(self, span: Span) -> bool:
        """Check if this is an LLM-related span with actual LLM content"""
        # Check if it's an agent span without LLM content
        if span.attributes:
            attrs = dict(span.attributes)
            
            # Skip agent spans that don't have prompt/completion attributes
            if attrs.get('gen_ai.operation.name') == 'agent':
                # Check if it has actual LLM content
                has_prompts = any(k for k in attrs.keys() if 'prompt' in k.lower())
                has_completions = any(k for k in attrs.keys() if 'completion' in k.lower())
                if not has_prompts and not has_completions:
                    if DEBUG:
                        logger.info(f"[SpanProcessor] Skipping agent span without LLM content: {span.name}")
                    return False
        
        # Check span name
        span_name_lower = span.name.lower()
        llm_patterns = ['openai', 'anthropic', 'chat', 'completion', 'embedding', 'gemini', 'claude', 'bedrock', 'vertex', 'cohere', 'groq']
        
        if any(pattern in span_name_lower for pattern in llm_patterns):
            return True
            
        # Check attributes
        if span.attributes:
            for key in span.attributes:
                if isinstance(key, str) and (key.startswith('gen_ai.') or key.startswith('llm.')):
                    return True
                    
        return False
    
    def _create_event_from_span_start(self, span: Span, client: Client) -> Optional[str]:
        """Create event when span starts"""
        try:
            attributes = dict(span.attributes or {})
            
            # Extract description
            if DEBUG:
                logger.info(f"[SpanProcessor -- DEBUG] Extracting Description from span start: {span}")
            description = self._extract_description(span, attributes)
            
            # Extract images
            images = self._extract_images(attributes)
            
            # Get model
            model = (
                attributes.get(SpanAttributes.LLM_REQUEST_MODEL) or
                attributes.get('gen_ai.request.model') or
                attributes.get('llm.model') or
                'unknown'
            )
            
            # Initial result based on whether it's streaming
            is_streaming = attributes.get(SpanAttributes.LLM_IS_STREAMING, False) or \
                         attributes.get('llm.is_streaming', False)
            initial_result = None if is_streaming else "Waiting for response..."
            
            # Apply masking to description if configured
            if client.masking_function:
                description = client.mask(description)
            
            # Create event - session.create_event will handle temporary step creation if needed
            event_kwargs = {
                'description': description,
                'result': initial_result,
                'model': model
            }

            if DEBUG:
                logger.info(f"[SpanProcessor -- DEBUG] event_kwargs: {event_kwargs}")
            
            if images:
                event_kwargs['screenshots'] = images
                
            # Check for step context
            step_id = attributes.get('lucidic.step_id')
            if step_id:
                event_kwargs['step_id'] = step_id
                
            return client.session.create_event(**event_kwargs)
            
        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            return None
    
    def _create_event_from_span_end(self, span: Span, client: Client) -> Optional[str]:
        """Create and complete event when span ends with all attributes available"""
        try:
            attributes = dict(span.attributes or {})
            
            if DEBUG:
                logger.info(f"[SpanProcessor] Creating event from span end with {len(attributes)} attributes")
            
            # Extract all information
            if VERBOSE:
                logger.info(f"[SpanProcessor -- DEBUG] Extracting Description attributes: {attributes}")
            description = self._extract_description(span, attributes)

            if VERBOSE:
                logger.info(f"[SpanProcessor -- DEBUG] Extracting Result attributes: {attributes}")
            raw_result = self._extract_result(span, attributes)

            if VERBOSE:
                logger.info(f"[SpanProcessor -- DEBUG] Extracting Images: span: attributes: {attributes}")
            images = self._extract_images(attributes)

            if VERBOSE:
                logger.info(f"[SpanProcessor -- DEBUG] Extracting Model: span: {span} attributes: {attributes}")
            model = (
                attributes.get(SpanAttributes.LLM_RESPONSE_MODEL) or
                attributes.get(SpanAttributes.LLM_REQUEST_MODEL) or
                attributes.get('gen_ai.response.model') or
                attributes.get('gen_ai.request.model') or
                'unknown'
            )

            # Format result as Input/Output
            # The description contains the input (prompts), raw_result contains the output (completions)
            formatted_result = f"{raw_result}"

            if VERBOSE:
                logger.info(f"[SpanProcessor -- DEBUG] description: {description}, result: {formatted_result}, model: {model}, images: {images}")
            
            # Apply masking
            if client.masking_function:
                formatted_result = client.mask(formatted_result)
            
            # Calculate cost
            cost = self._calculate_cost(attributes)
            
            # Calculate duration in seconds
            duration_seconds = None
            if span.start_time and span.end_time:
                duration_ns = span.end_time - span.start_time
                duration_seconds = duration_ns / 1_000_000_000
            
            # Check success
            is_successful = span.status.status_code != StatusCode.ERROR
            
            # Resolve target session id for this span
            target_session_id = attributes.get('lucidic.session_id')
            if not target_session_id:
                try:
                    target_session_id = current_session_id.get(None)
                except Exception:
                    target_session_id = None
            if not target_session_id:
                # Fallback to global client session if set
                if getattr(client, 'session', None) and getattr(client.session, 'session_id', None):
                    target_session_id = client.session.session_id
            if not target_session_id:
                if DEBUG:
                    logger.info("[SpanProcessor] No session id found for span; skipping event creation")
                return None

            # Create event with all data
            event_kwargs = {
                'description': description,
                'result': formatted_result,
                'model': model,
                'is_finished': True,
                'duration': duration_seconds
            }
            
            if images:
                event_kwargs['screenshots'] = images
                
            if cost is not None:
                event_kwargs['cost_added'] = cost
                
            # Check for step context
            step_id = attributes.get('lucidic.step_id')
            if step_id:
                event_kwargs['step_id'] = step_id
                
            # Create the event (already completed) for the resolved session id
            event_id = client.create_event_for_session(target_session_id, **event_kwargs)
                
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to create event from span end: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            return None
    
    def _update_event_from_span_end(self, span: Span, event_id: str, client: Client) -> None:
        """Update event when span ends"""
        try:
            attributes = dict(span.attributes or {})
            
            # Extract response
            result = self._extract_result(span, attributes)
            
            # Apply masking to result if configured
            if client.masking_function:
                result = client.mask(result)
            
            # Calculate cost
            cost = self._calculate_cost(attributes)
            
            # Calculate duration in seconds
            duration_seconds = None
            if span.start_time and span.end_time:
                duration_ns = span.end_time - span.start_time
                duration_seconds = duration_ns / 1_000_000_000
            
            # Check success
            is_successful = span.status.status_code != StatusCode.ERROR
            
            # Update event
            update_kwargs = {
                'event_id': event_id,
                'result': result,
                'is_finished': True,
                'duration': duration_seconds
            }
            
            if cost is not None:
                update_kwargs['cost_added'] = cost
                
            # Update model if we got a response model
            response_model = attributes.get(SpanAttributes.LLM_RESPONSE_MODEL) or \
                           attributes.get('gen_ai.response.model')
            if response_model:
                update_kwargs['model'] = response_model

            if DEBUG:
                logger.info(f"[SpanProcessor -- DEBUG] update_kwargs: {update_kwargs}")
                
            client.session.update_event(**update_kwargs)
            
        except Exception as e:
            logger.error(f"Failed to update event: {e}")
    
    def _extract_description(self, span: Span, attributes: Dict[str, Any]) -> str:
        """Extract description from span"""
        if VERBOSE:
            logger.info(f"[SpanProcessor] Extracting description from attributes: {list(attributes.keys())}")
        
        # Try to reconstruct messages from indexed attributes (OpenLLMetry format)
        messages = self._extract_indexed_messages(attributes)
        if messages:
            if VERBOSE:
                logger.info(f"[SpanProcessor] Reconstructed {len(messages)} messages from indexed attributes")
            return self._format_messages(messages)
        
        # Try prompts first (other formats)
        prompts = attributes.get(SpanAttributes.LLM_PROMPTS) or \
                 attributes.get('gen_ai.prompt') or \
                 attributes.get('llm.prompts')
                 
        if prompts:
            if DEBUG:
                logger.info(f"[SpanProcessor] Found prompts: {prompts}")
            return self._format_prompts(prompts)
            
        # Try messages
        messages = attributes.get('gen_ai.messages') or \
                  attributes.get('llm.messages')
                  
        if messages:
            if DEBUG:
                logger.info(f"[SpanProcessor] Found messages: {messages}")
            return self._format_messages(messages)


        # check for openai agents tool call
        tool_name = attributes.get('gen_ai.tool.name')
        if tool_name:
            if DEBUG:
                logger.info(f"[SpanProcessor] Found openai agents tool call: {tool_name}")
            
            # Extract and format tool parameters
            tool_params_str = attributes.get('gen_ai.tool.parameters')
            if tool_params_str:
                try:
                    # Parse the JSON string
                    tool_params = json.loads(tool_params_str)
                    # Format the parameters nicely
                    formatted_params = json.dumps(tool_params, indent=2)
                    return f"Agent Tool Call: {tool_name}\nParameters:{formatted_params}"
                except json.JSONDecodeError:
                    # If parsing fails, just include the raw string
                    return f"Agent Tool Call: {tool_name}\nParameters: {tool_params_str}"
            else:
                return f"Agent Tool Call: {tool_name}"
            
        # Fallback
        if DEBUG:
            # logger.info(f"[SpanProcessor] span attributes: {attributes}")
            logger.warning(f"[SpanProcessor] No prompts/messages found, using fallback")
        return f"LLM Request: {span.name}"
    
    def _extract_indexed_messages(self, attributes: Dict[str, Any]) -> list:
        """Extract messages from indexed attributes (gen_ai.prompt.0.role, gen_ai.prompt.0.content, etc.)"""
        messages = []
        i = 0
        
        # Keep extracting messages until we don't find any more
        while True:
            prefix = f"gen_ai.prompt.{i}"
            role = attributes.get(f"{prefix}.role")
            content = attributes.get(f"{prefix}.content")

            # Check if any attributes exist for this index
            attr_has_any = False
            for key in attributes.keys():
                if isinstance(key, str) and key.startswith(f"{prefix}."):
                    attr_has_any = True
                    break

            stored_text = get_stored_text(i)
            stored_images = get_stored_images()

            # Break if no indexed attrs and not the first synthetic message case
            if not attr_has_any and not (i == 0 and (stored_text or stored_images)):
                break

            message = {"role": role or "user"}

            if content:
                # Try to parse JSON content (for multimodal)
                try:
                    import json
                    parsed_content = json.loads(content)
                    message["content"] = parsed_content
                except Exception:
                    message["content"] = content
            else:
                # Content missing: synthesize from stored text/images
                synthetic_content = []
                if stored_text and i == 0:
                    synthetic_content.append({"type": "text", "text": stored_text})
                if stored_images and i == 0:
                    for img in stored_images:
                        synthetic_content.append({"type": "image_url", "image_url": {"url": img}})
                if synthetic_content:
                    if DEBUG:
                        logger.info(f"[SpanProcessor] Using stored text/images for message {i}")
                    message["content"] = synthetic_content
                elif not attr_has_any:
                    # No real attributes and nothing stored to synthesize -> stop
                    break

            messages.append(message)
            i += 1
            
        return messages
    
    def _extract_indexed_completions(self, attributes: Dict[str, Any]) -> list:
        """Extract completions from indexed attributes"""
        completions = []
        i = 0
        
        while True:
            prefix = f"gen_ai.completion.{i}"
            role = attributes.get(f"{prefix}.role")
            content = attributes.get(f"{prefix}.content")
            
            if not role and not content:
                break
                
            completion = {}
            if role:
                completion["role"] = role
            if content:
                completion["content"] = content
                
            if completion:
                completions.append(completion)
            
            i += 1
            
        return completions
    
    def _extract_result(self, span: Span, attributes: Dict[str, Any]) -> str:
        """Extract result from span"""
        if VERBOSE:
            logger.info(f"[SpanProcessor -- _extract_result -- DEBUG] Extracting result from attributes: {attributes}")

        # Try indexed completions first (OpenLLMetry format)
        completions = self._extract_indexed_completions(attributes)
        if completions:
            if VERBOSE:
                logger.info(f"[SpanProcessor] Found {len(completions)} indexed completions")
            # Format completions
            results = []
            for comp in completions:
                if "content" in comp:
                    results.append(str(comp["content"]))
            if results:
                return "\n".join(results)
        
        # Try completions
        completions = attributes.get(SpanAttributes.LLM_COMPLETIONS) or \
                     attributes.get('gen_ai.completion') or \
                     attributes.get('llm.completions')
                     
        if completions:
            if isinstance(completions, list):
                return "\n".join(str(c) for c in completions)
            else:
                return str(completions)
                
        # Check for error
        if span.status.status_code == StatusCode.ERROR:
            return f"Error: {span.status.description or 'Unknown error'}"
            
        # Check streaming
        if attributes.get(SpanAttributes.LLM_IS_STREAMING):
            content = attributes.get('llm.response.content') or \
                     attributes.get('gen_ai.response.content')
            if content:
                return content

        if attributes.get('gen_ai.system') and attributes.get('gen_ai.system') == 'openai_agents':
            if DEBUG:
                logger.info(f"[SpanProcessor -- Agent Tool Call Response Received]") # span attributes: {attributes}")

            # Check the operation type to determine what kind of response this is
            operation_name = attributes.get('gen_ai.operation.name')
            tool_result = attributes.get('gen_ai.tool.result')
            agent_name = attributes.get('gen_ai.agent.name')
            
            # For function/tool spans, just show the tool result
            if operation_name == 'function' and tool_result:
                return f"Tool Result: {tool_result}"
            
            # For agent spans or response spans without tool results, this might be a handoff
            # We can check if there's actual completion content
            completion_content = None
            for i in range(10):  # Check up to 10 completions
                content = attributes.get(f'gen_ai.completion.{i}.content')
                if content:
                    completion_content = content
                    break
            
            # If we have completion content, return it (this is the actual agent response)
            if completion_content:
                return completion_content
            
            # Otherwise, this is likely a handoff scenario
            # Since we can't determine the next agent, just indicate a handoff occurred
            return "Agent Handoff"

        return "Response received"
    
    def _extract_images(self, attributes: Dict[str, Any]) -> list:
        """Extract images from multimodal prompts"""
        images = []

        if VERBOSE:
            logger.info(f"[SpanProcessor -- _extract_images -- DEBUG] Extracting images from attributes: {attributes}")
        
        # First check indexed messages (OpenLLMetry format)
        messages = self._extract_indexed_messages(attributes)
        for msg in messages:
            if isinstance(msg, dict):
                images.extend(self._extract_images_from_message(msg))
        
        # Check for multimodal content in prompts
        prompts = attributes.get(SpanAttributes.LLM_PROMPTS) or \
                 attributes.get('gen_ai.prompt')
                 
        if isinstance(prompts, list):
            for prompt in prompts:
                if isinstance(prompt, dict):
                    images.extend(self._extract_images_from_message(prompt))
                    
        # Check messages too
        messages = attributes.get('gen_ai.messages') or \
                  attributes.get('llm.messages')
                  
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    images.extend(self._extract_images_from_message(msg))
        
        # If no images found but we have stored images in thread-local, retrieve them
        stored_images = get_stored_images()
        if not images and stored_images:
            if DEBUG:
                logger.info(f"[SpanProcessor] No images found in attributes, checking thread-local storage: {len(stored_images)} images")
            for img in stored_images:
                if img and not img.startswith('data:'):
                    images.append(f"data:image/jpeg;base64,{img}")
                else:
                    images.append(img)
        
        if DEBUG and images:
            logger.info(f"[SpanProcessor] Extracted {len(images)} images")
                    
        return images
    
    def _extract_images_from_message(self, message: dict) -> list:
        """Extract images from a single message"""
        images = []
        content = message.get('content', '')

        if VERBOSE:
            logger.info(f"[SpanProcessor -- _extract_images_from_message -- DEBUG] Extracting images from message: {message}, content: {content}")

        # Handle case where content might be a JSON string
        if isinstance(content, str) and content.strip().startswith('['):
            try:
                parsed_content = json.loads(content)
                if isinstance(parsed_content, list):
                    content = parsed_content
            except json.JSONDecodeError:
                # If parsing fails, keep content as string
                pass

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'image_url':
                    image_url = item.get('image_url', {})
                    if isinstance(image_url, dict):
                        url = image_url.get('url', '')
                        if url.startswith('data:image'):
                            images.append(url)
                        elif url.startswith('lucidic_image_'):
                            # This is a placeholder - retrieve from thread-local storage
                            image = self._retrieve_image_from_placeholder(url)
                            if image:
                                images.append(image)
                            
        return images
    
    def _retrieve_image_from_placeholder(self, placeholder: str) -> Optional[str]:
        """Retrieve image from thread-local storage using placeholder"""
        try:
            base64_data = get_image_by_placeholder(placeholder)
            if base64_data:
                # Ensure it has proper data URI format
                if not base64_data.startswith('data:'):
                    # Add data URI prefix if missing
                    base64_data = f"data:image/jpeg;base64,{base64_data}"
                return base64_data
        except Exception as e:
            if DEBUG:
                logger.error(f"[SpanProcessor] Failed to retrieve image from placeholder: {e}")
        return None
    
    def _format_prompts(self, prompts: Any) -> str:
        """Format prompts into description"""
        if isinstance(prompts, str):
            return prompts
        elif isinstance(prompts, list):
            return self._format_messages(prompts)
        else:
            return "Model request"
    
    def _format_messages(self, messages: list) -> str:
        """Format message list"""
        formatted = []
        
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                
                if isinstance(content, str):
                    formatted.append(f"{role}: {content}")
                elif isinstance(content, list):
                    # Extract text from multimodal
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            texts.append(item.get('text', ''))
                    if texts:
                        formatted.append(f"{role}: {' '.join(texts)}")
            elif isinstance(msg, str):
                formatted.append(msg)
                
        return '\n'.join(formatted) if formatted else "Model request"
    
    def _calculate_cost(self, attributes: Dict[str, Any]) -> Optional[float]:
        """Calculate cost from token usage"""
        prompt_tokens = (
            attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) or
            attributes.get('gen_ai.usage.prompt_tokens') or
            attributes.get('gen_ai.usage.input_tokens') or
            0
        )
        
        completion_tokens = (
            attributes.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS) or
            attributes.get('gen_ai.usage.completion_tokens') or
            attributes.get('gen_ai.usage.output_tokens') or
            0
        )
        
        total_tokens = prompt_tokens + completion_tokens
        
        if total_tokens > 0:
            model = (
                attributes.get(SpanAttributes.LLM_RESPONSE_MODEL) or
                attributes.get(SpanAttributes.LLM_REQUEST_MODEL) or
                attributes.get('gen_ai.response.model') or
                attributes.get('gen_ai.request.model')
            )
            
            if model:
                return calculate_cost(model, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens})
                
        return None
    
    def shutdown(self, timeout_millis: int = 30000) -> None:
        """Shutdown processor"""
        if self.span_to_event:
            logger.warning(f"Shutting down with {len(self.span_to_event)} incomplete spans")
            
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush - no-op for this processor"""
        return True