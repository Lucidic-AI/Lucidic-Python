"""Bridge between LiteLLM's CustomLogger and Lucidic's telemetry system"""
import logging
import os
import time
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from litellm import CustomLogger
except ImportError:
    # Create a dummy CustomLogger if litellm is not installed
    class CustomLogger:
        def __init__(self, **kwargs):
            pass

from lucidicai.client import Client
from lucidicai.model_pricing import calculate_cost

logger = logging.getLogger("Lucidic")
DEBUG = os.getenv("LUCIDIC_DEBUG", "False") == "True"


class LucidicLiteLLMCallback(CustomLogger):
    """
    Custom callback for LiteLLM that bridges to Lucidic's event system.
    
    This callback integrates LiteLLM's logging with Lucidic's session/step/event hierarchy,
    enabling automatic tracking of all LiteLLM-supported providers.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._active_events = {}  # Track active events for streaming
        self._pending_callbacks = set()  # Track pending callback executions
        self._callback_lock = threading.Lock()  # Thread-safe callback tracking
        
    def _register_callback(self, callback_id: str):
        """Register a callback as pending"""
        with self._callback_lock:
            self._pending_callbacks.add(callback_id)
            if DEBUG:
                logger.info(f"LiteLLM Bridge: Registered callback {callback_id}, pending: {len(self._pending_callbacks)}")
    
    def _complete_callback(self, callback_id: str):
        """Mark a callback as completed"""
        with self._callback_lock:
            self._pending_callbacks.discard(callback_id)
            if DEBUG:
                logger.info(f"LiteLLM Bridge: Completed callback {callback_id}, pending: {len(self._pending_callbacks)}")
    
    def wait_for_pending_callbacks(self, timeout: float = 5.0):
        """Wait for all pending callbacks to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._callback_lock:
                if not self._pending_callbacks:
                    if DEBUG:
                        logger.info("LiteLLM Bridge: All callbacks completed")
                    return True
            
            time.sleep(0.1)  # Check every 100ms
        
        # Timeout reached
        with self._callback_lock:
            pending_count = len(self._pending_callbacks)
            if pending_count > 0:
                logger.warning(f"LiteLLM Bridge: Timeout waiting for {pending_count} callbacks")
                # Clear pending callbacks to avoid memory leak
                self._pending_callbacks.clear()
        
        return False
        
    def log_pre_api_call(self, model, messages, kwargs):
        """Called before the LLM API call"""
        try:
            client = Client()
            if not client.session:
                return
                
            # Extract description from messages
            description = self._format_messages(messages)
            
            # Apply masking if configured
            if hasattr(client, 'mask') and callable(client.mask):
                description = client.mask(description)
                
            # Store pre-call info for later use
            call_id = kwargs.get("litellm_call_id", str(time.time())) if kwargs else str(time.time())
            self._active_events[call_id] = {
                "model": model,
                "messages": messages,
                "description": description,
                "start_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"LiteLLM Bridge error in pre_api_call: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
    
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Called on successful LLM completion"""
        # Generate unique callback ID
        callback_id = f"success_{id(kwargs)}_{start_time}"
        self._register_callback(callback_id)
        
        try:
            client = Client()
            if not client.session:
                self._complete_callback(callback_id)
                return
                
            # Get call info
            call_id = kwargs.get("litellm_call_id", str(start_time))
            pre_call_info = self._active_events.pop(call_id, {})
            
            # Extract model and provider info
            model = kwargs.get("model", pre_call_info.get("model", "unknown"))
            provider = self._extract_provider(model)
            
            # Get messages for description
            messages = kwargs.get("messages", pre_call_info.get("messages", []))
            description = pre_call_info.get("description") or self._format_messages(messages)
            
            # Extract response content
            result = self._extract_response_content(response_obj)
            
            # Apply masking to result if configured
            if hasattr(client, 'mask') and callable(client.mask):
                result = client.mask(result)
            
            # Calculate cost if usage info is available
            usage = self._extract_usage(response_obj)
            cost = None
            if usage:
                cost = self._calculate_litellm_cost(model, usage)
            
            # Extract any images from multimodal requests
            images = self._extract_images_from_messages(messages)
            
            # Create event
            event_kwargs = {
                "description": description,
                "result": result,
                "model": f"{provider}/{model}" if "/" not in model else model,
                "is_finished": True
            }
            
            if cost is not None:
                event_kwargs["cost_added"] = cost
                
            if images:
                event_kwargs["screenshots"] = images
                
            # Add metadata
            metadata = {
                "provider": provider,
                "duration_ms": (end_time - start_time) * 1000,
                "litellm": True
            }
            if usage:
                metadata["usage"] = usage
                
            event_kwargs["metadata"] = metadata
            
            # Create the event
            client.session.create_event(**event_kwargs)
            
            if DEBUG:
                logger.info(f"LiteLLM Bridge: Created event for {model} completion")
                
        except Exception as e:
            logger.error(f"LiteLLM Bridge error in log_success_event: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
        finally:
            self._complete_callback(callback_id)
    
    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Called on failed LLM completion"""
        # Generate unique callback ID
        callback_id = f"failure_{id(kwargs)}_{start_time}"
        self._register_callback(callback_id)
        
        try:
            client = Client()
            if not client.session:
                self._complete_callback(callback_id)
                return
                
            # Get call info
            call_id = kwargs.get("litellm_call_id", str(start_time))
            pre_call_info = self._active_events.pop(call_id, {})
            
            # Extract model info
            model = kwargs.get("model", pre_call_info.get("model", "unknown"))
            provider = self._extract_provider(model)
            
            # Get messages for description
            messages = kwargs.get("messages", pre_call_info.get("messages", []))
            description = pre_call_info.get("description") or self._format_messages(messages)
            
            # Format error
            error_msg = str(response_obj) if response_obj else "Unknown error"
            
            # Create failed event
            event_kwargs = {
                "description": description,
                "result": f"Error: {error_msg}",
                "model": f"{provider}/{model}" if "/" not in model else model,
                "is_finished": True,
                "metadata": {
                    "provider": provider,
                    "duration_ms": (end_time - start_time) * 1000,
                    "litellm": True,
                    "error": True
                }
            }
            
            client.session.create_event(**event_kwargs)
            
            if DEBUG:
                logger.info(f"LiteLLM Bridge: Created error event for {model}")
                
        except Exception as e:
            logger.error(f"LiteLLM Bridge error in log_failure_event: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
        finally:
            self._complete_callback(callback_id)
    
    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        """Called for streaming responses"""
        # For now, we'll handle the complete response in log_success_event
        # This could be enhanced to show real-time streaming updates
        pass
    
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Async version of log_success_event"""
        # Delegate to sync version - Lucidic client handles both sync/async internally
        self.log_success_event(kwargs, response_obj, start_time, end_time)
    
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Async version of log_failure_event"""
        # Delegate to sync version
        self.log_failure_event(kwargs, response_obj, start_time, end_time)
    
    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        """Async version of log_stream_event"""
        self.log_stream_event(kwargs, response_obj, start_time, end_time)
    
    def _extract_provider(self, model: str) -> str:
        """Extract provider from model string"""
        if "/" in model:
            return model.split("/")[0]
        
        # Try to infer provider from model name patterns
        model_lower = model.lower()
        if "gpt" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "vertex_ai"
        elif "llama" in model_lower:
            return "meta"
        elif "mistral" in model_lower:
            return "mistral"
        
        return "unknown"
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages into a description string"""
        if not messages:
            return "LiteLLM Request"
            
        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                if isinstance(content, str):
                    formatted.append(f"{role}: {content}")
                elif isinstance(content, list):
                    # Handle multimodal content
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            texts.append(item.get("text", ""))
                    if texts:
                        formatted.append(f"{role}: {' '.join(texts)}")
                        
        return "\n".join(formatted) if formatted else "LiteLLM Request"
    
    def _extract_response_content(self, response_obj) -> str:
        """Extract response content from LiteLLM response object"""
        try:
            # Handle different response types
            if hasattr(response_obj, "choices") and response_obj.choices:
                # Standard completion response
                choice = response_obj.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    return choice.message.content or "No content"
                elif hasattr(choice, "text"):
                    return choice.text or "No content"
            
            # Fallback to string representation
            return str(response_obj)
            
        except Exception as e:
            logger.error(f"Error extracting response content: {e}")
            return "Response received"
    
    def _extract_usage(self, response_obj) -> Optional[Dict[str, int]]:
        """Extract usage information from response"""
        try:
            if hasattr(response_obj, "usage"):
                usage = response_obj.usage
                if hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
                    return {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens if hasattr(usage, "total_tokens") else (usage.prompt_tokens + usage.completion_tokens)
                    }
        except Exception as e:
            logger.debug(f"Could not extract usage: {e}")
        
        return None
    
    def _calculate_litellm_cost(self, model: str, usage: Dict[str, int]) -> Optional[float]:
        """Calculate cost using Lucidic's pricing model"""
        try:
            # LiteLLM model names might need normalization for pricing lookup
            normalized_model = model
            if "/" in model:
                # Extract the model name after the provider prefix
                # e.g., "openai/gpt-4o" -> "gpt-4o"
                parts = model.split("/", 1)
                if len(parts) == 2:
                    normalized_model = parts[1]
            
            return calculate_cost(normalized_model, usage)
        except Exception as e:
            logger.debug(f"Could not calculate cost for {model}: {e}")
            return None
    
    def _extract_images_from_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract base64 images from multimodal messages"""
        images = []
        
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                                if url.startswith("data:image"):
                                    images.append(url)
                                    
        return images