"""Streaming response wrapper for Lucidic AI SDK"""
import logging
from typing import Any, Dict, Iterator, Optional, AsyncIterator
import time
import json

from lucidicai.client import Client

logger = logging.getLogger("Lucidic")


class StreamingResponseWrapper:
    """Wrapper for streaming responses that tracks chunks and updates events"""
    
    def __init__(self, response: Any, session: Any, kwargs: Dict[str, Any]):
        self.response = response
        self.kwargs = kwargs
        self.chunks = []
        self.start_time = time.time()
        self.event_id = None
        self.accumulated_content = ""
        self.usage = None
        
        # We no longer create an initial event; emit a single immutable event on finalize
    
    def _create_initial_event(self):
        return
    
    def _format_messages(self, messages):
        """Format messages for description and extract images"""
        description = ""
        images = []
        
        if isinstance(messages, list):
            content_parts = []
            for message in messages:
                if isinstance(message, dict):
                    role = message.get('role', 'unknown')
                    content = message.get('content', '')
                    
                    if isinstance(content, str):
                        content_parts.append(f"{role}: {content}")
                    elif isinstance(content, list):
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
            
            description = '\n'.join(content_parts) if content_parts else "Streaming request"
        
        return description, images
    
    def _serialize_messages(self, messages):
        """Serialize messages for raw_request"""
        if not isinstance(messages, list):
            return messages
            
        serialized = []
        for message in messages:
            if isinstance(message, dict):
                msg_copy = message.copy()
                content = msg_copy.get('content', '')
                
                # Handle multimodal content
                if isinstance(content, list):
                    new_content = []
                    for item in content:
                        if isinstance(item, dict):
                            item_copy = item.copy()
                            if item.get('type') == 'image_url':
                                # Truncate image data for logging
                                image_url = item.get('image_url', {})
                                if isinstance(image_url, dict) and 'url' in image_url:
                                    url = image_url['url']
                                    if url.startswith('data:'):
                                        item_copy['image_url'] = {'url': 'data:image/...base64...'}
                            new_content.append(item_copy)
                    msg_copy['content'] = new_content
                
                serialized.append(msg_copy)
        
        return serialized
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over streaming chunks"""
        try:
            logger.info(f"[Streaming] Starting iteration for event {self.event_id}")
            chunk_count = 0
            
            for chunk in self.response:
                chunk_count += 1
                self.chunks.append(chunk)
                
                # Extract content from chunk
                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        self.accumulated_content += delta.content
                        logger.debug(f"[Streaming] Chunk {chunk_count}: '{delta.content}' ({len(delta.content)} chars)")
                
                # Extract usage if available
                if hasattr(chunk, 'usage') and chunk.usage:
                    self.usage = chunk.usage
                    logger.debug(f"[Streaming] Got usage data in chunk {chunk_count}")
                
                yield chunk
            
            logger.info(f"[Streaming] Iteration complete, received {chunk_count} chunks")
                
        except Exception as e:
            logger.error(f"[Streaming] Error during iteration: {str(e)}")
            # Still finalize the event even on error
            self._finalize_event()
            raise
        finally:
            # Update event when streaming completes
            self._finalize_event()
    
    async def __aiter__(self) -> AsyncIterator[Any]:
        """Async iterate over streaming chunks"""
        try:
            logger.info(f"[Streaming] Starting async iteration for event {self.event_id}")
            chunk_count = 0
            
            async for chunk in self.response:
                chunk_count += 1
                self.chunks.append(chunk)
                
                # Extract content from chunk
                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        self.accumulated_content += delta.content
                        logger.debug(f"[Streaming] Async chunk {chunk_count}: '{delta.content}' ({len(delta.content)} chars)")
                
                # Extract usage if available
                if hasattr(chunk, 'usage') and chunk.usage:
                    self.usage = chunk.usage
                    logger.debug(f"[Streaming] Got usage data in async chunk {chunk_count}")
                
                yield chunk
            
            logger.info(f"[Streaming] Async iteration complete, received {chunk_count} chunks")
            
        except Exception as e:
            logger.error(f"[Streaming] Error during async iteration: {str(e)}")
            raise
        finally:
            # Always finalize the event
            logger.info(f"[Streaming] Finalizing async streaming event {self.event_id}")
            self._finalize_event()
    
    def _finalize_event(self):
        """Finalize the event with accumulated data"""
        try:
            logger.info(f"[Streaming] Finalizing event {self.event_id}, accumulated content length: {len(self.accumulated_content)}")
            
            if not self.session:
                # Try to get session from client
                try:
                    from lucidicai.client import Client
                    self.session = Client().session
                except:
                    logger.warning("[Streaming] No session available to finalize event")
                    return
                
            duration = time.time() - self.start_time
            
            # Calculate tokens and cost
            total_tokens = 0
            prompt_tokens = 0
            completion_tokens = 0
            
            if self.usage:
                total_tokens = getattr(self.usage, 'total_tokens', 0)
                prompt_tokens = getattr(self.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(self.usage, 'completion_tokens', 0)
            else:
                # Estimate tokens from content
                completion_tokens = len(self.accumulated_content.split()) * 1.3  # Rough estimate
                prompt_tokens = len(str(self.kwargs.get('messages', '')).split()) * 1.3
                total_tokens = prompt_tokens + completion_tokens
            
            # Calculate cost
            model = self.kwargs.get('model', 'unknown')
            cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
            
            # Create single immutable event at end
            result_text = self.accumulated_content if self.accumulated_content else "Stream completed (no content received)"
            Client().create_event(
                type="llm_generation",
                model=model,
                messages=self.kwargs.get('messages', []),
                output=result_text,
                input_tokens=int(prompt_tokens),
                output_tokens=int(completion_tokens),
                cost=cost,
                duration=duration,
            )
            logger.info(f"[Streaming] Emitted immutable streaming event")
            
        except Exception as e:
            logger.error(f"[Streaming] Error finalizing event {self.event_id}: {str(e)}")
    
    def _calculate_cost(self, model: str, prompt_tokens: float, completion_tokens: float) -> float:
        """Calculate cost based on model and tokens"""
        try:
            from lucidicai.model_pricing import calculate_cost
            usage_dict = {
                'prompt_tokens': int(prompt_tokens),
                'completion_tokens': int(completion_tokens),
                'total_tokens': int(prompt_tokens + completion_tokens)
            }
            return calculate_cost(model, usage_dict)
        except Exception as e:
            logger.debug(f"[Streaming] Cost calculation failed: {str(e)}")
            return 0.0