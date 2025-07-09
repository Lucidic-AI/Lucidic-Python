"""Custom OpenTelemetry exporter for Lucidic backend compatibility"""
import json
import logging
from typing import Sequence, Optional, Dict, Any, List
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode
from opentelemetry.semconv_ai import SpanAttributes

from lucidicai.client import Client
from lucidicai.model_pricing import calculate_cost
from lucidicai.image_upload import extract_base64_images

logger = logging.getLogger("Lucidic")
import os

DEBUG = os.getenv("LUCIDIC_DEBUG", "False") == "True"



class LucidicSpanExporter(SpanExporter):
    """Custom exporter that converts OpenTelemetry spans to Lucidic events"""
    
    def __init__(self):
        self.pending_events = {}  # Track events by span_id
        
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans by converting them to Lucidic events"""
        try:
            client = Client()
            if not client.session:
                logger.debug("No active session, skipping span export")
                return SpanExportResult.SUCCESS
                
            for span in spans:
                self._process_span(span, client)
                
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Failed to export spans: {e}")
            return SpanExportResult.FAILURE
    
    def _process_span(self, span: ReadableSpan, client: Client) -> None:
        """Process a single span and convert to Lucidic event"""
        try:
            # Skip non-LLM spans
            if not self._is_llm_span(span):
                return
                
            # Extract relevant attributes
            attributes = dict(span.attributes or {})
            
            # Create or update event based on span lifecycle
            span_id = format(span.context.span_id, '016x')
            
            if span_id not in self.pending_events:
                # New span - create event
                event_id = self._create_event_from_span(span, attributes, client)
                if event_id:
                    self.pending_events[span_id] = {
                        'event_id': event_id,
                        'start_time': span.start_time
                    }
            else:
                # Span ended - update event
                event_info = self.pending_events.pop(span_id)
                self._update_event_from_span(span, attributes, event_info['event_id'], client)
                
        except Exception as e:
            logger.error(f"Failed to process span {span.name}: {e}")
    
    def _is_llm_span(self, span: ReadableSpan) -> bool:
        """Check if this is an LLM-related span"""
        # Check span name patterns
        llm_patterns = ['openai', 'anthropic', 'chat', 'completion', 'embedding', 'llm']
        span_name_lower = span.name.lower()
        
        if any(pattern in span_name_lower for pattern in llm_patterns):
            return True
            
        # Check for LLM attributes
        if span.attributes:
            for key in span.attributes:
                if key.startswith('gen_ai.') or key.startswith('llm.'):
                    return True
                    
        return False
    
    def _create_event_from_span(self, span: ReadableSpan, attributes: Dict[str, Any], client: Client) -> Optional[str]:
        """Create a Lucidic event from span start"""
        try:
            # Extract description from prompts/messages
            description = self._extract_description(span, attributes)
            
            # Extract images if present
            images = self._extract_images(attributes)
            
            # Get model info
            model = attributes.get(SpanAttributes.LLM_RESPONSE_MODEL) or \
                   attributes.get(SpanAttributes.LLM_REQUEST_MODEL) or \
                   attributes.get('gen_ai.request.model') or 'unknown'
            
            # Create event
            event_kwargs = {
                'description': description,
                'result': "Processing...",  # Will be updated when span ends
                'model': model
            }
            
            if images:
                event_kwargs['screenshots'] = images
                
            # Check if we have a specific step_id in span attributes
            step_id = attributes.get('lucidic.step_id')
            if step_id:
                event_kwargs['step_id'] = step_id
                
            return client.session.create_event(**event_kwargs)
            
        except Exception as e:
            logger.error(f"Failed to create event from span: {e}")
            return None
    
    def _update_event_from_span(self, span: ReadableSpan, attributes: Dict[str, Any], event_id: str, client: Client) -> None:
        """Update a Lucidic event from span end"""
        try:
            # Extract response/result
            result = self._extract_result(span, attributes)
            
            # Calculate cost if we have token usage
            cost = self._calculate_cost(attributes)
            
            # Determine success
            is_successful = span.status.status_code != StatusCode.ERROR
            
            update_kwargs = {
                'event_id': event_id,
                'result': result,
                'is_finished': True,
                'is_successful': is_successful
            }
            
            if cost is not None:
                update_kwargs['cost_added'] = cost
                
            client.session.update_event(**update_kwargs)
            
        except Exception as e:
            logger.error(f"Failed to update event from span: {e}")
    
    def _extract_description(self, span: ReadableSpan, attributes: Dict[str, Any]) -> str:
        """Extract description from span attributes"""
        # Try to get prompts/messages
        prompts = attributes.get(SpanAttributes.LLM_PROMPTS) or \
                 attributes.get('gen_ai.prompt')
        
        if DEBUG:
            logger.info(f"[SpaneExporter -- DEBUG] Extracting Description attributes: {attributes}, prompts: {prompts}")

        if prompts:
            if isinstance(prompts, list) and prompts:
                # Handle message list format
                return self._format_messages(prompts)
            elif isinstance(prompts, str):
                return prompts
                
        # Fallback to span name
        return f"LLM Call: {span.name}"
    
    def _extract_result(self, span: ReadableSpan, attributes: Dict[str, Any]) -> str:
        """Extract result/response from span attributes"""
        # Try to get completions
        completions = attributes.get(SpanAttributes.LLM_COMPLETIONS) or \
                     attributes.get('gen_ai.completion')
        
        if completions:
            if isinstance(completions, list) and completions:
                # Handle multiple completions
                return "\n".join(str(c) for c in completions)
            elif isinstance(completions, str):
                return completions
                
        # Check for error
        if span.status.status_code == StatusCode.ERROR:
            return f"Error: {span.status.description or 'Unknown error'}"
            
        return "Response received"
    
    def _extract_images(self, attributes: Dict[str, Any]) -> List[str]:
        """Extract base64 images from attributes"""
        images = []
        
        # Check prompts for multimodal content
        prompts = attributes.get(SpanAttributes.LLM_PROMPTS) or \
                 attributes.get('gen_ai.prompt')
                 
        if isinstance(prompts, list):
            for prompt in prompts:
                if isinstance(prompt, dict) and 'content' in prompt:
                    content = prompt['content']
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'image_url':
                                image_url = item.get('image_url', {})
                                if isinstance(image_url, dict) and 'url' in image_url:
                                    url = image_url['url']
                                    if url.startswith('data:image'):
                                        images.append(url)
                                        
        return images
    
    def _format_messages(self, messages: List[Any]) -> str:
        """Format message list into description"""
        formatted = []
        
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                
                if isinstance(content, str):
                    formatted.append(f"{role}: {content}")
                elif isinstance(content, list):
                    # Handle multimodal content
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                    if text_parts:
                        formatted.append(f"{role}: {' '.join(text_parts)}")
                        
        return '\n'.join(formatted) if formatted else "Model request"
    
    def _calculate_cost(self, attributes: Dict[str, Any]) -> Optional[float]:
        """Calculate cost from token usage"""
        prompt_tokens = attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) or \
                       attributes.get('gen_ai.usage.prompt_tokens') or 0
        completion_tokens = attributes.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS) or \
                           attributes.get('gen_ai.usage.completion_tokens') or 0
        
        if prompt_tokens or completion_tokens:
            model = attributes.get(SpanAttributes.LLM_RESPONSE_MODEL) or \
                   attributes.get(SpanAttributes.LLM_REQUEST_MODEL) or \
                   attributes.get('gen_ai.request.model')
                   
            if model:
                return calculate_cost(prompt_tokens, completion_tokens, model)
                
        return None
    
    def shutdown(self) -> None:
        """Shutdown the exporter"""
        # Process any remaining pending events
        if self.pending_events:
            logger.warning(f"Shutting down with {len(self.pending_events)} pending events")
            
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending spans"""
        return True