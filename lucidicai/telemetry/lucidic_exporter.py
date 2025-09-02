"""Custom OpenTelemetry exporter for Lucidic (Exporter-only mode).

Converts completed spans into immutable typed LLM events via Client.create_event(),
which enqueues non-blocking delivery through the EventQueue.
"""
import json
import logging
from typing import Sequence, Optional, Dict, Any, List
from datetime import datetime, timezone
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode
from opentelemetry.semconv_ai import SpanAttributes

from lucidicai.client import Client
from lucidicai.context import current_session_id, current_parent_event_id
from lucidicai.model_pricing import calculate_cost
from .extract import detect_is_llm_span, extract_images, extract_prompts, extract_completions, extract_model

logger = logging.getLogger("Lucidic")
import os

DEBUG = os.getenv("LUCIDIC_DEBUG", "False") == "True"
VERBOSE = os.getenv("LUCIDIC_VERBOSE", "False") == "True"


class LucidicSpanExporter(SpanExporter):
    """Exporter that creates immutable LLM events for completed spans."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            client = Client()
            if DEBUG and spans:
                logger.debug(f"[LucidicSpanExporter] Processing {len(spans)} spans")
            for span in spans:
                self._process_span(span, client)
            if DEBUG and spans:
                logger.debug(f"[LucidicSpanExporter] Successfully exported {len(spans)} spans")
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Failed to export spans: {e}")
            return SpanExportResult.FAILURE

    def _process_span(self, span: ReadableSpan, client: Client) -> None:
        """Convert a single LLM span into a typed, immutable event."""
        try:
            if not detect_is_llm_span(span):
                return

            attributes = dict(span.attributes or {})

            # Resolve session id
            target_session_id = attributes.get('lucidic.session_id')
            if not target_session_id:
                try:
                    target_session_id = current_session_id.get(None)
                except Exception:
                    target_session_id = None
            if not target_session_id and getattr(client, 'session', None) and getattr(client.session, 'session_id', None):
                target_session_id = client.session.session_id
            if not target_session_id:
                return

            # Parent nesting - get from span attributes (captured at span creation)
            parent_id = attributes.get('lucidic.parent_event_id')
            if not parent_id:
                # Fallback to trying context (may work if same thread)
                try:
                    parent_id = current_parent_event_id.get(None)
                except Exception:
                    parent_id = None

            # Timing
            occurred_at = datetime.fromtimestamp(span.start_time / 1_000_000_000, tz=timezone.utc) if span.start_time else datetime.now(tz=timezone.utc)
            duration_seconds = ((span.end_time - span.start_time) / 1_000_000_000) if (span.start_time and span.end_time) else None

            # Typed fields using extract utilities
            model = extract_model(attributes) or 'unknown'
            provider = self._detect_provider_name(attributes)
            messages = extract_prompts(attributes) or []
            params = self._extract_params(attributes)
            output_text = extract_completions(span, attributes) or "Response received"
            input_tokens = self._extract_prompt_tokens(attributes)
            output_tokens = self._extract_completion_tokens(attributes)
            cost = self._calculate_cost(attributes)
            images = extract_images(attributes)

            # Create immutable event via non-blocking queue
            event_id = client.create_event(
                type="llm_generation",
                session_id=target_session_id,
                parent_event_id=parent_id,
                occurred_at=occurred_at,
                duration=duration_seconds,
                provider=provider,
                model=model,
                messages=messages,
                params=params,
                output=output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                raw={"images": images} if images else None,
            )
            
            if DEBUG:
                logger.debug(f"[LucidicSpanExporter] Created LLM event {event_id} for session {target_session_id[:8]}...")

        except Exception as e:
            logger.error(f"Failed to process span {span.name}: {e}")
    
    
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
            
            # Resolve target session id for this span
            target_session_id = attributes.get('lucidic.session_id')
            if not target_session_id:
                try:
                    target_session_id = current_session_id.get(None)
                except Exception:
                    target_session_id = None
            if not target_session_id:
                if getattr(client, 'session', None) and getattr(client.session, 'session_id', None):
                    target_session_id = client.session.session_id
            if not target_session_id:
                return None

            # Create event
            event_kwargs = {
                'description': description,
                'result': "Processing...",  # Will be updated when span ends
                'model': model
            }
            
            if images:
                event_kwargs['screenshots'] = images
                
            return client.create_event_for_session(target_session_id, **event_kwargs)
            
        except Exception as e:
            logger.error(f"Failed to create event from span: {e}")
            return None
    
    def _update_event_from_span(self, span: ReadableSpan, attributes: Dict[str, Any], event_id: str, client: Client) -> None:
        """Deprecated: events are immutable; no updates performed."""
        return
    
    def _extract_description(self, span: ReadableSpan, attributes: Dict[str, Any]) -> str:
        """Extract description from span attributes"""
        # Try to get prompts/messages
        prompts = attributes.get(SpanAttributes.LLM_PROMPTS) or \
                 attributes.get('gen_ai.prompt')
        
        if VERBOSE:
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
    
    def _detect_provider_name(self, attributes: Dict[str, Any]) -> str:
        name = attributes.get('gen_ai.system') or attributes.get('service.name')
        if name:
            return str(name)
        return "openai" if 'openai' in (str(attributes.get('service.name', '')).lower()) else "unknown"
    

    def _extract_params(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "temperature": attributes.get('gen_ai.request.temperature'),
            "max_tokens": attributes.get('gen_ai.request.max_tokens'),
            "top_p": attributes.get('gen_ai.request.top_p'),
        }

    def _extract_prompt_tokens(self, attributes: Dict[str, Any]) -> int:
        return (
            attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) or
            attributes.get('gen_ai.usage.prompt_tokens') or
            attributes.get('gen_ai.usage.input_tokens') or 0
        )

    def _extract_completion_tokens(self, attributes: Dict[str, Any]) -> int:
        return (
            attributes.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS) or
            attributes.get('gen_ai.usage.completion_tokens') or
            attributes.get('gen_ai.usage.output_tokens') or 0
        )
    
    def _calculate_cost(self, attributes: Dict[str, Any]) -> Optional[float]:
        prompt_tokens = (
            attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) or
            attributes.get('gen_ai.usage.prompt_tokens') or
            attributes.get('gen_ai.usage.input_tokens') or 0
        )
        completion_tokens = (
            attributes.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS) or
            attributes.get('gen_ai.usage.completion_tokens') or
            attributes.get('gen_ai.usage.output_tokens') or 0
        )
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        if total_tokens > 0:
            model = (
                attributes.get(SpanAttributes.LLM_RESPONSE_MODEL) or
                attributes.get(SpanAttributes.LLM_REQUEST_MODEL) or
                attributes.get('gen_ai.response.model') or
                attributes.get('gen_ai.request.model')
            )
            if model:
                usage = {"prompt_tokens": prompt_tokens or 0, "completion_tokens": completion_tokens or 0, "total_tokens": total_tokens}
                return calculate_cost(model, usage)
        return None
    
    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True