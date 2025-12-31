"""OpenAI Agents SDK instrumentor that hooks into OpenAI API calls"""
import logging
from typing import Any, Dict, Optional, List
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanKind
import threading
import json

logger = logging.getLogger("Lucidic")

# Thread-local storage for context
_thread_local = threading.local()


class OpenAIAgentsInstrumentor:
    """instrumentor that captures OpenAI API calls within agent runs"""
    
    def __init__(self, tracer_provider=None):
        self._tracer_provider = tracer_provider or trace.get_tracer_provider()
        self._tracer = self._tracer_provider.get_tracer(__name__)
        self._is_instrumented = False
        self._original_openai_create = None
        
    def instrument(self):
        """Enable instrumentation"""
        if self._is_instrumented:
            logger.warning("OpenAI Agents SDK already instrumented")
            return
            
        try:
            # First, patch OpenAI to capture API calls
            self._patch_openai()
            
            # Then set up agents tracing
            from agents import set_trace_processors
            from agents.tracing.processors import TracingProcessor
            
            processor = OpenAIAgentsTracingProcessor(self)
            set_trace_processors([processor])
            
            self._is_instrumented = True
            logger.info("OpenAI Agents SDK instrumentation enabled")
            
        except Exception as e:
            logger.error(f"Failed to instrument OpenAI Agents SDK: {e}")
            raise
    
    def uninstrument(self):
        """Disable instrumentation"""
        if not self._is_instrumented:
            return
            
        try:
            # Restore OpenAI
            self._unpatch_openai()
            
            # Restore default processor
            from agents import set_trace_processors
            from agents.tracing.processors import default_processor
            set_trace_processors([default_processor])
            
            self._is_instrumented = False
            logger.info("OpenAI Agents SDK instrumentation disabled")
            
        except Exception as e:
            logger.error(f"Failed to uninstrument: {e}")
    
    def _patch_openai(self):
        """Patch OpenAI client to capture messages"""
        try:
            import openai
            
            # Store original
            self._original_openai_create = openai.chat.completions.create
            
            def wrapped_create(*args, **kwargs):
                # Capture the messages
                messages = kwargs.get('messages', [])
                
                # Store in thread local
                if not hasattr(_thread_local, 'current_messages'):
                    _thread_local.current_messages = []
                _thread_local.current_messages = messages
                
                # Call original
                response = self._original_openai_create(*args, **kwargs)
                
                # Store response
                _thread_local.current_response = response
                
                return response
            
            # Replace
            openai.chat.completions.create = wrapped_create
            logger.debug("Patched OpenAI chat.completions.create")
            
        except Exception as e:
            logger.error(f"Failed to patch OpenAI: {e}")
    
    def _unpatch_openai(self):
        """Restore OpenAI client"""
        if self._original_openai_create:
            try:
                import openai
                openai.chat.completions.create = self._original_openai_create
                logger.debug("Restored OpenAI chat.completions.create")
            except Exception as e:
                logger.debug(f"[OpenAIAgents] Failed to restore OpenAI client: {e}")


class OpenAIAgentsTracingProcessor:
    """processor that captures richer data"""
    
    def __init__(self, instrumentor: OpenAIAgentsInstrumentor):
        self.instrumentor = instrumentor
        self.tracer = instrumentor._tracer
        self._active_spans = {}
        self._agent_context = {}  # Store agent context
        
    def on_span_start(self, span_data: Any) -> None:
        """Called when a span starts"""
        try:
            span_id = str(id(span_data))
            actual_data = getattr(span_data, 'span_data', span_data)
            data_type = actual_data.__class__.__name__
            
            # Create span name
            if hasattr(actual_data, 'name'):
                span_name = f"openai.agents.{actual_data.name}"
                agent_name = actual_data.name
            else:
                span_name = f"openai.agents.{data_type}"
                agent_name = data_type
            
            # For agent spans, store context
            if data_type == "AgentSpanData":
                self._agent_context[agent_name] = {
                    'instructions': getattr(actual_data, 'instructions', None),
                    'name': agent_name
                }
            
            # Create span
            otel_span = self.tracer.start_span(
                name=span_name,
                kind=SpanKind.INTERNAL,
                attributes={
                    "gen_ai.system": "openai_agents",
                    "gen_ai.operation.name": data_type.lower().replace("spandata", ""),
                }
            )
            
            # Add agent name
            if hasattr(actual_data, 'name'):
                otel_span.set_attribute("gen_ai.agent.name", actual_data.name)
            
            self._active_spans[span_id] = {
                'span': otel_span,
                'type': data_type,
                'data': actual_data
            }
            
        except Exception as e:
            logger.error(f"Error in on_span_start: {e}")
    
    def on_span_end(self, span_data: Any) -> None:
        """Called when a span ends"""
        try:
            span_id = str(id(span_data))
            
            if span_id not in self._active_spans:
                return
            
            span_info = self._active_spans.pop(span_id)
            otel_span = span_info['span']
            data_type = span_info['type']
            actual_data = getattr(span_data, 'span_data', span_data)
            
            # Handle different span types
            if data_type == "ResponseSpanData":
                self._handle_response_span(otel_span, actual_data)
            elif data_type == "FunctionSpanData":
                self._handle_function_span(otel_span, actual_data)
            elif data_type == "AgentSpanData":
                self._handle_agent_span(otel_span, actual_data)
            
            # Set status and end
            otel_span.set_status(Status(StatusCode.OK))
            otel_span.end()
            
        except Exception as e:
            logger.error(f"Error in on_span_end: {e}")
    
    def _handle_response_span(self, otel_span: Any, span_data: Any) -> None:
        """Handle response span - this is where we capture prompts and completions"""
        try:
            # Log what we're working with
            logger.debug(f"Handling response span, span_data type: {type(span_data)}")
            
            # First check span_data.input for user messages
            prompt_index = 0
            
            # Get instructions (system prompt) from response
            if hasattr(span_data, 'response') and span_data.response:
                resp = span_data.response
                if hasattr(resp, 'instructions') and resp.instructions:
                    otel_span.set_attribute(f"gen_ai.prompt.{prompt_index}.role", "system")
                    otel_span.set_attribute(f"gen_ai.prompt.{prompt_index}.content", str(resp.instructions)[:2048])
                    prompt_index += 1
            
            # Get user messages from span_data.input
            if hasattr(span_data, 'input') and span_data.input:
                if isinstance(span_data.input, list):
                    # Input is a list of messages
                    for msg in span_data.input:
                        if isinstance(msg, dict):
                            role = msg.get('role', '')
                            content = msg.get('content', '')
                            otel_span.set_attribute(f"gen_ai.prompt.{prompt_index}.role", role)
                            otel_span.set_attribute(f"gen_ai.prompt.{prompt_index}.content", str(content)[:2048])
                            prompt_index += 1
                elif isinstance(span_data.input, str):
                    # Input is a string
                    otel_span.set_attribute(f"gen_ai.prompt.{prompt_index}.role", "user")
                    otel_span.set_attribute(f"gen_ai.prompt.{prompt_index}.content", str(span_data.input)[:2048])
                    prompt_index += 1
            
            # Get response from output
            if hasattr(span_data, 'response') and span_data.response:
                resp = span_data.response
                
                # Look for the assistant's response in output
                if hasattr(resp, 'output') and resp.output:
                    for item in resp.output:
                        if hasattr(item, 'type') and item.type == 'message':
                            if hasattr(item, 'content'):
                                content = item.content
                                if isinstance(content, list):
                                    # Extract text
                                    texts = []
                                    for c in content:
                                        if hasattr(c, 'text'):
                                            texts.append(c.text)
                                    if texts:
                                        content = " ".join(texts)
                                
                                otel_span.set_attribute("gen_ai.completion.0.role", "assistant")
                                otel_span.set_attribute("gen_ai.completion.0.content", str(content)[:2048])
                                break
            
            # Note: Response extraction from thread local was removed since we already
            # extract the completion from span_data.response.output above
            
            # Set model and usage
            if hasattr(span_data, 'response') and span_data.response:
                resp = span_data.response
                if hasattr(resp, 'model'):
                    otel_span.set_attribute("gen_ai.response.model", resp.model)
                    otel_span.set_attribute("gen_ai.request.model", resp.model)
                
                if hasattr(resp, 'usage') and resp.usage:
                    usage = resp.usage
                    if hasattr(usage, 'input_tokens'):
                        otel_span.set_attribute("gen_ai.usage.prompt_tokens", usage.input_tokens)
                    if hasattr(usage, 'output_tokens'):
                        otel_span.set_attribute("gen_ai.usage.completion_tokens", usage.output_tokens)
                    if hasattr(usage, 'total_tokens'):
                        otel_span.set_attribute("gen_ai.usage.total_tokens", usage.total_tokens)
                        
        except Exception as e:
            logger.error(f"Error handling response span: {e}")
    
    def _handle_function_span(self, otel_span: Any, span_data: Any) -> None:
        """Handle function/tool spans"""
        if hasattr(span_data, 'name'):
            otel_span.set_attribute("gen_ai.tool.name", span_data.name)
        
        if hasattr(span_data, 'input'):
            otel_span.set_attribute("gen_ai.tool.parameters", json.dumps(span_data.input)[:500])
        
        if hasattr(span_data, 'output'):
            otel_span.set_attribute("gen_ai.tool.result", str(span_data.output)[:500])
    
    def _handle_agent_span(self, otel_span: Any, span_data: Any) -> None:
        """Handle agent spans"""
        # Agent spans typically don't have much data at end
        pass
    
    def on_trace_start(self, trace_data: Any) -> None:
        """Called when a trace starts"""
        # Clear thread local
        if hasattr(_thread_local, 'current_messages'):
            del _thread_local.current_messages
        if hasattr(_thread_local, 'current_response'):
            del _thread_local.current_response
    
    def on_trace_end(self, trace_data: Any) -> None:
        """Called when a trace ends"""
        pass
    
    def force_flush(self, timeout_seconds: float = 30.0) -> bool:
        """Force flush"""
        return True
    
    def shutdown(self) -> None:
        """Shutdown"""
        pass