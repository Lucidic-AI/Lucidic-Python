"""OpenTelemetry-based provider implementation"""
import logging
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from opentelemetry import trace, context
from opentelemetry.trace import Tracer, Span
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
# Instrumentors are imported lazily inside methods to avoid import errors
from opentelemetry.semconv_ai import SpanAttributes

from .lucidic_exporter import LucidicSpanExporter
from .lucidic_span_processor import LucidicSpanProcessor
from .base_provider import BaseProvider
from lucidicai.client import Client

logger = logging.getLogger("Lucidic")


class OpenTelemetryProvider(BaseProvider):
    """Provider that uses OpenTelemetry instrumentations instead of monkey-patching"""
    
    def __init__(self):
        super().__init__()
        self._provider_name = "OpenTelemetry"
        self.tracer_provider = None
        self.tracer = None
        self.instrumentors = {}
        self._active_spans = {}
        
    def initialize_telemetry(self, service_name: str = "lucidic-ai", agent_id: str = None) -> None:
        """Initialize OpenTelemetry with Lucidic exporter"""
        # Create resource with service info
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "lucidic.agent_id": agent_id or ""
        })
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        
        # Add our custom exporter
        lucidic_exporter = LucidicSpanExporter()
        span_processor = BatchSpanProcessor(lucidic_exporter)
        self.tracer_provider.add_span_processor(span_processor)
        # Also add session-stamping processor to ensure correct attribution
        try:
            self.tracer_provider.add_span_processor(LucidicSpanProcessor())
        except Exception:
            pass
        
        # Set as global provider (ignore if already set)
        try:
            trace.set_tracer_provider(self.tracer_provider)
        except Exception:
            pass
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__)
        
    def handle_response(self, response, kwargs, session: Optional = None):
        """Handle responses - not needed with OTEL approach"""
        return response
        
    def override(self):
        """Initialize OpenTelemetry instrumentations"""
        try:
            client = Client()
            
            # Initialize telemetry if not already done
            if not self.tracer_provider:
                self.initialize_telemetry(agent_id=client.agent_id)
            
            # No actual override needed - instrumentations will be enabled separately
            logger.info("[OpenTelemetry Provider] Initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            raise
            
    def undo_override(self):
        """Uninstrument all providers"""
        for name, instrumentor in self.instrumentors.items():
            try:
                instrumentor.uninstrument()
                logger.info(f"[OpenTelemetry Provider] Uninstrumented {name}")
            except Exception as e:
                logger.error(f"Failed to uninstrument {name}: {e}")
                
        self.instrumentors.clear()
        
        # Shutdown tracer provider
        if self.tracer_provider:
            self.tracer_provider.shutdown()
            
    def instrument_openai(self) -> None:
        """Instrument OpenAI with OpenLLMetry"""
        if "openai" not in self.instrumentors:
            try:
                from opentelemetry.instrumentation.openai import OpenAIInstrumentor
                instrumentor = OpenAIInstrumentor()
                instrumentor.instrument(
                    tracer_provider=self.tracer_provider,
                    enrich_token_usage=True,
                    exception_logger=lambda e: logger.error(f"OpenAI error: {e}")
                )
                self.instrumentors["openai"] = instrumentor
                logger.info("[OpenTelemetry Provider] Instrumented OpenAI")
            except Exception as e:
                logger.error(f"Failed to instrument OpenAI: {e}")
                
    def instrument_anthropic(self) -> None:
        """Instrument Anthropic with OpenLLMetry"""
        if "anthropic" not in self.instrumentors:
            try:
                from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
                instrumentor = AnthropicInstrumentor()
                instrumentor.instrument(
                    tracer_provider=self.tracer_provider,
                    exception_logger=lambda e: logger.error(f"Anthropic error: {e}")
                )
                self.instrumentors["anthropic"] = instrumentor
                logger.info("[OpenTelemetry Provider] Instrumented Anthropic")
            except Exception as e:
                logger.error(f"Failed to instrument Anthropic: {e}")
                
    def instrument_langchain(self) -> None:
        """Instrument LangChain with OpenLLMetry"""
        if "langchain" not in self.instrumentors:
            try:
                from opentelemetry.instrumentation.langchain import LangchainInstrumentor
                instrumentor = LangchainInstrumentor()
                instrumentor.instrument(tracer_provider=self.tracer_provider)
                self.instrumentors["langchain"] = instrumentor
                logger.info("[OpenTelemetry Provider] Instrumented LangChain")
            except Exception as e:
                logger.error(f"Failed to instrument LangChain: {e}")
                
    def instrument_pydantic_ai(self) -> None:
        """Instrument Pydantic AI"""
        # Note: OpenLLMetry doesn't have a Pydantic AI instrumentation yet
        # We'll need to create custom instrumentation or use manual spans
        logger.info("[OpenTelemetry Provider] Pydantic AI instrumentation not yet available in OpenLLMetry")
        
    @contextmanager
    def trace_step(self, step_id: str, state: str = None, action: str = None, goal: str = None):
        """Context manager to associate spans with a specific step"""
        span = self.tracer.start_span(
            name=f"step.{step_id}",
            attributes={
                "lucidic.step_id": step_id,
                "lucidic.step.state": state or "",
                "lucidic.step.action": action or "",
                "lucidic.step.goal": goal or ""
            }
        )
        
        token = context.attach(trace.set_span_in_context(span))
        try:
            yield span
        finally:
            context.detach(token)
            span.end()
            
    def add_image_to_span(self, image_data: str, image_type: str = "screenshot") -> None:
        """Add image data to current span"""
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute(f"lucidic.image.{image_type}", image_data)
            
    def set_step_context(self, step_id: str) -> None:
        """Set step ID in current span context"""
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute("lucidic.step_id", step_id)