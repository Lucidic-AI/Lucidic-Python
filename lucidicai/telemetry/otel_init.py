"""OpenTelemetry initialization and configuration for Lucidic"""
import logging
from typing import List, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor

from .lucidic_span_processor import LucidicSpanProcessor
from .otel_provider import OpenTelemetryProvider
from lucidicai.client import Client

logger = logging.getLogger("Lucidic")


class LucidicTelemetry:
    """Manages OpenTelemetry initialization for Lucidic"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.tracer_provider = None
            self.span_processor = None
            self.instrumentors = {}
            self.provider = OpenTelemetryProvider()
            self._initialized = True
    
    def initialize(self, agent_id: str, service_name: str = "lucidic-ai") -> None:
        """Initialize OpenTelemetry with Lucidic configuration"""
        if self.tracer_provider:
            logger.debug("OpenTelemetry already initialized")
            return
            
        try:
            # Create resource
            resource = Resource.create({
                "service.name": service_name,
                "service.version": "1.0.0",
                "lucidic.agent_id": agent_id,
            })
            
            # Create tracer provider
            self.tracer_provider = TracerProvider(resource=resource)
            
            # Add our custom span processor for real-time event handling
            self.span_processor = LucidicSpanProcessor()
            self.tracer_provider.add_span_processor(self.span_processor)
            
            # Set as global provider
            trace.set_tracer_provider(self.tracer_provider)
            
            logger.info("[LucidicTelemetry] OpenTelemetry initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            raise
    
    def instrument_providers(self, providers: List[str]) -> None:
        """Instrument specified providers"""
        for provider in providers:
            try:
                if provider == "openai" and provider not in self.instrumentors:
                    self._instrument_openai()
                elif provider == "anthropic" and provider not in self.instrumentors:
                    self._instrument_anthropic()
                elif provider == "langchain" and provider not in self.instrumentors:
                    self._instrument_langchain()
                elif provider == "pydantic_ai":
                    # Custom instrumentation needed
                    logger.info(f"[LucidicTelemetry] Pydantic AI will use manual instrumentation")
                elif provider == "openai_agents":
                    # OpenAI Agents uses the same OpenAI instrumentation
                    self._instrument_openai_agents()
            except Exception as e:
                logger.error(f"Failed to instrument {provider}: {e}")
    
    def _instrument_openai(self) -> None:
        """Instrument OpenAI"""
        try:
            # Get client for masking function
            client = Client()
            
            # Configure instrumentation
            instrumentor = OpenAIInstrumentor()
            
            # Create a custom callback for getting attributes
            def get_custom_attributes():
                attrs = {}
                
                # Add step context if available
                if client.session and client.session.active_step:
                    attrs["lucidic.step_id"] = client.session.active_step.step_id
                    
                return attrs
            
            instrumentor.instrument(
                tracer_provider=self.tracer_provider,
                enrich_token_usage=True,
                exception_logger=lambda e: logger.error(f"OpenAI error: {e}"),
                get_common_metrics_attributes=get_custom_attributes,
                enable_trace_context_propagation=True,
                use_legacy_attributes=True  # Force legacy attributes mode for now
            )
            
            self.instrumentors["openai"] = instrumentor
            logger.info("[LucidicTelemetry] Instrumented OpenAI")
            
        except Exception as e:
            logger.error(f"Failed to instrument OpenAI: {e}")
            raise
    
    def _instrument_anthropic(self) -> None:
        """Instrument Anthropic"""
        try:
            instrumentor = AnthropicInstrumentor()
            
            # Get client for context
            client = Client()
            
            def get_custom_attributes():
                attrs = {}
                if client.session and client.session.active_step:
                    attrs["lucidic.step_id"] = client.session.active_step.step_id
                return attrs
            
            instrumentor.instrument(
                tracer_provider=self.tracer_provider,
                exception_logger=lambda e: logger.error(f"Anthropic error: {e}"),
                get_common_metrics_attributes=get_custom_attributes,
                use_legacy_attributes=True  # Force legacy attributes mode
            )
            
            self.instrumentors["anthropic"] = instrumentor
            logger.info("[LucidicTelemetry] Instrumented Anthropic")
            
        except Exception as e:
            logger.error(f"Failed to instrument Anthropic: {e}")
            raise
    
    def _instrument_langchain(self) -> None:
        """Instrument LangChain"""
        try:
            instrumentor = LangchainInstrumentor()
            instrumentor.instrument(tracer_provider=self.tracer_provider)
            
            self.instrumentors["langchain"] = instrumentor
            logger.info("[LucidicTelemetry] Instrumented LangChain")
            
        except Exception as e:
            logger.error(f"Failed to instrument LangChain: {e}")
            raise

    def _instrument_openai_agents(self) -> None:
        """Instrument OpenAI Agents SDK"""
        try:
            from .openai_agents_instrumentor import OpenAIAgentsInstrumentor
            
            instrumentor = OpenAIAgentsInstrumentor(tracer_provider=self.tracer_provider)
            instrumentor.instrument()
            
            self.instrumentors["openai_agents"] = instrumentor
            logger.info("[LucidicTelemetry] Instrumented OpenAI Agents SDK")
            
        except Exception as e:
            logger.error(f"Failed to instrument OpenAI Agents SDK: {e}")
            raise
    
    def uninstrument_all(self) -> None:
        """Uninstrument all providers"""
        for name, instrumentor in self.instrumentors.items():
            try:
                instrumentor.uninstrument()
                logger.info(f"[LucidicTelemetry] Uninstrumented {name}")
            except Exception as e:
                logger.error(f"Failed to uninstrument {name}: {e}")
                
        self.instrumentors.clear()
        
        # Shutdown tracer provider
        if self.tracer_provider:
            self.tracer_provider.shutdown()
            self.tracer_provider = None
            self.span_processor = None
            
    def is_initialized(self) -> bool:
        """Check if telemetry is initialized"""
        return self.tracer_provider is not None