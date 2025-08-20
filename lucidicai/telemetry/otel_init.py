"""OpenTelemetry initialization and configuration for Lucidic

Adds thread-safety and idempotence to avoid duplicate tracer provider
registration and repeated instrumentation under concurrency.
"""
import logging
from typing import List, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
# Instrumentors are imported lazily inside methods to avoid import errors

from .lucidic_span_processor import LucidicSpanProcessor
from .otel_provider import OpenTelemetryProvider
from lucidicai.client import Client

logger = logging.getLogger("Lucidic")

import threading


_init_lock = threading.Lock()

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
        with _init_lock:
            if self.tracer_provider:
                logger.debug("OpenTelemetry already initialized")
                return
            try:
                resource = Resource.create({
                    "service.name": service_name,
                    "service.version": "1.0.0",
                    "lucidic.agent_id": agent_id,
                })
                provider = TracerProvider(resource=resource)
                processor = LucidicSpanProcessor()
                provider.add_span_processor(processor)
                try:
                    trace.set_tracer_provider(provider)
                except Exception as e:
                    # Another provider may already be registered; proceed with ours as a local provider
                    logger.debug(f"Global tracer provider already set: {e}")
                self.tracer_provider = provider
                self.span_processor = processor
                logger.info("[LucidicTelemetry] OpenTelemetry initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenTelemetry: {e}")
                raise
    
    def instrument_providers(self, providers: List[str]) -> None:
        """Instrument specified providers"""
        with _init_lock:
            for provider in providers:
                # Map synonyms to canonical names
                canonical = provider
                if provider in ("google_generativeai",):
                    canonical = "google"
                elif provider in ("vertex_ai",):
                    canonical = "vertexai"
                elif provider in ("aws_bedrock", "amazon_bedrock"):
                    canonical = "bedrock"
                try:
                    if canonical == "openai" and canonical not in self.instrumentors:
                        self._instrument_openai()
                    elif canonical == "anthropic" and canonical not in self.instrumentors:
                        self._instrument_anthropic()
                    elif canonical == "langchain" and canonical not in self.instrumentors:
                        self._instrument_langchain()
                    elif canonical == "google" and canonical not in self.instrumentors:
                        self._instrument_google_generativeai()
                    elif canonical == "vertexai" and canonical not in self.instrumentors:
                        self._instrument_vertexai()
                    elif canonical == "bedrock" and canonical not in self.instrumentors:
                        self._instrument_bedrock()
                    elif canonical == "cohere" and canonical not in self.instrumentors:
                        self._instrument_cohere()
                    elif canonical == "groq" and canonical not in self.instrumentors:
                        self._instrument_groq()
                    elif canonical == "pydantic_ai":
                        logger.info(f"[LucidicTelemetry] Pydantic AI will use manual instrumentation")
                    elif canonical == "openai_agents":
                        self._instrument_openai_agents()
                    elif canonical == "litellm":
                        logger.info(f"[LucidicTelemetry] LiteLLM will use callback-based instrumentation")
                except Exception as e:
                    logger.error(f"Failed to instrument {canonical}: {e}")
    
    def _instrument_openai(self) -> None:
        """Instrument OpenAI"""
        try:
            from opentelemetry.instrumentation.openai import OpenAIInstrumentor
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
            from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
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
            from opentelemetry.instrumentation.langchain import LangchainInstrumentor
            instrumentor = LangchainInstrumentor()
            instrumentor.instrument(tracer_provider=self.tracer_provider)
            
            self.instrumentors["langchain"] = instrumentor
            logger.info("[LucidicTelemetry] Instrumented LangChain")
            
        except Exception as e:
            logger.error(f"Failed to instrument LangChain: {e}")
            raise

    def _instrument_google_generativeai(self) -> None:
        """Instrument Google Generative AI"""
        try:
            from opentelemetry.instrumentation.google_generativeai import GoogleGenerativeAiInstrumentor
            instrumentor = GoogleGenerativeAiInstrumentor(exception_logger=lambda e: logger.error(f"Google Generative AI error: {e}"))
            instrumentor.instrument(tracer_provider=self.tracer_provider)
            self.instrumentors["google"] = instrumentor
            logger.info("[LucidicTelemetry] Instrumented Google Generative AI")
        except Exception as e:
            logger.error(f"Failed to instrument Google Generative AI: {e}")
            raise

    def _instrument_vertexai(self) -> None:
        """Instrument Vertex AI"""
        try:
            from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor
            instrumentor = VertexAIInstrumentor(exception_logger=lambda e: logger.error(f"Vertex AI error: {e}"))
            instrumentor.instrument(tracer_provider=self.tracer_provider)
            self.instrumentors["vertexai"] = instrumentor
            logger.info("[LucidicTelemetry] Instrumented Vertex AI")
        except Exception as e:
            logger.error(f"Failed to instrument Vertex AI: {e}")
            raise

    def _instrument_cohere(self) -> None:
        """Instrument Cohere"""
        try:
            from opentelemetry.instrumentation.cohere import CohereInstrumentor
            instrumentor = CohereInstrumentor(exception_logger=lambda e: logger.error(f"Cohere error: {e}"), use_legacy_attributes=True)
            instrumentor.instrument(tracer_provider=self.tracer_provider)
            self.instrumentors["cohere"] = instrumentor
            logger.info("[LucidicTelemetry] Instrumented Cohere")
        except Exception as e:
            logger.error(f"Failed to instrument Cohere: {e}")
            raise

    def _instrument_bedrock(self) -> None:
        """Instrument AWS Bedrock"""
        try:
            from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
            instrumentor = BedrockInstrumentor(enrich_token_usage=True, exception_logger=lambda e: logger.error(f"Bedrock error: {e}"))
            instrumentor.instrument(tracer_provider=self.tracer_provider)
            self.instrumentors["bedrock"] = instrumentor
            logger.info("[LucidicTelemetry] Instrumented Bedrock")
        except Exception as e:
            logger.error(f"Failed to instrument Bedrock: {e}")
            raise

    def _instrument_groq(self) -> None:
        """Instrument Groq"""
        try:
            from lucidicai.client import Client
            client = Client()
            def get_custom_attributes():
                attrs = {}
                if client.session and client.session.active_step:
                    attrs["lucidic.step_id"] = client.session.active_step.step_id
                return attrs
            from opentelemetry.instrumentation.groq import GroqInstrumentor
            instrumentor = GroqInstrumentor(exception_logger=lambda e: logger.error(f"Groq error: {e}"), use_legacy_attributes=True, get_common_metrics_attributes=get_custom_attributes)
            instrumentor.instrument(tracer_provider=self.tracer_provider)
            self.instrumentors["groq"] = instrumentor
            logger.info("[LucidicTelemetry] Instrumented Groq")
        except Exception as e:
            logger.error(f"Failed to instrument Groq: {e}")
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

    def force_flush(self) -> None:
        """Best-effort force flush of telemetry before shutdown.

        Uses whichever force_flush hooks are available on the provider or span processor.
        Swallows all exceptions to avoid interfering with process shutdown paths.
        """
        try:
            provider = getattr(self, 'tracer_provider', None)
            if provider and hasattr(provider, 'force_flush'):
                try:
                    provider.force_flush()
                except Exception:
                    pass
            processor = getattr(self, 'span_processor', None)
            if processor and hasattr(processor, 'force_flush'):
                try:
                    processor.force_flush()
                except Exception:
                    pass
        except Exception:
            # Never raise from force_flush
            pass