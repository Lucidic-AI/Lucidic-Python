"""Unified telemetry initialization - SpanExporter-only architecture.

Sets up a TracerProvider with BatchSpanProcessor + LucidicSpanExporter and
initializes OpenTelemetry instrumentations based on a providers list.
"""
import logging
from typing import List, Tuple

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

from .lucidic_exporter import LucidicSpanExporter
from .session_stamp_processor import SessionStampProcessor

logger = logging.getLogger("Lucidic")


def initialize_telemetry(providers: List[str], agent_id: str) -> Tuple[TracerProvider, List]:
    """Single function to set up all telemetry with exporter-only path.

    Returns (provider, instrumentors) for optional shutdown/cleanup tracking.
    """
    resource = Resource.create({
        "service.name": "lucidic-ai",
        "service.version": "1.0.0",
        "lucidic.agent_id": agent_id,
    })

    provider = TracerProvider(resource=resource)
    
    # Add session stamp processor to stamp spans with session ID
    stamp_processor = SessionStampProcessor()
    provider.add_span_processor(stamp_processor)
    
    # Add exporter processor for sending spans to Lucidic
    exporter = LucidicSpanExporter()
    export_processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(export_processor)

    try:
        trace.set_tracer_provider(provider)
    except Exception:
        # If already set globally, we still proceed with our local provider
        pass

    instrumentors = []

    # Normalize provider names
    canonical = set()
    for p in providers or []:
        if p in ("google_generativeai",):
            canonical.add("google")
        elif p in ("vertex_ai",):
            canonical.add("vertexai")
        elif p in ("aws_bedrock", "amazon_bedrock"):
            canonical.add("bedrock")
        else:
            canonical.add(p)

    # OpenAI
    if "openai" in canonical:
        try:
            from opentelemetry.instrumentation.openai import OpenAIInstrumentor
            inst = OpenAIInstrumentor()
            inst.instrument(tracer_provider=provider, enrich_token_usage=True)
            instrumentors.append(("openai", inst))
            logger.info("[Telemetry] Instrumented OpenAI")
        except Exception as e:
            logger.error(f"Failed to instrument OpenAI: {e}")

    # Anthropic
    if "anthropic" in canonical:
        try:
            from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
            inst = AnthropicInstrumentor()
            inst.instrument(tracer_provider=provider)
            instrumentors.append(("anthropic", inst))
            logger.info("[Telemetry] Instrumented Anthropic")
        except Exception as e:
            logger.error(f"Failed to instrument Anthropic: {e}")

    # LangChain
    if "langchain" in canonical:
        try:
            from opentelemetry.instrumentation.langchain import LangchainInstrumentor
            inst = LangchainInstrumentor()
            inst.instrument(tracer_provider=provider)
            instrumentors.append(("langchain", inst))
            logger.info("[Telemetry] Instrumented LangChain")
        except Exception as e:
            logger.error(f"Failed to instrument LangChain: {e}")

    # Google Generative AI
    if "google" in canonical:
        try:
            from opentelemetry.instrumentation.google_generativeai import GoogleGenerativeAiInstrumentor
            inst = GoogleGenerativeAiInstrumentor()
            inst.instrument(tracer_provider=provider)
            instrumentors.append(("google", inst))
            logger.info("[Telemetry] Instrumented Google Generative AI")
        except Exception as e:
            logger.error(f"Failed to instrument Google Generative AI: {e}")

    # Vertex AI
    if "vertexai" in canonical:
        try:
            from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor
            inst = VertexAIInstrumentor()
            inst.instrument(tracer_provider=provider)
            instrumentors.append(("vertexai", inst))
            logger.info("[Telemetry] Instrumented Vertex AI")
        except Exception as e:
            logger.error(f"Failed to instrument Vertex AI: {e}")

    # Bedrock
    if "bedrock" in canonical:
        try:
            from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
            inst = BedrockInstrumentor(enrich_token_usage=True)
            inst.instrument(tracer_provider=provider)
            instrumentors.append(("bedrock", inst))
            logger.info("[Telemetry] Instrumented Bedrock")
        except Exception as e:
            logger.error(f"Failed to instrument Bedrock: {e}")

    # Cohere
    if "cohere" in canonical:
        try:
            from opentelemetry.instrumentation.cohere import CohereInstrumentor
            inst = CohereInstrumentor()
            inst.instrument(tracer_provider=provider)
            instrumentors.append(("cohere", inst))
            logger.info("[Telemetry] Instrumented Cohere")
        except Exception as e:
            logger.error(f"Failed to instrument Cohere: {e}")

    # Groq
    if "groq" in canonical:
        try:
            from opentelemetry.instrumentation.groq import GroqInstrumentor
            inst = GroqInstrumentor()
            inst.instrument(tracer_provider=provider)
            instrumentors.append(("groq", inst))
            logger.info("[Telemetry] Instrumented Groq")
        except Exception as e:
            logger.error(f"Failed to instrument Groq: {e}")

    # LiteLLM - callback-based (not OpenTelemetry)
    if "litellm" in canonical:
        logger.info("[Telemetry] LiteLLM uses callback-based instrumentation")
        # LiteLLM requires setup via litellm_bridge.py
        try:
            from .litellm_bridge import setup_litellm_callback
            setup_litellm_callback()
            instrumentors.append(("litellm", None))  # No instrumentor object
        except Exception as e:
            logger.error(f"Failed to setup LiteLLM: {e}")

    # Pydantic AI - manual spans
    if "pydantic_ai" in canonical:
        logger.info("[Telemetry] Pydantic AI requires manual span creation")
        # No automatic instrumentation available
        instrumentors.append(("pydantic_ai", None))

    # OpenAI Agents - custom instrumentor
    if "openai_agents" in canonical:
        try:
            from .openai_agents_instrumentor import OpenAIAgentsInstrumentor
            inst = OpenAIAgentsInstrumentor(tracer_provider=provider)
            inst.instrument()
            instrumentors.append(("openai_agents", inst))
            logger.info("[Telemetry] Instrumented OpenAI Agents SDK")
        except Exception as e:
            logger.error(f"Failed to instrument OpenAI Agents: {e}")

    return provider, instrumentors


