"""Unified telemetry initialization - SpanExporter-only architecture.

Provides functions to instrument OpenTelemetry providers.
Provider creation is now handled by the Client singleton.
"""
import logging
import threading
from typing import Dict, Any, Optional

from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger("Lucidic")

# Global tracking to prevent duplicate instrumentation
_global_instrumentors = {}
_instrumentation_lock = threading.Lock()


def instrument_providers(providers: list, tracer_provider: TracerProvider, existing_instrumentors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Instrument the requested providers with the given TracerProvider.
    Only instruments providers that haven't been instrumented yet.
    Uses global tracking to prevent duplicate instrumentation across threads.
    
    Args:
        providers: List of provider names to instrument
        tracer_provider: The TracerProvider to use for instrumentation
        existing_instrumentors: Dict of already instrumented providers (ignored, kept for compatibility)
        
    Returns:
        Dict of newly instrumented providers (name -> instrumentor)
    """
    global _global_instrumentors
    new_instrumentors = {}

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

    # Use global lock to prevent race conditions
    with _instrumentation_lock:
        # OpenAI
        if "openai" in canonical and "openai" not in _global_instrumentors:
            try:
                from opentelemetry.instrumentation.openai import OpenAIInstrumentor
                inst = OpenAIInstrumentor()
                inst.instrument(tracer_provider=tracer_provider, enrich_token_usage=True)
                _global_instrumentors["openai"] = inst
                new_instrumentors["openai"] = inst

                # Clean up any problematic instrumentation from standard library
                from .openai_uninstrument import clean_openai_instrumentation
                clean_openai_instrumentation()

                # Add patch for responses API methods (not covered by standard instrumentation)
                import os
                if os.getenv('LUCIDIC_DISABLE_RESPONSES_PATCH', 'false').lower() != 'true':
                    from .openai_patch import get_responses_patcher
                    patcher = get_responses_patcher(tracer_provider)
                    patcher.patch()
                    _global_instrumentors["openai_responses_patch"] = patcher
                else:
                    logger.info("[Telemetry] Skipping responses API patch (disabled via LUCIDIC_DISABLE_RESPONSES_PATCH)")

                logger.info("[Telemetry] Instrumented OpenAI (including responses.parse, responses.create, beta.chat.completions.parse)")
            except Exception as e:
                logger.error(f"Failed to instrument OpenAI: {e}")

        # Anthropic
        if "anthropic" in canonical and "anthropic" not in _global_instrumentors:
            try:
                from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
                inst = AnthropicInstrumentor()
                inst.instrument(tracer_provider=tracer_provider)
                _global_instrumentors["anthropic"] = inst
                new_instrumentors["anthropic"] = inst
                logger.info("[Telemetry] Instrumented Anthropic")
            except Exception as e:
                logger.error(f"Failed to instrument Anthropic: {e}")

        # LangChain
        if "langchain" in canonical and "langchain" not in _global_instrumentors:
            try:
                from opentelemetry.instrumentation.langchain import LangchainInstrumentor
                inst = LangchainInstrumentor()
                inst.instrument(tracer_provider=tracer_provider)
                _global_instrumentors["langchain"] = inst
                new_instrumentors["langchain"] = inst
                logger.info("[Telemetry] Instrumented LangChain")
            except Exception as e:
                logger.error(f"Failed to instrument LangChain: {e}")

        # Google Generative AI
        if "google" in canonical and "google" not in _global_instrumentors:
            try:
                from opentelemetry.instrumentation.google_generativeai import GoogleGenerativeAiInstrumentor
                inst = GoogleGenerativeAiInstrumentor()
                inst.instrument(tracer_provider=tracer_provider)
                _global_instrumentors["google"] = inst
                new_instrumentors["google"] = inst
                logger.info("[Telemetry] Instrumented Google Generative AI")
            except Exception as e:
                logger.error(f"Failed to instrument Google Generative AI: {e}")

        # Vertex AI
        if "vertexai" in canonical and "vertexai" not in _global_instrumentors:
            try:
                from opentelemetry.instrumentation.vertexai import VertexAIInstrumentor
                inst = VertexAIInstrumentor()
                inst.instrument(tracer_provider=tracer_provider)
                _global_instrumentors["vertexai"] = inst
                new_instrumentors["vertexai"] = inst
                logger.info("[Telemetry] Instrumented Vertex AI")
            except Exception as e:
                logger.error(f"Failed to instrument Vertex AI: {e}")

        # Bedrock
        if "bedrock" in canonical and "bedrock" not in _global_instrumentors:
            try:
                from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
                inst = BedrockInstrumentor(enrich_token_usage=True)
                inst.instrument(tracer_provider=tracer_provider)
                _global_instrumentors["bedrock"] = inst
                new_instrumentors["bedrock"] = inst
                logger.info("[Telemetry] Instrumented Bedrock")
            except Exception as e:
                logger.error(f"Failed to instrument Bedrock: {e}")

        # Cohere
        if "cohere" in canonical and "cohere" not in _global_instrumentors:
            try:
                from opentelemetry.instrumentation.cohere import CohereInstrumentor
                inst = CohereInstrumentor()
                inst.instrument(tracer_provider=tracer_provider)
                _global_instrumentors["cohere"] = inst
                new_instrumentors["cohere"] = inst
                logger.info("[Telemetry] Instrumented Cohere")
            except Exception as e:
                logger.error(f"Failed to instrument Cohere: {e}")

        # Groq
        if "groq" in canonical and "groq" not in _global_instrumentors:
            try:
                from opentelemetry.instrumentation.groq import GroqInstrumentor
                inst = GroqInstrumentor()
                inst.instrument(tracer_provider=tracer_provider)
                _global_instrumentors["groq"] = inst
                new_instrumentors["groq"] = inst
                logger.info("[Telemetry] Instrumented Groq")
            except Exception as e:
                logger.error(f"Failed to instrument Groq: {e}")

        # LiteLLM - callback-based (not OpenTelemetry)
        if "litellm" in canonical and "litellm" not in _global_instrumentors:
            logger.info("[Telemetry] LiteLLM uses callback-based instrumentation")
            # LiteLLM requires setup via litellm_bridge.py
            try:
                from .litellm_bridge import setup_litellm_callback
                setup_litellm_callback()
                _global_instrumentors["litellm"] = None  # No instrumentor object
                new_instrumentors["litellm"] = None
            except Exception as e:
                logger.error(f"Failed to setup LiteLLM: {e}")

        # Pydantic AI - manual spans
        if "pydantic_ai" in canonical and "pydantic_ai" not in _global_instrumentors:
            logger.info("[Telemetry] Pydantic AI requires manual span creation")
            # No automatic instrumentation available
            _global_instrumentors["pydantic_ai"] = None
            new_instrumentors["pydantic_ai"] = None

        # OpenAI Agents - custom instrumentor
        if "openai_agents" in canonical and "openai_agents" not in _global_instrumentors:
            try:
                from .openai_agents_instrumentor import OpenAIAgentsInstrumentor
                inst = OpenAIAgentsInstrumentor(tracer_provider=tracer_provider)
                inst.instrument()
                _global_instrumentors["openai_agents"] = inst
                new_instrumentors["openai_agents"] = inst
                logger.info("[Telemetry] Instrumented OpenAI Agents SDK")
            except Exception as e:
                logger.error(f"Failed to instrument OpenAI Agents: {e}")

    return new_instrumentors


# Keep the old function for backward compatibility (deprecated)
def initialize_telemetry(providers: list, agent_id: str):
    """
    DEPRECATED: Use Client.initialize_telemetry() instead.
    This function is kept for backward compatibility but will not work correctly
    in multi-threaded environments.
    """
    logger.warning("[Telemetry] initialize_telemetry() is deprecated. Telemetry should be initialized via Client.")
    # Return empty tuple to satisfy old callers
    return None, []