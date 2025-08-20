"""OpenTelemetry-based handlers that maintain backward compatibility

Adds guards to avoid repeated monkey-patching under concurrent init.
"""
import logging
from typing import Optional

from .base_provider import BaseProvider
from .otel_init import LucidicTelemetry

logger = logging.getLogger("Lucidic")

import threading

_patch_lock = threading.Lock()
_openai_patched = False
_anthropic_patched = False

class OTelOpenAIHandler(BaseProvider):
    """OpenAI handler using OpenTelemetry instrumentation"""
    
    def __init__(self):
        super().__init__()
        self._provider_name = "OpenAI"
        self.telemetry = LucidicTelemetry()
        
    def handle_response(self, response, kwargs, session: Optional = None):
        """Not needed with OpenTelemetry approach"""
        return response
        
    def override(self):
        """Enable OpenAI instrumentation"""
        try:
            from lucidicai.client import Client
            client = Client()
            
            # Initialize telemetry if needed
            if not self.telemetry.is_initialized():
                self.telemetry.initialize(agent_id=client.agent_id)
                
            # Instrument OpenAI
            self.telemetry.instrument_providers(["openai"])
            
            # Also patch OpenAI client to intercept images
            try:
                with _patch_lock:
                    global _openai_patched
                    if not _openai_patched:
                        import openai
                        from .utils.universal_image_interceptor import UniversalImageInterceptor, patch_openai_client
                        interceptor = UniversalImageInterceptor.create_interceptor("openai")
                        if hasattr(openai, 'ChatCompletion'):
                            original = openai.ChatCompletion.create
                            openai.ChatCompletion.create = interceptor(original)
                        original_client_init = openai.OpenAI.__init__
                        def patched_init(self, *args, **kwargs):
                            original_client_init(self, *args, **kwargs)
                            patch_openai_client(self)
                        openai.OpenAI.__init__ = patched_init
                        if hasattr(openai, 'AsyncOpenAI'):
                            original_async_init = openai.AsyncOpenAI.__init__
                            def patched_async_init(self, *args, **kwargs):
                                original_async_init(self, *args, **kwargs)
                                patch_openai_client(self)
                            openai.AsyncOpenAI.__init__ = patched_async_init
                        _openai_patched = True
            except Exception as e:
                logger.warning(f"Could not patch OpenAI for image interception: {e}")
            
            logger.info("[OTel OpenAI Handler] Instrumentation enabled")
            
        except Exception as e:
            logger.error(f"Failed to enable OpenAI instrumentation: {e}")
            raise
            
    def undo_override(self):
        """Disable instrumentation"""
        # Telemetry uninstrumentation is handled globally
        logger.info("[OTel OpenAI Handler] Instrumentation will be disabled on shutdown")


class OTelAnthropicHandler(BaseProvider):
    """Anthropic handler using OpenTelemetry instrumentation"""
    
    def __init__(self):
        super().__init__()
        self._provider_name = "Anthropic"
        self.telemetry = LucidicTelemetry()
        
    def handle_response(self, response, kwargs, session: Optional = None):
        """Not needed with OpenTelemetry approach"""
        return response
        
    def override(self):
        """Enable Anthropic instrumentation"""
        try:
            from lucidicai.client import Client
            client = Client()
            
            # Initialize telemetry if needed
            if not self.telemetry.is_initialized():
                self.telemetry.initialize(agent_id=client.agent_id)
                
            # Instrument Anthropic
            self.telemetry.instrument_providers(["anthropic"])
            
            # Also patch Anthropic client to intercept images
            try:
                with _patch_lock:
                    global _anthropic_patched
                    if not _anthropic_patched:
                        import anthropic
                        from .utils.universal_image_interceptor import UniversalImageInterceptor, patch_anthropic_client
                        interceptor = UniversalImageInterceptor.create_interceptor("anthropic")
                        async_interceptor = UniversalImageInterceptor.create_async_interceptor("anthropic")
                        original_client_init = anthropic.Anthropic.__init__
                        def patched_init(self, *args, **kwargs):
                            original_client_init(self, *args, **kwargs)
                            patch_anthropic_client(self)
                        anthropic.Anthropic.__init__ = patched_init
                        if hasattr(anthropic, 'AsyncAnthropic'):
                            original_async_init = anthropic.AsyncAnthropic.__init__
                            def patched_async_init(self, *args, **kwargs):
                                original_async_init(self, *args, **kwargs)
                                patch_anthropic_client(self)
                            anthropic.AsyncAnthropic.__init__ = patched_async_init
                        _anthropic_patched = True
            except Exception as e:
                logger.warning(f"Could not patch Anthropic for image interception: {e}")
            
            logger.info("[OTel Anthropic Handler] Instrumentation enabled")
            
        except Exception as e:
            logger.error(f"Failed to enable Anthropic instrumentation: {e}")
            raise
            
    def undo_override(self):
        """Disable instrumentation"""
        logger.info("[OTel Anthropic Handler] Instrumentation will be disabled on shutdown")


class OTelLangChainHandler(BaseProvider):
    """LangChain handler using OpenTelemetry instrumentation"""
    
    def __init__(self):
        super().__init__()
        self._provider_name = "LangChain"
        self.telemetry = LucidicTelemetry()
        
    def handle_response(self, response, kwargs, session: Optional = None):
        """Not needed with OpenTelemetry approach"""
        return response
        
    def override(self):
        """Enable LangChain instrumentation"""
        try:
            from lucidicai.client import Client
            client = Client()
            
            # Initialize telemetry if needed
            if not self.telemetry.is_initialized():
                self.telemetry.initialize(agent_id=client.agent_id)
                
            # Instrument LangChain
            self.telemetry.instrument_providers(["langchain"])
            
            logger.info("[OTel LangChain Handler] Instrumentation enabled")
            
        except Exception as e:
            logger.error(f"Failed to enable LangChain instrumentation: {e}")
            raise
            
    def undo_override(self):
        """Disable instrumentation"""
        logger.info("[OTel LangChain Handler] Instrumentation will be disabled on shutdown")


class OTelPydanticAIHandler(BaseProvider):
    """Pydantic AI handler - requires custom implementation"""
    
    def __init__(self):
        super().__init__()
        self._provider_name = "PydanticAI"
        self.telemetry = LucidicTelemetry()
        self._original_methods = {}
        
    def handle_response(self, response, kwargs, session: Optional = None):
        """Handle Pydantic AI responses"""
        return response
        
    def override(self):
        """Enable Pydantic AI instrumentation"""
        try:
            from lucidicai.client import Client
            client = Client()
            
            # Initialize telemetry if needed
            if not self.telemetry.is_initialized():
                self.telemetry.initialize(agent_id=client.agent_id)
                
            # For now, we'll use the original Pydantic AI handler
            # until OpenLLMetry adds support
            from .pydantic_ai_handler import PydanticAIHandler
            self._fallback_handler = PydanticAIHandler()
            self._fallback_handler.override()
            
            logger.info("[OTel PydanticAI Handler] Using fallback handler until OpenLLMetry support is available")
            
        except Exception as e:
            logger.error(f"Failed to enable Pydantic AI instrumentation: {e}")
            raise
            
    def undo_override(self):
        """Disable instrumentation"""
        if hasattr(self, '_fallback_handler'):
            self._fallback_handler.undo_override()
        logger.info("[OTel PydanticAI Handler] Instrumentation disabled")


class OTelOpenAIAgentsHandler(BaseProvider):
    """OpenAI Agents handler using OpenTelemetry instrumentation"""
    
    def __init__(self):
        super().__init__()
        self._provider_name = "OpenAI Agents"
        self.telemetry = LucidicTelemetry()
        self._is_instrumented = False
        
    def handle_response(self, response, kwargs, session: Optional = None):
        """Not needed with OpenTelemetry approach"""
        return response
        
    def override(self):
        """Enable OpenAI Agents instrumentation"""
        try:
            from lucidicai.client import Client
            client = Client()
            
            # Initialize telemetry if needed
            if not self.telemetry.is_initialized():
                self.telemetry.initialize(agent_id=client.agent_id)
                
            # Only instrument OpenAI Agents (it will handle OpenAI calls internally)
            self.telemetry.instrument_providers(["openai_agents"])
            
            self._is_instrumented = True
            
            logger.info("[OTel OpenAI Agents Handler] Full instrumentation enabled")
            
        except Exception as e:
            logger.error(f"Failed to enable OpenAI Agents instrumentation: {e}")
            raise
            
    def undo_override(self):
        """Disable instrumentation"""
        self._is_instrumented = False
        logger.info("[OTel OpenAI Agents Handler] Instrumentation will be disabled on shutdown")


class OTelLiteLLMHandler(BaseProvider):
    """LiteLLM handler using CustomLogger callback system"""
    
    def __init__(self):
        super().__init__()
        self._provider_name = "LiteLLM"
        self.telemetry = LucidicTelemetry()
        self._callback = None
        self._original_callbacks = None
        
    def handle_response(self, response, kwargs, session: Optional = None):
        """Not needed with callback approach"""
        return response
        
    def override(self):
        """Enable LiteLLM instrumentation via callbacks"""
        try:
            import litellm
            from lucidicai.client import Client
            from .litellm_bridge import LucidicLiteLLMCallback
            
            client = Client()
            
            # Initialize telemetry if needed
            if not self.telemetry.is_initialized():
                self.telemetry.initialize(agent_id=client.agent_id)
            
            # Create our callback instance
            self._callback = LucidicLiteLLMCallback()
            
            # Store original callbacks
            self._original_callbacks = litellm.callbacks if hasattr(litellm, 'callbacks') else None
            
            # Add our callback to LiteLLM
            if litellm.callbacks is None:
                litellm.callbacks = []
            
            # Add our callback if not already present
            if self._callback not in litellm.callbacks:
                litellm.callbacks.append(self._callback)
            
            # Also set success/failure callbacks
            if not hasattr(litellm, 'success_callback') or litellm.success_callback is None:
                litellm.success_callback = []
            if not hasattr(litellm, 'failure_callback') or litellm.failure_callback is None:
                litellm.failure_callback = []
            
            # Add to callback lists if not present
            if self._callback not in litellm.success_callback:
                litellm.success_callback.append(self._callback)
            if self._callback not in litellm.failure_callback:
                litellm.failure_callback.append(self._callback)
            
            logger.info("[OTel LiteLLM Handler] Callback instrumentation enabled")
            
        except ImportError:
            logger.error("LiteLLM not installed. Please install with: pip install litellm")
            raise
        except Exception as e:
            logger.error(f"Failed to enable LiteLLM instrumentation: {e}")
            raise
            
    def undo_override(self):
        """Disable LiteLLM instrumentation"""
        try:
            import litellm
            
            # Wait for pending callbacks to complete before cleanup
            if self._callback and hasattr(self._callback, 'wait_for_pending_callbacks'):
                logger.info("[OTel LiteLLM Handler] Waiting for pending callbacks to complete...")
                self._callback.wait_for_pending_callbacks(timeout=5.0)
            
            # Remove our callback from all callback lists
            if self._callback:
                if hasattr(litellm, 'callbacks') and litellm.callbacks and self._callback in litellm.callbacks:
                    litellm.callbacks.remove(self._callback)
                    
                if hasattr(litellm, 'success_callback') and litellm.success_callback and self._callback in litellm.success_callback:
                    litellm.success_callback.remove(self._callback)
                    
                if hasattr(litellm, 'failure_callback') and litellm.failure_callback and self._callback in litellm.failure_callback:
                    litellm.failure_callback.remove(self._callback)
            
            # Restore original callbacks if we stored them
            if self._original_callbacks is not None:
                litellm.callbacks = self._original_callbacks
            
            logger.info("[OTel LiteLLM Handler] Instrumentation disabled")
            
        except Exception as e:
            logger.error(f"Error disabling LiteLLM instrumentation: {e}")


class OTelBedrockHandler(BaseProvider):
    """AWS Bedrock handler using OpenTelemetry instrumentation"""
    
    def __init__(self):
        super().__init__()
        self._provider_name = "Bedrock"
        self.telemetry = LucidicTelemetry()
    
    def handle_response(self, response, kwargs, session: Optional = None):
        return response
    
    def override(self):
        try:
            from lucidicai.client import Client
            client = Client()
            if not self.telemetry.is_initialized():
                self.telemetry.initialize(agent_id=client.agent_id)
            self.telemetry.instrument_providers(["bedrock"])
            logger.info("[OTel Bedrock Handler] Instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to enable Bedrock instrumentation: {e}")
            raise
    
    def undo_override(self):
        logger.info("[OTel Bedrock Handler] Instrumentation will be disabled on shutdown")


class OTelGoogleGenerativeAIHandler(BaseProvider):
    """Google Generative AI handler using OpenTelemetry instrumentation"""
    
    def __init__(self):
        super().__init__()
        self._provider_name = "Google Generative AI"
        self.telemetry = LucidicTelemetry()
    
    def handle_response(self, response, kwargs, session: Optional = None):
        return response
    
    def override(self):
        try:
            from lucidicai.client import Client
            client = Client()
            if not self.telemetry.is_initialized():
                self.telemetry.initialize(agent_id=client.agent_id)
            self.telemetry.instrument_providers(["google"])
            # Best-effort image interception for Google clients where applicable
            try:
                from .utils.universal_image_interceptor import patch_google_client, patch_google_genai
                _ = patch_google_client
                patch_google_genai()
            except Exception as e:
                logger.debug(f"[OTel Google Handler] Image interception not applied: {e}")
            logger.info("[OTel Google Handler] Instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to enable Google Generative AI instrumentation: {e}")
            raise
    
    def undo_override(self):
        logger.info("[OTel Google Handler] Instrumentation will be disabled on shutdown")


class OTelVertexAIHandler(BaseProvider):
    """Vertex AI handler using OpenTelemetry instrumentation"""
    
    def __init__(self):
        super().__init__()
        self._provider_name = "Vertex AI"
        self.telemetry = LucidicTelemetry()
    
    def handle_response(self, response, kwargs, session: Optional = None):
        return response
    
    def override(self):
        try:
            from lucidicai.client import Client
            client = Client()
            if not self.telemetry.is_initialized():
                self.telemetry.initialize(agent_id=client.agent_id)
            self.telemetry.instrument_providers(["vertexai"])
            # Best-effort image interception for Vertex AI clients where applicable
            try:
                from .utils.universal_image_interceptor import patch_vertexai_client
                _ = patch_vertexai_client
            except Exception as e:
                logger.debug(f"[OTel Vertex Handler] Image interception not applied: {e}")
            logger.info("[OTel Vertex Handler] Instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to enable Vertex AI instrumentation: {e}")
            raise
    
    def undo_override(self):
        logger.info("[OTel Vertex Handler] Instrumentation will be disabled on shutdown")


class OTelCohereHandler(BaseProvider):
    """Cohere handler using OpenTelemetry instrumentation"""
    
    def __init__(self):
        super().__init__()
        self._provider_name = "Cohere"
        self.telemetry = LucidicTelemetry()
    
    def handle_response(self, response, kwargs, session: Optional = None):
        return response
    
    def override(self):
        try:
            from lucidicai.client import Client
            client = Client()
            if not self.telemetry.is_initialized():
                self.telemetry.initialize(agent_id=client.agent_id)
            self.telemetry.instrument_providers(["cohere"])
            logger.info("[OTel Cohere Handler] Instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to enable Cohere instrumentation: {e}")
            raise
    
    def undo_override(self):
        logger.info("[OTel Cohere Handler] Instrumentation will be disabled on shutdown")


class OTelGroqHandler(BaseProvider):
    """Groq handler using OpenTelemetry instrumentation"""
    
    def __init__(self):
        super().__init__()
        self._provider_name = "Groq"
        self.telemetry = LucidicTelemetry()
    
    def handle_response(self, response, kwargs, session: Optional = None):
        return response
    
    def override(self):
        try:
            from lucidicai.client import Client
            client = Client()
            if not self.telemetry.is_initialized():
                self.telemetry.initialize(agent_id=client.agent_id)
            self.telemetry.instrument_providers(["groq"])
            # Best-effort image interception for Groq (OpenAI-compatible)
            try:
                import groq  # noqa: F401
                from .utils.universal_image_interceptor import UniversalImageInterceptor
                # We cannot reliably patch class constructors here without instance; users calling Groq client
                # will still have images captured via OpenLLMetry attributes; optional future improvement.
                _ = UniversalImageInterceptor
            except Exception as e:
                logger.debug(f"[OTel Groq Handler] Image interception not applied: {e}")
            logger.info("[OTel Groq Handler] Instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to enable Groq instrumentation: {e}")
            raise
    
    def undo_override(self):
        logger.info("[OTel Groq Handler] Instrumentation will be disabled on shutdown")