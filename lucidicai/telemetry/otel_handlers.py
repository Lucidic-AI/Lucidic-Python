"""OpenTelemetry-based handlers that maintain backward compatibility"""
import logging
from typing import Optional

from .base_provider import BaseProvider
from .otel_init import LucidicTelemetry

logger = logging.getLogger("Lucidic")


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
                import openai
                from .utils.universal_image_interceptor import UniversalImageInterceptor, patch_openai_client
                
                # Create interceptor for OpenAI
                interceptor = UniversalImageInterceptor.create_interceptor("openai")
                
                # Patch the module-level create method
                if hasattr(openai, 'ChatCompletion'):
                    # Old API
                    original = openai.ChatCompletion.create
                    openai.ChatCompletion.create = interceptor(original)
                    
                # Also patch any client instances that might be created
                original_client_init = openai.OpenAI.__init__
                def patched_init(self, *args, **kwargs):
                    original_client_init(self, *args, **kwargs)
                    # Patch this instance
                    patch_openai_client(self)
                
                openai.OpenAI.__init__ = patched_init
                
                # Also patch AsyncOpenAI
                if hasattr(openai, 'AsyncOpenAI'):
                    original_async_init = openai.AsyncOpenAI.__init__
                    def patched_async_init(self, *args, **kwargs):
                        original_async_init(self, *args, **kwargs)
                        # Patch this instance
                        patch_openai_client(self)
                    
                    openai.AsyncOpenAI.__init__ = patched_async_init
                
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
                import anthropic
                from .utils.universal_image_interceptor import UniversalImageInterceptor, patch_anthropic_client
                
                # Create interceptors for Anthropic
                interceptor = UniversalImageInterceptor.create_interceptor("anthropic")
                async_interceptor = UniversalImageInterceptor.create_async_interceptor("anthropic")
                
                # Patch any client instances that might be created
                original_client_init = anthropic.Anthropic.__init__
                def patched_init(self, *args, **kwargs):
                    original_client_init(self, *args, **kwargs)
                    # Patch this instance
                    patch_anthropic_client(self)
                
                anthropic.Anthropic.__init__ = patched_init
                
                # Also patch async client
                if hasattr(anthropic, 'AsyncAnthropic'):
                    original_async_init = anthropic.AsyncAnthropic.__init__
                    def patched_async_init(self, *args, **kwargs):
                        original_async_init(self, *args, **kwargs)
                        # Patch this instance
                        patch_anthropic_client(self)
                    
                    anthropic.AsyncAnthropic.__init__ = patched_async_init
                
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