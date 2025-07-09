"""Universal image interceptor for all LLM providers to capture images from multimodal requests"""
import logging
import os
from typing import Any, Dict, List, Union, Tuple
from .image_storage import store_image, get_stored_images
from .text_storage import store_text, clear_stored_texts

logger = logging.getLogger("Lucidic")
DEBUG = os.getenv("LUCIDIC_DEBUG", "False") == "True"


class UniversalImageInterceptor:
    """Universal image interceptor that can handle different provider formats"""
    
    @staticmethod
    def extract_and_store_images_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and store images from OpenAI-style messages
        
        OpenAI format:
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }
        """
        processed_messages = []
        clear_stored_texts()  # Clear any previous texts
        
        for msg_idx, message in enumerate(messages):
            if isinstance(message, dict):
                content = message.get('content')
                if isinstance(content, list):
                    # Extract and store text content for multimodal messages
                    text_parts = []
                    has_images = False
                    
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                            elif item.get('type') == 'image_url':
                                has_images = True
                                image_data = item.get('image_url', {})
                                if isinstance(image_data, dict):
                                    url = image_data.get('url', '')
                                    if url.startswith('data:image'):
                                        # Store the image
                                        placeholder = store_image(url)
                                        if DEBUG:
                                            logger.info(f"[Universal Interceptor] Stored OpenAI image, placeholder: {placeholder}")
                    
                    # If we have both text and images, store the text separately
                    if text_parts and has_images:
                        combined_text = ' '.join(text_parts)
                        store_text(combined_text, msg_idx)
                        if DEBUG:
                            logger.info(f"[Universal Interceptor] Stored text for multimodal message {msg_idx}: {combined_text[:50]}...")
                
                processed_messages.append(message)
        return processed_messages
    
    @staticmethod
    def extract_and_store_images_anthropic(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and store images from Anthropic-style messages
        
        Anthropic format:
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}
            ]
        }
        """
        processed_messages = []
        for message in messages:
            if isinstance(message, dict):
                content = message.get('content')
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'image':
                            source = item.get('source', {})
                            if isinstance(source, dict):
                                data = source.get('data', '')
                                media_type = source.get('media_type', 'image/jpeg')
                                if data:
                                    # Convert to data URL format for consistency
                                    data_url = f"data:{media_type};base64,{data}"
                                    placeholder = store_image(data_url)
                                    if DEBUG:
                                        logger.info(f"[Universal Interceptor] Stored Anthropic image, placeholder: {placeholder}")
                processed_messages.append(message)
        return processed_messages
    
    @staticmethod
    def extract_and_store_images_google(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and store images from Google/Gemini-style messages
        
        Google format can vary but often uses:
        {
            "parts": [
                {"text": "What's in this image?"},
                {"inline_data": {"mime_type": "image/jpeg", "data": "..."}}
            ]
        }
        """
        processed_messages = []
        for message in messages:
            if isinstance(message, dict):
                parts = message.get('parts', [])
                for part in parts:
                    if isinstance(part, dict) and 'inline_data' in part:
                        inline_data = part['inline_data']
                        if isinstance(inline_data, dict):
                            data = inline_data.get('data', '')
                            mime_type = inline_data.get('mime_type', 'image/jpeg')
                            if data:
                                # Convert to data URL format
                                data_url = f"data:{mime_type};base64,{data}"
                                placeholder = store_image(data_url)
                                if DEBUG:
                                    logger.info(f"[Universal Interceptor] Stored Google image, placeholder: {placeholder}")
                processed_messages.append(message)
        return processed_messages
    
    @staticmethod
    def intercept_images(messages: Any, provider: str = "auto") -> Any:
        """Universal image interception for any provider
        
        Args:
            messages: The messages to process
            provider: The provider type ("openai", "anthropic", "google", "auto")
                     If "auto", will try to detect the format
        
        Returns:
            The processed messages (unchanged, but images are stored)
        """
        if not messages or not isinstance(messages, list):
            return messages
        
        # Auto-detect provider format if needed
        if provider == "auto":
            provider = UniversalImageInterceptor._detect_provider_format(messages)
        
        # Process based on provider
        if provider == "openai":
            return UniversalImageInterceptor.extract_and_store_images_openai(messages)
        elif provider == "anthropic":
            return UniversalImageInterceptor.extract_and_store_images_anthropic(messages)
        elif provider == "google":
            return UniversalImageInterceptor.extract_and_store_images_google(messages)
        else:
            if DEBUG:
                logger.warning(f"[Universal Interceptor] Unknown provider format: {provider}")
            return messages
    
    @staticmethod
    def _detect_provider_format(messages: List[Dict[str, Any]]) -> str:
        """Detect the provider format based on message structure"""
        for message in messages:
            if isinstance(message, dict):
                # Check for OpenAI format
                content = message.get('content')
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'image_url' and 'image_url' in item:
                                return "openai"
                            elif item.get('type') == 'image' and 'source' in item:
                                return "anthropic"
                
                # Check for Google format
                if 'parts' in message:
                    return "google"
        
        return "unknown"
    
    @staticmethod
    def create_interceptor(provider: str):
        """Create a function interceptor for a specific provider
        
        Args:
            provider: The provider type ("openai", "anthropic", "google")
        
        Returns:
            A function that can wrap the provider's API call
        """
        def interceptor(original_func):
            def wrapper(*args, **kwargs):
                # Extract messages from kwargs (common location)
                messages = kwargs.get('messages', None)
                
                # For Anthropic, messages might be in args[1]
                if messages is None and len(args) > 1 and isinstance(args[1], list):
                    messages = args[1]
                
                # Intercept images if messages found
                if messages:
                    UniversalImageInterceptor.intercept_images(messages, provider)
                
                # Call the original function
                return original_func(*args, **kwargs)
            
            return wrapper
        
        return interceptor
    
    @staticmethod
    def create_async_interceptor(provider: str):
        """Create an async function interceptor for a specific provider"""
        def interceptor(original_func):
            async def wrapper(*args, **kwargs):
                # Extract messages from kwargs (common location)
                messages = kwargs.get('messages', None)
                
                # For Anthropic, messages might be in args[1]
                if messages is None and len(args) > 1 and isinstance(args[1], list):
                    messages = args[1]
                
                # Intercept images if messages found
                if messages:
                    UniversalImageInterceptor.intercept_images(messages, provider)
                
                # Call the original function
                return await original_func(*args, **kwargs)
            
            return wrapper
        
        return interceptor


def patch_openai_client(client):
    """Patch an OpenAI client instance to intercept images"""
    interceptor = UniversalImageInterceptor.create_interceptor("openai")
    
    if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
        original_create = client.chat.completions.create
        client.chat.completions.create = interceptor(original_create)
        if DEBUG:
            logger.info("[Universal Interceptor] Patched OpenAI client for image interception")
    return client


def patch_anthropic_client(client):
    """Patch an Anthropic client instance to intercept images"""
    interceptor = UniversalImageInterceptor.create_interceptor("anthropic")
    async_interceptor = UniversalImageInterceptor.create_async_interceptor("anthropic")
    
    if hasattr(client, 'messages'):
        if hasattr(client.messages, 'create'):
            original_create = client.messages.create
            client.messages.create = interceptor(original_create)
        
        # Handle async version
        if hasattr(client.messages, 'acreate'):
            original_acreate = client.messages.acreate
            client.messages.acreate = async_interceptor(original_acreate)
        
        if DEBUG:
            logger.info("[Universal Interceptor] Patched Anthropic client for image interception")
    return client


def patch_google_client(client):
    """Patch a Google/Gemini client instance to intercept images"""
    interceptor = UniversalImageInterceptor.create_interceptor("google")
    
    # Google's client structure varies, but often uses generate_content
    if hasattr(client, 'generate_content'):
        original_generate = client.generate_content
        client.generate_content = interceptor(original_generate)
        if DEBUG:
            logger.info("[Universal Interceptor] Patched Google client for image interception")
    return client