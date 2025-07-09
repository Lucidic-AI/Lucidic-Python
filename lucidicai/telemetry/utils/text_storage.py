"""Thread-local storage for text content from multimodal messages"""
import threading
from typing import List, Optional
import logging

logger = logging.getLogger("Lucidic")

# Thread-local storage for text content
_text_storage = threading.local()

def store_text(text: str, message_index: int = 0) -> None:
    """Store text content for a message
    
    Args:
        text: The text content to store
        message_index: The index of the message (default 0)
    """
    if not hasattr(_text_storage, 'texts'):
        _text_storage.texts = {}
    
    _text_storage.texts[message_index] = text
    logger.debug(f"[TextStorage] Stored text for message {message_index}: {text[:50]}...")

def get_stored_text(message_index: int = 0) -> Optional[str]:
    """Get stored text for a message
    
    Args:
        message_index: The index of the message (default 0)
        
    Returns:
        The stored text or None if not found
    """
    if not hasattr(_text_storage, 'texts'):
        return None
    
    return _text_storage.texts.get(message_index)

def get_all_stored_texts() -> dict:
    """Get all stored texts
    
    Returns:
        Dictionary of message_index -> text
    """
    if not hasattr(_text_storage, 'texts'):
        return {}
    
    return _text_storage.texts.copy()

def clear_stored_texts() -> None:
    """Clear all stored texts"""
    if hasattr(_text_storage, 'texts'):
        _text_storage.texts = {}
        logger.debug("[TextStorage] Cleared all stored texts")