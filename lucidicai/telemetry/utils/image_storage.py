"""Thread-local storage for images to work around OpenTelemetry attribute size limits"""
import threading
import logging
import os

logger = logging.getLogger("Lucidic")
DEBUG = os.getenv("LUCIDIC_DEBUG", "False") == "True"

# Thread-local storage for images
_thread_local = threading.local()

def store_image(image_base64: str) -> str:
    """Store image in thread-local storage and return placeholder"""
    if not hasattr(_thread_local, 'images'):
        _thread_local.images = []
    
    _thread_local.images.append(image_base64)
    placeholder = f"lucidic_image_{len(_thread_local.images) - 1}"
    
    if DEBUG:
        logger.info(f"[ImageStorage] Stored image of size {len(image_base64)}, placeholder: {placeholder}")
    
    return placeholder

def get_stored_images():
    """Get all stored images"""
    if hasattr(_thread_local, 'images'):
        return _thread_local.images
    return []

def clear_stored_images():
    """Clear stored images"""
    if hasattr(_thread_local, 'images'):
        _thread_local.images.clear()

def get_image_by_placeholder(placeholder: str):
    """Get image by placeholder"""
    if hasattr(_thread_local, 'images') and placeholder.startswith('lucidic_image_'):
        try:
            index = int(placeholder.split('_')[-1])
            if 0 <= index < len(_thread_local.images):
                return _thread_local.images[index]
        except (ValueError, IndexError):
            pass
    return None