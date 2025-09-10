"""Consolidated image handling utilities.

This module unifies all image-related functionality from:
- image_upload.py
- telemetry/utils/image_storage.py
- Various extraction functions scattered across the codebase
"""
import base64
import io
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
import requests

logger = logging.getLogger("Lucidic")


class ImageHandler:
    """Centralized image handling for the SDK."""
    
    # Thread-local storage for images (from telemetry)
    _thread_local = threading.local()
    
    @classmethod
    def extract_base64_images(cls, data: Any) -> List[str]:
        """Extract base64 image URLs from various data structures.
        
        Args:
            data: Can be a string, dict, list, or nested structure containing image data
            
        Returns:
            List of base64 image data URLs (data:image/...)
        """
        images = []
        
        if isinstance(data, str):
            if data.startswith('data:image'):
                images.append(data)
        elif isinstance(data, dict):
            # Check for specific image fields
            if 'image' in data and isinstance(data['image'], str):
                if data['image'].startswith('data:image'):
                    images.append(data['image'])
            
            # Check for image_url structures (OpenAI format)
            if data.get('type') == 'image_url':
                image_url = data.get('image_url', {})
                if isinstance(image_url, dict) and 'url' in image_url:
                    url = image_url['url']
                    if url.startswith('data:image'):
                        images.append(url)
            
            # Recursively check all values
            for value in data.values():
                images.extend(cls.extract_base64_images(value))
                
        elif isinstance(data, list):
            for item in data:
                images.extend(cls.extract_base64_images(item))
                
        return images
    
    @classmethod
    def extract_images_from_messages(cls, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract images from chat messages (OpenAI/Anthropic format).
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List of base64 image data URLs
        """
        images = []
        
        for message in messages:
            if not isinstance(message, dict):
                continue
                
            content = message.get('content', '')
            
            # Handle multimodal content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'image_url':
                            image_url = item.get('image_url', {})
                            if isinstance(image_url, dict):
                                url = image_url.get('url', '')
                                if url.startswith('data:image'):
                                    images.append(url)
                        elif item.get('type') == 'image':
                            # Anthropic format
                            source = item.get('source', {})
                            if isinstance(source, dict):
                                data = source.get('data', '')
                                if data:
                                    media_type = source.get('media_type', 'image/jpeg')
                                    images.append(f"data:{media_type};base64,{data}")
        
        return images
    
    @classmethod
    def store_image_thread_local(cls, image_base64: str) -> str:
        """Store image in thread-local storage and return placeholder.
        
        Used for working around OpenTelemetry attribute size limits.
        
        Args:
            image_base64: Base64 encoded image data
            
        Returns:
            Placeholder string for the stored image
        """
        if not hasattr(cls._thread_local, 'images'):
            cls._thread_local.images = []
        
        cls._thread_local.images.append(image_base64)
        placeholder = f"lucidic_image_{len(cls._thread_local.images) - 1}"
        
        logger.debug(f"[ImageHandler] Stored image in thread-local, placeholder: {placeholder}")
        return placeholder
    
    @classmethod
    def get_stored_images(cls) -> List[str]:
        """Get all images stored in thread-local storage."""
        if hasattr(cls._thread_local, 'images'):
            return cls._thread_local.images
        return []
    
    @classmethod
    def clear_stored_images(cls) -> None:
        """Clear thread-local image storage."""
        if hasattr(cls._thread_local, 'images'):
            cls._thread_local.images.clear()
    
    @classmethod
    def get_image_by_placeholder(cls, placeholder: str) -> Optional[str]:
        """Retrieve image by its placeholder from thread-local storage."""
        if hasattr(cls._thread_local, 'images') and placeholder.startswith('lucidic_image_'):
            try:
                index = int(placeholder.split('_')[-1])
                if 0 <= index < len(cls._thread_local.images):
                    return cls._thread_local.images[index]
            except (ValueError, IndexError):
                pass
        return None
    
    @classmethod
    def path_to_base64(cls, image_path: str, format: str = "JPEG") -> str:
        """Convert image file to base64 string.
        
        Args:
            image_path: Path to the image file
            format: Output format (JPEG or PNG)
            
        Returns:
            Base64 encoded image string
        """
        img = Image.open(image_path)
        
        if format == "JPEG":
            # Convert to RGB if necessary
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA" or img.mode == "LA":
                    alpha = img.split()[-1]
                    background.paste(img, mask=alpha)
                else:
                    background.paste(img)
                img = background
            else:
                img = img.convert("RGB")
        
        buffered = io.BytesIO()
        img.save(buffered, format=format)
        img_bytes = buffered.getvalue()
        
        return base64.b64encode(img_bytes).decode('utf-8')
    
    @classmethod
    def base64_to_pil(cls, base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image.
        
        Args:
            base64_str: Base64 encoded image (with or without data URI prefix)
            
        Returns:
            PIL Image object
        """
        # Remove data URI prefix if present
        if base64_str.startswith('data:'):
            base64_str = base64_str.split(',')[1] if ',' in base64_str else base64_str
        
        image_data = base64.b64decode(base64_str)
        image_stream = io.BytesIO(image_data)
        return Image.open(image_stream)
    
    @classmethod
    def prepare_for_upload(cls, image_data: Union[str, bytes], format: str = "JPEG") -> Tuple[io.BytesIO, str]:
        """Prepare image data for upload to S3.
        
        Args:
            image_data: Base64 string or raw bytes
            format: Target format (JPEG or GIF)
            
        Returns:
            Tuple of (BytesIO object, content-type)
        """
        if format == "JPEG":
            # Handle base64 string
            if isinstance(image_data, str):
                pil_image = cls.base64_to_pil(image_data)
            else:
                pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB
            if pil_image.mode in ("RGBA", "LA"):
                background = Image.new("RGB", pil_image.size, (255, 255, 255))
                alpha = pil_image.split()[-1]
                background.paste(pil_image, mask=alpha)
                pil_image = background
            else:
                pil_image = pil_image.convert("RGB")
            
            image_obj = io.BytesIO()
            pil_image.save(image_obj, format="JPEG")
            image_obj.seek(0)
            content_type = "image/jpeg"
            
        elif format == "GIF":
            if isinstance(image_data, str):
                image_data = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image_obj = io.BytesIO(image_data)
            content_type = "image/gif"
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return image_obj, content_type


class ImageUploader:
    """Handle image uploads to S3."""
    
    @staticmethod
    def get_presigned_url(
        agent_id: str,
        session_id: Optional[str] = None,
        event_id: Optional[str] = None,
        nthscreenshot: Optional[int] = None
    ) -> Tuple[str, str, str]:
        """Get a presigned URL for uploading an image to S3.
        
        Args:
            agent_id: The ID of the agent
            session_id: Optional session ID for the image
            event_id: Optional event ID for the image
            nthscreenshot: Optional nth screenshot for the image
        
        Returns:
            Tuple of (presigned_url, bucket_name, object_key)
        """
        # Import here to avoid circular dependency
        from ..sdk.init import get_http
        
        http = get_http()
        if not http:
            raise RuntimeError("SDK not initialized")
        
        request_data = {"agent_id": agent_id}
        
        if session_id:
            request_data["session_id"] = session_id
        
        if event_id:
            request_data["event_id"] = event_id
            if nthscreenshot is None:
                raise ValueError("nth_screenshot is required when event_id is provided")
            request_data["nth_screenshot"] = nthscreenshot
        
        response = http.get('getpresigneduploadurl', params=request_data)
        return response['presigned_url'], response['bucket_name'], response['object_key']
    
    @staticmethod
    def upload_to_s3(url: str, image_data: Union[str, bytes, io.BytesIO], format: str = "JPEG") -> None:
        """Upload an image to S3 using presigned URL.
        
        Args:
            url: The presigned URL for the upload
            image_data: Image data (base64 string, bytes, or BytesIO)
            format: Format of the image (JPEG or GIF)
        """
        # Prepare image for upload
        if isinstance(image_data, io.BytesIO):
            image_obj = image_data
            content_type = "image/jpeg" if format == "JPEG" else "image/gif"
        else:
            image_obj, content_type = ImageHandler.prepare_for_upload(image_data, format)
        
        # Upload to S3
        upload_response = requests.put(
            url,
            data=image_obj.getvalue() if hasattr(image_obj, 'getvalue') else image_obj,
            headers={"Content-Type": content_type}
        )
        upload_response.raise_for_status()
        
        logger.debug(f"[ImageUploader] Successfully uploaded image to S3")


# Convenience functions for backward compatibility
def extract_base64_images(data: Any) -> List[str]:
    """Extract base64 images from data (backward compatibility)."""
    return ImageHandler.extract_base64_images(data)


def screenshot_path_to_jpeg(screenshot_path: str) -> str:
    """Convert screenshot to base64 JPEG (backward compatibility)."""
    return ImageHandler.path_to_base64(screenshot_path, "JPEG")


def upload_image_to_s3(url: str, image: Union[str, bytes], format: str) -> None:
    """Upload image to S3 (backward compatibility)."""
    ImageUploader.upload_to_s3(url, image, format)


def get_presigned_url(
    agent_id: str,
    step_id: Optional[str] = None,
    session_id: Optional[str] = None,
    event_id: Optional[str] = None,
    nthscreenshot: Optional[int] = None
) -> Tuple[str, str, str]:
    """Get presigned URL (backward compatibility)."""
    # Note: step_id parameter is deprecated
    return ImageUploader.get_presigned_url(agent_id, session_id, event_id, nthscreenshot)