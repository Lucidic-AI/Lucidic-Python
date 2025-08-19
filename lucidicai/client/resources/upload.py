"""Upload API resource handler for images and screenshots"""

import base64
import io
from typing import Optional, Dict, Any, List
from PIL import Image
from lucidicai.client.http_client import HttpClient
from lucidicai.util.logger import logger


class UploadResource:
    """Handles upload-related API operations"""
    
    def __init__(self, http_client: HttpClient):
        """Initialize upload resource with HTTP client
        
        Args:
            http_client: HTTP client instance for API requests
        """
        self.http = http_client
    
    def get_presigned_url(
        self,
        agent_id: str,
        session_id: Optional[str] = None,
        step_id: Optional[str] = None,
        event_id: Optional[str] = None,
        nth_screenshot: Optional[int] = None
    ) -> str:
        """Get presigned URL for S3 upload
        
        Args:
            agent_id: Agent ID
            session_id: Optional session ID
            step_id: Optional step ID
            event_id: Optional event ID
            nth_screenshot: Screenshot number for events with multiple screenshots
            
        Returns:
            Presigned URL for upload
        """
        params = {
            "agent_id": agent_id,
            "session_id": session_id,
            "step_id": step_id,
            "event_id": event_id,
            "nth_screenshot": nth_screenshot,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        response = self.http.get("getpresigneduploadurl", params)
        return response.get("presigned_url")
    
    def upload_image_to_s3(self, presigned_url: str, image_data: bytes) -> bool:
        """Upload image data to S3 using presigned URL
        
        Args:
            presigned_url: Presigned S3 URL
            image_data: Image data as bytes
            
        Returns:
            True if upload successful
        """
        import requests
        try:
            response = requests.put(
                presigned_url,
                data=image_data,
                headers={"Content-Type": "image/jpeg"}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to upload image to S3: {e}")
            return False
    
    def process_screenshot(
        self,
        screenshot: Optional[str] = None,
        screenshot_path: Optional[str] = None
    ) -> Optional[bytes]:
        """Process screenshot from base64 or file path
        
        Args:
            screenshot: Base64 encoded screenshot
            screenshot_path: Path to screenshot file
            
        Returns:
            Processed image data as JPEG bytes, or None if no input
        """
        if not screenshot and not screenshot_path:
            return None
        
        try:
            if screenshot_path:
                # Load from file
                img = Image.open(screenshot_path)
            elif screenshot:
                # Decode from base64
                img_data = base64.b64decode(screenshot)
                img = Image.open(io.BytesIO(img_data))
            else:
                return None
            
            # Convert to RGB if necessary (removes alpha channel)
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPEG
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to process screenshot: {e}")
            return None
    
    def upload_screenshots(
        self,
        agent_id: str,
        screenshots: List[str],
        event_id: str,
        session_id: Optional[str] = None
    ) -> int:
        """Upload multiple screenshots for an event
        
        Args:
            agent_id: Agent ID
            screenshots: List of base64 encoded screenshots
            event_id: Event ID
            session_id: Optional session ID
            
        Returns:
            Number of successfully uploaded screenshots
        """
        uploaded = 0
        for i, screenshot in enumerate(screenshots):
            try:
                # Get presigned URL
                url = self.get_presigned_url(
                    agent_id=agent_id,
                    session_id=session_id,
                    event_id=event_id,
                    nth_screenshot=i
                )
                
                # Process and upload
                img_data = self.process_screenshot(screenshot=screenshot)
                if img_data and self.upload_image_to_s3(url, img_data):
                    uploaded += 1
            except Exception as e:
                logger.error(f"Failed to upload screenshot {i}: {e}")
        
        return uploaded