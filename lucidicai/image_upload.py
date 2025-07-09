import base64
import io

import requests
from PIL import Image

def get_presigned_url(agent_id, step_id=None, session_id=None, event_id=None, nthscreenshot=None):
    """
    Get a presigned URL for uploading an image to S3.
    
    Args:
        agent_id: The ID of the agent.
        step_id: Optional step ID for the image. Either supply step_id or session_id.
        session_id: Optional session ID for the image. Either supply step_id or session_id.
        event_id: Optional event ID for the image.
        nthscreenshot: Optional nth screenshot for the image.
    
    Returns:
        A tuple containing the presigned URL, bucket name, and object key.
    """
    from .client import Client
    request_data = {
        "agent_id": agent_id,
    }
    if step_id is not None:
        request_data["step_id"] = step_id
    if session_id is not None:
        request_data["session_id"] = session_id
    if event_id is not None:
        request_data["event_id"] = event_id
        if nthscreenshot is None:
            raise ValueError("nth_screenshot is required when event_id is provided")
        request_data["nth_screenshot"] = nthscreenshot

    response = Client().make_request('getpresigneduploadurl', 'GET', request_data)
    return response['presigned_url'], response['bucket_name'], response['object_key']

def upload_image_to_s3(url, image, format):
    """
    Upload an image to S3.
    
    Args:
        url: The presigned URL for the upload.
        image: The image data to upload.
        format: The format of the image ("JPEG" or "GIF").
    
    Raises:
        ValueError: If the format is not supported.
    """
    if format == "JPEG":
        # Handle data URIs by extracting just the base64 part
        if isinstance(image, str) and image.startswith('data:'):
            # Extract base64 data from data URI
            base64_data = image.split(',')[1] if ',' in image else image
        else:
            base64_data = image
        
        image_stream = io.BytesIO(base64.b64decode(base64_data))
        image_stream.seek(0)
        pil_image = Image.open(image_stream)
        if pil_image.mode in ("RGBA", "LA"):  # TODO: Natively support PNGs
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
        image_obj = image
        content_type = "image/gif"
    upload_response = requests.put(
        url,
        data=image_obj.getvalue(),
        headers={"Content-Type": content_type}
    )
    upload_response.raise_for_status()

def screenshot_path_to_jpeg(screenshot_path):
    img = Image.open(screenshot_path)
    img = img.convert("RGB") 
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def extract_base64_images(data):
    """Extract base64 image URLs from various data structures
    
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
        for value in data.values():
            images.extend(extract_base64_images(value))
    elif isinstance(data, list):
        for item in data:
            images.extend(extract_base64_images(item))
            
    return images