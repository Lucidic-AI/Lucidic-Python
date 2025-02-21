import requests
from typing import Optional
from requests import Response

class APIKeyVerificationError(Exception):
    """Base exception for API key verification errors"""
    pass

class SessionError(Exception):
    """Base exception for session-related errors"""
    pass

class SessionHTTPError(SessionError):
    """Exception for HTTP errors during session operations"""
    def __init__(self, message: str, response: Optional[Response] = None):
        super().__init__(message)
        self.response = response

class SessionResponseError(SessionError):
    """Exception for invalid responses during session operations"""
    def __init__(self, message: str, response_text: str):
        super().__init__(f"{message}: {response_text}")
        self.response_text = response_text

def handle_session_response(response: Response, required_fields: Optional[list[str]] = None) -> dict:
    """
    Handle a session-related API response with proper error handling.
    
    Args:
        response: The response from the server
        required_fields: List of fields that must be present in the response
        
    Returns:
        dict: The parsed response data
        
    Raises:
        SessionHTTPError: If there was an HTTP error
        SessionResponseError: If the response was invalid
    """
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        error_msg = "Unknown error"
        try:
            error_msg = e.response.json()
        except:
            if e.response.text:
                error_msg = e.response.text
            else:
                error_msg = str(e)
        raise SessionHTTPError(f"HTTP Error: {error_msg}", e.response)
    
    try:
        data = response.json()
    except ValueError as e:
        raise SessionResponseError("Invalid JSON response", response.text)
        
    if required_fields:
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise SessionResponseError(
                f"Missing required fields: {', '.join(missing)}", 
                response.text
            )
            
    return data