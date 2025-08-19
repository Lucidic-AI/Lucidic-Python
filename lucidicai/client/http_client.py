"""HTTP client for Lucidic API communication"""

import os
import requests
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from lucidicai.util.errors import APIKeyVerificationError, InvalidOperationError
from lucidicai.util.logger import logger


class HttpClient:
    """Base HTTP client for Lucidic API requests"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """Initialize HTTP client with API key and base URL
        
        Args:
            api_key: API key for authentication
            base_url: Optional base URL override
        """
        self.api_key = api_key
        self.base_url = base_url or self._get_default_base_url()
        self.session = self._create_session()
        self._configure_headers()
        
    def _get_default_base_url(self) -> str:
        """Get default base URL based on environment"""
        if os.getenv("LUCIDIC_DEBUG", "False").lower() == "true":
            return "http://localhost:8000/api"
        return "https://analytics.lucidic.ai/api"
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry configuration"""
        session = requests.Session()
        retry_cfg = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retry_cfg, pool_connections=20, pool_maxsize=100)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session
    
    def _configure_headers(self):
        """Configure default headers for all requests"""
        self.session.headers.update({
            "Authorization": f"Api-Key {self.api_key}",
            "User-Agent": "lucidic-python-sdk/2.0",
            "Content-Type": "application/json",
        })
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make GET request to API endpoint
        
        Args:
            endpoint: API endpoint path
            params: Optional query parameters
            
        Returns:
            JSON response data
        """
        return self._make_request("GET", endpoint, params=params)
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make POST request to API endpoint
        
        Args:
            endpoint: API endpoint path
            data: Request body data
            
        Returns:
            JSON response data
        """
        return self._make_request("POST", endpoint, json=data)
    
    def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make PUT request to API endpoint
        
        Args:
            endpoint: API endpoint path
            data: Request body data
            
        Returns:
            JSON response data
        """
        return self._make_request("PUT", endpoint, json=data)
    
    def delete(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make DELETE request to API endpoint
        
        Args:
            endpoint: API endpoint path
            params: Optional query parameters
            
        Returns:
            JSON response data
        """
        return self._make_request("DELETE", endpoint, params=params)
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            JSON response data
            
        Raises:
            APIKeyVerificationError: If API key is invalid
            InvalidOperationError: If request fails
        """
        url = f"{self.base_url}/{endpoint}"
        
        # Add timestamp to request data
        if method in ["POST", "PUT"]:
            if "json" in kwargs and kwargs["json"]:
                kwargs["json"]["current_time"] = datetime.now().astimezone(timezone.utc).isoformat()
            else:
                kwargs["json"] = {"current_time": datetime.now().astimezone(timezone.utc).isoformat()}
        
        try:
            response = self.session.request(method, url, **kwargs)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise InvalidOperationError("Cannot reach backend. Check your internet connection.")
        
        # Handle specific status codes
        if response.status_code == 401:
            raise APIKeyVerificationError("Invalid API key: 401 Unauthorized")
        elif response.status_code == 402:
            raise InvalidOperationError("Invalid operation: 402 Insufficient Credits")
        elif response.status_code == 403:
            raise APIKeyVerificationError("Invalid API key: 403 Forbidden")
        
        # Raise for other HTTP errors
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_text = response.text if response.text else str(e)
            raise InvalidOperationError(f"Request to Lucidic AI Backend failed: {error_text}")
        
        return response.json()
    
    def verify_api_key(self) -> Dict[str, str]:
        """Verify API key is valid
        
        Returns:
            Dict with project info
            
        Raises:
            APIKeyVerificationError: If API key is invalid
        """
        return self.get("verifyapikey")