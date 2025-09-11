"""Pure HTTP client for Lucidic API communication.

This module contains only the HTTP client logic, separated from
session management and other concerns.
"""
import json
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from ..core.config import SDKConfig, get_config
from ..core.errors import APIKeyVerificationError
from ..utils.logger import debug, info, warning, error, mask_sensitive, truncate_data


class HttpClient:
    """HTTP client for API communication."""
    
    def __init__(self, config: Optional[SDKConfig] = None):
        """Initialize the HTTP client.
        
        Args:
            config: SDK configuration (uses global if not provided)
        """
        self.config = config or get_config()
        self.base_url = self.config.network.base_url
        
        # Create session with connection pooling
        self.session = requests.Session()
        
        # Configure retries
        retry_cfg = Retry(
            total=self.config.network.max_retries,
            backoff_factor=self.config.network.backoff_factor,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        
        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_cfg,
            pool_connections=self.config.network.connection_pool_size,
            pool_maxsize=self.config.network.connection_pool_maxsize
        )
        
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Set headers
        self._update_headers()
        
        # Verify API key if configured
        if self.config.api_key:
            self._verify_api_key()
    
    def _update_headers(self) -> None:
        """Update session headers with authentication."""
        headers = {
            "User-Agent": "lucidic-sdk/2.0",
            "Content-Type": "application/json"
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Api-Key {self.config.api_key}"
        
        if self.config.agent_id:
            headers["x-agent-id"] = self.config.agent_id
        
        self.session.headers.update(headers)
    
    def _verify_api_key(self) -> None:
        """Verify the API key with the backend."""
        debug("[HTTP] Verifying API key")
        try:
            response = self.get("verifyapikey")
            # Backend returns 200 OK for valid key, check if we got a response
            if response is None:
                raise APIKeyVerificationError("No response from API")
            info("[HTTP] API key verified successfully")
        except APIKeyVerificationError:
            raise
        except requests.RequestException as e:
            raise APIKeyVerificationError(f"Could not verify API key: {e}")
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            Response data as dictionary
        """
        return self.request("GET", endpoint, params=params)
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a POST request.
        
        Args:
            endpoint: API endpoint (without base URL)
            data: Request body data
            
        Returns:
            Response data as dictionary
        """
        # Add current_time to all POST requests like TypeScript SDK does
        from datetime import datetime, timezone
        if data is None:
            data = {}
        data["current_time"] = datetime.now(timezone.utc).isoformat()
        return self.request("POST", endpoint, json=data)
    
    def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a PUT request.
        
        Args:
            endpoint: API endpoint (without base URL)
            data: Request body data
            
        Returns:
            Response data as dictionary
        """
        # Add current_time to all PUT requests like TypeScript SDK does
        from datetime import datetime, timezone
        if data is None:
            data = {}
        data["current_time"] = datetime.now(timezone.utc).isoformat()
        return self.request("PUT", endpoint, json=data)
    
    def delete(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a DELETE request.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            Response data as dictionary
        """
        return self.request("DELETE", endpoint, params=params)
    
    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an HTTP request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint (without base URL)
            params: Query parameters
            json: Request body (for POST/PUT)
            **kwargs: Additional arguments for requests
            
        Returns:
            Response data as dictionary
            
        Raises:
            requests.RequestException: On HTTP errors
        """
        url = f"{self.base_url}/{endpoint}"
        
        # Log request details
        debug(f"[HTTP] {method} {url}")
        if params:
            debug(f"[HTTP] Query params: {mask_sensitive(params)}")
        if json:
            debug(f"[HTTP] Request body: {truncate_data(mask_sensitive(json))}")
        
        response = self.session.request(
            method=method,
            url=url,
            params=params,
            json=json,
            timeout=self.config.network.timeout,
            **kwargs
        )
        
        # Raise for HTTP errors with more detail
        if not response.ok:
            # Try to get error details from response
            try:
                error_data = response.json()
                error_msg = error_data.get('detail', response.text)
            except:
                error_msg = response.text
            
            error(f"[HTTP] Error {response.status_code}: {error_msg}")
        
        response.raise_for_status()
        
        # Parse JSON response
        try:
            data = response.json()
        except ValueError:
            # For empty responses (like verifyapikey), return success
            if response.status_code == 200 and not response.text:
                data = {"success": True}
            else:
                # Return text if not JSON
                data = {"response": response.text}
        
        debug(f"[HTTP] Response ({response.status_code}): {truncate_data(data)}")
        
        return data
    
    def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            self.session.close()