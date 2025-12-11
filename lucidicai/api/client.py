"""Pure HTTP client for Lucidic API communication.

This module contains only the HTTP client logic using httpx,
supporting both synchronous and asynchronous operations.
"""
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from ..core.config import SDKConfig, get_config
from ..core.errors import APIKeyVerificationError
from ..utils.logger import debug, info, warning, error, mask_sensitive, truncate_data


class HttpClient:
    """HTTP client for API communication with sync and async support."""
    
    def __init__(self, config: Optional[SDKConfig] = None):
        """Initialize the HTTP client.
        
        Args:
            config: SDK configuration (uses global if not provided)
        """
        self.config = config or get_config()
        self.base_url = self.config.network.base_url
        
        # Build default headers
        self._headers = self._build_headers()
        
        # Transport configuration for connection pooling and retries
        self._transport_kwargs = {
            "retries": self.config.network.max_retries,
        }
        
        # Connection limits for pooling
        self._limits = httpx.Limits(
            max_connections=self.config.network.connection_pool_maxsize,
            max_keepalive_connections=self.config.network.connection_pool_size,
        )
        
        # Lazy-initialized clients
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._async_client_loop: Optional[asyncio.AbstractEventLoop] = None
    
    def _build_headers(self) -> Dict[str, str]:
        """Build default headers for requests."""
        headers = {
            "User-Agent": "lucidic-sdk/2.0",
            "Content-Type": "application/json"
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Api-Key {self.config.api_key}"
        
        if self.config.agent_id:
            headers["x-agent-id"] = self.config.agent_id
        
        return headers
    
    @property
    def sync_client(self) -> httpx.Client:
        """Get or create the synchronous HTTP client."""
        if self._sync_client is None or self._sync_client.is_closed:
            transport = httpx.HTTPTransport(**self._transport_kwargs)
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                headers=self._headers,
                timeout=httpx.Timeout(self.config.network.timeout),
                limits=self._limits,
                transport=transport,
            )
        return self._sync_client
    
    @property
    def async_client(self) -> httpx.AsyncClient:
        """Get or create the asynchronous HTTP client.
        
        The client is recreated if the event loop has changed, since
        httpx.AsyncClient is tied to a specific event loop.
        """
        # Check if we need to recreate the client
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass  # No running loop
        
        # Recreate client if: no client, client closed, or event loop changed
        needs_new_client = (
            self._async_client is None or 
            self._async_client.is_closed or
            (current_loop is not None and self._async_client_loop is not current_loop)
        )
        
        if needs_new_client:
            # Close old client if it exists and isn't already closed
            if self._async_client is not None and not self._async_client.is_closed:
                try:
                    # Can't await in a property, so we just let it be garbage collected
                    pass
                except Exception:
                    pass
            
            transport = httpx.AsyncHTTPTransport(**self._transport_kwargs)
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers,
                timeout=httpx.Timeout(self.config.network.timeout),
                limits=self._limits,
                transport=transport,
            )
            self._async_client_loop = current_loop
            
        return self._async_client
    
    def _add_timestamp(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Add current_time to request data."""
        if data is None:
            data = {}
        data["current_time"] = datetime.now(timezone.utc).isoformat()
        return data
    
    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and parse JSON.
        
        Args:
            response: httpx Response object
            
        Returns:
            Response data as dictionary
            
        Raises:
            APIKeyVerificationError: On 401 Unauthorized responses
            httpx.HTTPStatusError: On other HTTP errors
        """
        # Log and raise for HTTP errors
        if not response.is_success:
            try:
                error_data = response.json()
                error_msg = error_data.get('detail', response.text)
            except Exception:
                error_msg = response.text
            
            error(f"[HTTP] Error {response.status_code}: {error_msg}")
            
            # Raise specific error for authentication/authorization failures
            if response.status_code in (401, 403):
                raise APIKeyVerificationError(f"Authentication failed: {error_msg}")
        
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
    
    # ==================== Synchronous Methods ====================
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a synchronous GET request.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            Response data as dictionary
        """
        return self.request("GET", endpoint, params=params)
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a synchronous POST request.
        
        Args:
            endpoint: API endpoint (without base URL)
            data: Request body data
            
        Returns:
            Response data as dictionary
        """
        data = self._add_timestamp(data)
        return self.request("POST", endpoint, json=data)
    
    def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a synchronous PUT request.
        
        Args:
            endpoint: API endpoint (without base URL)
            data: Request body data
            
        Returns:
            Response data as dictionary
        """
        data = self._add_timestamp(data)
        return self.request("PUT", endpoint, json=data)
    
    def delete(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a synchronous DELETE request.
        
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
        """Make a synchronous HTTP request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint (without base URL)
            params: Query parameters
            json: Request body (for POST/PUT)
            **kwargs: Additional arguments for httpx
            
        Returns:
            Response data as dictionary
            
        Raises:
            httpx.HTTPError: On HTTP errors
        """
        url = f"/{endpoint}"
        
        # Log request details
        debug(f"[HTTP] {method} {self.base_url}{url}")
        if params:
            debug(f"[HTTP] Query params: {mask_sensitive(params)}")
        if json:
            debug(f"[HTTP] Request body: {truncate_data(mask_sensitive(json))}")
        
        response = self.sync_client.request(
            method=method,
            url=url,
            params=params,
            json=json,
            **kwargs
        )
        
        return self._handle_response(response)
    
    # ==================== Asynchronous Methods ====================
    
    async def aget(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an asynchronous GET request.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            Response data as dictionary
        """
        return await self.arequest("GET", endpoint, params=params)
    
    async def apost(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an asynchronous POST request.
        
        Args:
            endpoint: API endpoint (without base URL)
            data: Request body data
            
        Returns:
            Response data as dictionary
        """
        data = self._add_timestamp(data)
        return await self.arequest("POST", endpoint, json=data)
    
    async def aput(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an asynchronous PUT request.
        
        Args:
            endpoint: API endpoint (without base URL)
            data: Request body data
            
        Returns:
            Response data as dictionary
        """
        data = self._add_timestamp(data)
        return await self.arequest("PUT", endpoint, json=data)
    
    async def adelete(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an asynchronous DELETE request.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            Response data as dictionary
        """
        return await self.arequest("DELETE", endpoint, params=params)
    
    async def arequest(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an asynchronous HTTP request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint (without base URL)
            params: Query parameters
            json: Request body (for POST/PUT)
            **kwargs: Additional arguments for httpx
            
        Returns:
            Response data as dictionary
            
        Raises:
            httpx.HTTPError: On HTTP errors
        """
        url = f"/{endpoint}"
        
        # Log request details
        debug(f"[HTTP] {method} {self.base_url}{url}")
        if params:
            debug(f"[HTTP] Query params: {mask_sensitive(params)}")
        if json:
            debug(f"[HTTP] Request body: {truncate_data(mask_sensitive(json))}")
        
        response = await self.async_client.request(
            method=method,
            url=url,
            params=params,
            json=json,
            **kwargs
        )
        
        return self._handle_response(response)
    
    # ==================== Lifecycle Methods ====================
    
    def close(self) -> None:
        """Close the synchronous HTTP client."""
        if self._sync_client is not None and not self._sync_client.is_closed:
            self._sync_client.close()
            self._sync_client = None
    
    async def aclose(self) -> None:
        """Close the asynchronous HTTP client."""
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None
    
    def __enter__(self) -> "HttpClient":
        """Context manager entry for sync client."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit for sync client."""
        self.close()
    
    async def __aenter__(self) -> "HttpClient":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.aclose()
