"""Pure HTTP client for Lucidic API communication.

This module contains only the HTTP client logic using httpx,
supporting both synchronous and asynchronous operations.
"""
import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from ..core.config import SDKConfig, get_config
from ..core.errors import exception_from_response
from ..utils.logger import debug, info, warning, error, mask_sensitive, truncate_data


# Status codes the transport retries with backoff (LUC-901). Connection-level
# failures are retried separately by the httpx transport's own ``retries=``.
# 429 (throttled) and 503 (unavailable / workflow couldn't start) are retried.
_RETRY_STATUSES = frozenset({429, 503})

# Retries are gated to idempotent methods. A POST/PATCH may have committed a
# write before the backend returned 429/503 — the EvoSim kickoff, for example,
# commits the run row and only then 503s if Temporal can't start — so a blind
# retry would duplicate it. GET/HEAD/PUT/DELETE are safe to replay. Trigger
# endpoints that want at-least-once retry need a server-side idempotency key
# first (LUC-808); until then POST/PATCH fail fast, as they did pre-C0.
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS"})

# Hard ceiling on a single retry sleep, so a large (or proxy-injected)
# ``Retry-After`` can't pin the calling thread for minutes/hours.
_MAX_RETRY_DELAY_SECONDS = 30.0


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse a ``Retry-After: <seconds>`` header. The HTTP-date form is not
    supported (returns None → caller falls back to computed backoff)."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _retry_delay(response: httpx.Response, attempt: int, backoff: float) -> float:
    """Seconds to wait before the next attempt: honor ``Retry-After`` when the
    server sends it, else exponential backoff (``backoff * 2**attempt``).

    Clamped to ``_MAX_RETRY_DELAY_SECONDS`` so a huge or proxy-injected
    ``Retry-After`` can't block the caller for minutes/hours.
    """
    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
    delay = retry_after if retry_after is not None else backoff * (2 ** attempt)
    return min(delay, _MAX_RETRY_DELAY_SECONDS)


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
    
    def _raise_if_error(self, response: httpx.Response) -> None:
        """Decode a non-2xx response and raise a typed ``LucidicError`` (LUC-900).

        The single decode point: the backend ``{"error": ...}`` /
        ``{"detail": ...}`` / mock-call ``{"error": {"code"}}`` envelopes all
        map to typed exceptions via ``exception_from_response``. No-op on 2xx.
        """
        if response.is_success:
            return
        text = response.text
        try:
            body = response.json()
        except Exception:
            body = None
        retry_after = _parse_retry_after(response.headers.get("Retry-After"))
        exc = exception_from_response(
            response.status_code, body, text, retry_after=retry_after
        )
        error(f"[HTTP] Error {response.status_code}: {exc}")
        raise exc

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Raise a typed error on non-2xx, else parse and return the JSON body.

        Raises:
            LucidicError subclass: on any non-2xx response (see ``core.errors``).
        """
        self._raise_if_error(response)

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

    def _send_with_retry(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Response:
        """Send one request, retrying transient 429/503 on idempotent methods (LUC-901).

        Connection-level failures are retried by the httpx transport's own
        ``retries=``; this loop adds status-based retries for 429/503, but only
        for idempotent methods (GET/HEAD/PUT/DELETE) — a POST/PATCH may have
        committed a write before the error, so retrying could duplicate it.
        Delays honor ``Retry-After`` (capped). Returns the final response (which
        may still be an error — ``_handle_response`` types it).
        """
        max_retries = self.config.network.max_retries
        backoff = self.config.network.backoff_factor
        attempt = 0
        while True:
            response = self.sync_client.request(
                method=method, url=url, params=params, json=json, **kwargs
            )
            retryable = (
                method.upper() in _IDEMPOTENT_METHODS
                and response.status_code in _RETRY_STATUSES
            )
            if retryable and attempt < max_retries:
                delay = _retry_delay(response, attempt, backoff)
                warning(
                    f"[HTTP] {response.status_code} on {method} {url}; "
                    f"retry {attempt + 1}/{max_retries} in {delay:.2f}s"
                )
                time.sleep(delay)
                attempt += 1
                continue
            return response

    async def _asend_with_retry(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Response:
        """Async sibling of ``_send_with_retry``."""
        max_retries = self.config.network.max_retries
        backoff = self.config.network.backoff_factor
        attempt = 0
        while True:
            response = await self.async_client.request(
                method=method, url=url, params=params, json=json, **kwargs
            )
            retryable = (
                method.upper() in _IDEMPOTENT_METHODS
                and response.status_code in _RETRY_STATUSES
            )
            if retryable and attempt < max_retries:
                delay = _retry_delay(response, attempt, backoff)
                warning(
                    f"[HTTP] {response.status_code} on {method} {url}; "
                    f"retry {attempt + 1}/{max_retries} in {delay:.2f}s"
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue
            return response
    
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
    
    def patch(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a synchronous PATCH request.

        Args:
            endpoint: API endpoint (without base URL)
            data: Request body data

        Returns:
            Response data as dictionary
        """
        data = self._add_timestamp(data)
        return self.request("PATCH", endpoint, json=data)

    def delete(self, endpoint: str, params: Optional[Dict[str, Any]] = None,
               data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a synchronous DELETE request.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            data: Optional JSON body. Most deletes carry none; a few endpoints
                take a small flag in the DELETE body (e.g. the experiments
                delete's ``delete_sessions``).

        Returns:
            Response data as dictionary
        """
        return self.request("DELETE", endpoint, params=params, json=data)

    def head(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> httpx.Headers:
        """Make a synchronous HEAD request and return the response headers (LUC-903).

        List endpoints answer HEAD with counts in headers (e.g. ``X-Total-Count``,
        and sessions' ``X-Tags``) so callers can get a cheap count/tag summary
        without paging. Raises a typed error on non-2xx.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters (same filters as the matching ``.list()``)

        Returns:
            The response headers (case-insensitive ``httpx.Headers``).
        """
        url = f"/{endpoint}"
        debug(f"[HTTP] HEAD {self.base_url}{url}")
        response = self._send_with_retry("HEAD", url, params=params)
        self._raise_if_error(response)
        return response.headers

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
        
        response = self._send_with_retry(method, url, params=params, json=json, **kwargs)

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
    
    async def apatch(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an asynchronous PATCH request.

        Args:
            endpoint: API endpoint (without base URL)
            data: Request body data

        Returns:
            Response data as dictionary
        """
        data = self._add_timestamp(data)
        return await self.arequest("PATCH", endpoint, json=data)

    async def adelete(self, endpoint: str, params: Optional[Dict[str, Any]] = None,
                      data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an asynchronous DELETE request.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            data: Optional JSON body (see ``delete``).

        Returns:
            Response data as dictionary
        """
        return await self.arequest("DELETE", endpoint, params=params, json=data)

    async def ahead(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> httpx.Headers:
        """Async sibling of ``head`` (LUC-903)."""
        url = f"/{endpoint}"
        debug(f"[HTTP] HEAD {self.base_url}{url}")
        response = await self._asend_with_retry("HEAD", url, params=params)
        self._raise_if_error(response)
        return response.headers

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
        
        response = await self._asend_with_retry(method, url, params=params, json=json, **kwargs)

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
