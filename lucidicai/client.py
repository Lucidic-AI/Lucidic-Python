import os
import time
import threading
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

import requests
import logging
import json
from requests.adapters import HTTPAdapter, Retry
from urllib3.util import Retry


from .errors import APIKeyVerificationError, InvalidOperationError, LucidicNotInitializedError
from .session import Session
from .singleton import singleton, clear_singletons
from .lru import LRUCache
from .event import Event
from .event_queue import EventQueue
import uuid

NETWORK_RETRIES = 3

logger = logging.getLogger("Lucidic")


@singleton
class Client:
    def __init__(
        self,
        api_key: str,
        agent_id: str,
    ):
        self.base_url = "https://backend.lucidic.ai/api" if not (os.getenv("LUCIDIC_DEBUG", 'False') == 'True') else "http://localhost:8000/api"
        self.initialized = False
        self.session = None
        self.previous_sessions = LRUCache(500)  # For LRU cache of previously initialized sessions
        self.custom_session_id_translations = LRUCache(500) # For translations of custom session IDs to real session IDs
        self.api_key = api_key
        self.agent_id = agent_id
        self.masking_function = None
        self.auto_end = False  # Default to False until explicitly set during init
        self._shutdown = False  # Flag to prevent requests after shutdown
        self.request_session = requests.Session()
        retry_cfg = Retry(
            total=3,                     # 3 attempts in total
            backoff_factor=0.5,          # exponential back-off: 0.5s, 1s, 2s …
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retry_cfg, pool_connections=20, pool_maxsize=100)
        self.request_session.mount("https://", adapter)
        self.set_api_key(api_key)
        self.prompts = dict()
        # Initialize event queue (non-blocking event delivery)
        self._event_queue = EventQueue(self)
        
        # Track telemetry state to prevent re-initialization
        # These are process-wide singletons for telemetry
        self._telemetry_lock = threading.Lock()  # Prevent race conditions
        self._tracer_provider = None
        self._instrumentors = {}  # Dict to track which providers are instrumented
        self._telemetry_initialized = False
        
        # Track active sessions to prevent premature EventQueue shutdown
        self._active_sessions_lock = threading.Lock()
        self._active_sessions = set()  # Set of active session IDs

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        self.request_session.headers.update({"Authorization": f"Api-Key {self.api_key}", "User-Agent": "lucidic-sdk/1.1"})
        try:
            self.verify_api_key(self.base_url, api_key)
        except APIKeyVerificationError:
            raise APIKeyVerificationError("Invalid API Key")

    def clear(self):
        # Clean up singleton state
        clear_singletons()
        self.initialized = False
        self.session = None
        del self

    def verify_api_key(self, base_url: str, api_key: str) -> Tuple[str, str]:
        data = self.make_request('verifyapikey', 'GET', {})  # TODO: Verify against agent ID provided
        return data["project"], data["project_id"]

    def set_provider(self, provider) -> None:
        """Deprecated: manual provider overrides removed (no-op)."""
        return

    def init_session(
        self,
        session_name: str,
        task: Optional[str] = None,
        rubrics: Optional[list] = None,
        tags: Optional[list] = None,
        production_monitoring: Optional[bool] = False,
        session_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        dataset_item_id: Optional[str] = None,
    ) -> None:
        if session_id:
            # Check if it's a known session ID, maybe custom and maybe real
            if session_id in self.custom_session_id_translations:
                session_id = self.custom_session_id_translations[session_id]
            # Check if it's the same as the current session
            if self.session and self.session.session_id == session_id:
                return self.session.session_id
            # Check if it's a previous session that we have saved
            if session_id in self.previous_sessions:
                if self.session:
                    self.previous_sessions[self.session.session_id] = self.session
                self.session = self.previous_sessions.pop(session_id)  # Remove from previous sessions because it's now the current session
                return self.session.session_id

        # Either there's no session ID, or we don't know about the old session
        # We need to go to the backend in both cases
        request_data = {
            "agent_id": self.agent_id,
            "session_name": session_name,
            "task": task,
            "experiment_id": experiment_id,
            "rubrics": rubrics,
            "tags": tags,
            "session_id": session_id,
            "dataset_item_id": dataset_item_id,
            "production_monitoring": production_monitoring
        }
        data = self.make_request('initsession', 'POST', request_data)
        real_session_id = data["session_id"]
        if session_id and session_id != real_session_id:
            self.custom_session_id_translations[session_id] = real_session_id
        
        if self.session:
            self.previous_sessions[self.session.session_id] = self.session

        self.session = Session(
            agent_id=self.agent_id,
            session_id=real_session_id,
            session_name=session_name,
            experiment_id=experiment_id,
            task=task,
            rubrics=rubrics,
            tags=tags,
        )
        
        # Track this as an active session
        with self._active_sessions_lock:
            self._active_sessions.add(real_session_id)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Client] Added active session {real_session_id[:8]}..., total: {len(self._active_sessions)}")
        
        self.initialized = True
        return self.session.session_id

    def mark_session_inactive(self, session_id: str) -> None:
        """Mark a session as inactive. Used when ending a session."""
        with self._active_sessions_lock:
            if session_id in self._active_sessions:
                self._active_sessions.discard(session_id)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[Client] Removed active session {session_id[:8]}..., remaining: {len(self._active_sessions)}")
    
    def has_active_sessions(self) -> bool:
        """Check if there are any active sessions."""
        with self._active_sessions_lock:
            return len(self._active_sessions) > 0
    
    def create_event_for_session(self, session_id: str, **kwargs) -> str:
        """Create an event for a specific session id (new typed model).

        This avoids mutating the global session and directly uses the new
        event API. Prefer passing typed fields and a 'type' argument.
        """
        kwargs = dict(kwargs)
        kwargs['session_id'] = session_id
        return self.create_event(**kwargs)

    def create_experiment(self, **kwargs) -> str:
        kwargs['agent_id'] = self.agent_id
        return self.make_request('createexperiment', 'POST', kwargs)['experiment_id']

    def get_prompt(self, prompt_name, cache_ttl, label) -> str:
        current_time = time.time()
        key = (prompt_name, label)
        if key in self.prompts:
            prompt, expiration_time = self.prompts[key]
            if expiration_time == float('inf') or current_time < expiration_time:
                return prompt
        params={
            "agent_id": self.agent_id,
            "prompt_name": prompt_name,
            "label": label
        }
        prompt = self.make_request('getprompt', 'GET', params)['prompt_content']
        
        if cache_ttl != 0:
            if cache_ttl == -1:
                expiration_time = float('inf')
            else:
                expiration_time = current_time + cache_ttl
            self.prompts[key] = (prompt, expiration_time)
        return prompt

    def make_request(self, endpoint, method, data):
        # Check if client is shutting down
        if self._shutdown:
            logger.warning(f"[HTTP] Attempted request after shutdown: {endpoint}")
            return {}

        data = {k: v for k, v in data.items() if v is not None}

        http_methods = {
            "GET": lambda data: self.request_session.get(f"{self.base_url}/{endpoint}", params=data),
            "POST": lambda data: self.request_session.post(f"{self.base_url}/{endpoint}", json=data),
            "PUT": lambda data: self.request_session.put(f"{self.base_url}/{endpoint}", json=data),
            "DELETE": lambda data: self.request_session.delete(f"{self.base_url}/{endpoint}", params=data),
        }  # TODO: make into enum
        data['current_time'] = datetime.now().astimezone(timezone.utc).isoformat()
        # Debug: print final payload about to be sent
        try:
            dbg = json.dumps({"endpoint": endpoint, "method": method, "body": data}, ensure_ascii=False)
            logger.debug(f"[HTTP] Sending request: {dbg}")
        except Exception:
            logger.debug(f"[HTTP] Sending request to {endpoint} {method}")
        func = http_methods[method]
        response = None
        for _ in range(NETWORK_RETRIES):
            try:
                response = func(data)
                break
            except Exception:
                pass
        if response is None:
            raise InvalidOperationError("Cannot reach backend. Check your internet connection.")
        if response.status_code == 401:
            raise APIKeyVerificationError("Invalid API key: 401 Unauthorized")
        if response.status_code == 402:
            raise InvalidOperationError("Invalid operation: 402 Insufficient Credits")
        if response.status_code == 403:
            raise APIKeyVerificationError(f"Invalid API key: 403 Forbidden")
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise InvalidOperationError(f"Request to Lucidic AI Backend failed: {e.response.text}")
        return response.json()

    # ==== New Typed Event Model Helpers ====
    def _build_payload(self, type: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Build type-specific payload and place unrecognized keys in misc."""
        # Remove non-payload top-level fields from kwargs copy
        non_payload_fields = [
            'parent_event_id', 'tags', 'metadata', 'occurred_at', 'duration', 'session_id',
            'event_id'
        ]
        for field in non_payload_fields:
            if field in kwargs:
                kwargs.pop(field, None)

        if type == "llm_generation":
            return self._build_llm_payload(kwargs)
        elif type == "function_call":
            return self._build_function_payload(kwargs)
        elif type == "error_traceback":
            return self._build_error_payload(kwargs)
        else:
            return self._build_generic_payload(kwargs)

    def _build_llm_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "request": {},
            "response": {},
            "usage": {},
            "status": "ok",
            "misc": {}
        }
        # Request fields
        for field in ["provider", "model", "messages", "params"]:
            if field in kwargs:
                payload["request"][field] = kwargs.pop(field)
        # Response fields
        for field in ["output", "messages", "tool_calls", "thinking", "raw"]:
            if field in kwargs:
                payload["response"][field] = kwargs.pop(field)
        # Usage fields
        for field in ["input_tokens", "output_tokens", "cache", "cost"]:
            if field in kwargs:
                payload["usage"][field] = kwargs.pop(field)
        # Status / error
        if 'status' in kwargs:
            payload['status'] = kwargs.pop('status')
        if 'error' in kwargs:
            payload['error'] = kwargs.pop('error')
        payload["misc"] = kwargs
        return payload

    def _build_function_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "function_name": kwargs.pop("function_name", "unknown"),
            "arguments": kwargs.pop("arguments", {}),
            "return_value": kwargs.pop("return_value", None),
            "misc": kwargs
        }
        return payload

    def _build_error_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "error": kwargs.pop("error", ""),
            "traceback": kwargs.pop("traceback", ""),
            "misc": kwargs
        }
        return payload

    def _build_generic_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "details": kwargs.pop("details", kwargs.pop("description", "")),
            "misc": kwargs
        }
        return payload

    def create_event(self, type: str = "generic", **kwargs) -> str:
        """Create a typed event (non-blocking) and return client-side UUID.

        - Generates and returns client_event_id immediately
        - Enqueues the full event for background processing via EventQueue
        - Supports parent nesting via client-side parent_event_id
        - Handles client-side blob thresholding in the queue
        """
        # Resolve session_id: explicit -> context -> current session
        session_id = kwargs.pop('session_id', None)
        if not session_id:
            try:
                from .context import current_session_id
                session_id = current_session_id.get(None)
            except Exception:
                session_id = None
        if not session_id and self.session:
            session_id = self.session.session_id
        if not session_id:
            raise InvalidOperationError("No active session for event creation")

        # Parent event id from kwargs or parent context (client-side)
        parent_event_id = kwargs.get('parent_event_id')
        if not parent_event_id:
            try:
                from .context import current_parent_event_id
                parent_event_id = current_parent_event_id.get(None)
            except Exception:
                parent_event_id = None

        # Build payload (typed)
        payload = self._build_payload(type, dict(kwargs))

        # Occurred-at
        from datetime import datetime as _dt
        _occ = kwargs.get("occurred_at")
        if isinstance(_occ, str):
            occurred_at_str = _occ
        elif isinstance(_occ, _dt):
            if _occ.tzinfo is None:
                local_tz = _dt.now().astimezone().tzinfo
                occurred_at_str = _occ.replace(tzinfo=local_tz).isoformat()
            else:
                occurred_at_str = _occ.isoformat()
        else:
            occurred_at_str = _dt.now().astimezone().isoformat()

        # Client-side UUIDs
        client_event_id = kwargs.get('event_id') or str(uuid.uuid4())

        # Build request body with client ids
        event_request: Dict[str, Any] = {
            "session_id": session_id,
            "client_event_id": client_event_id,
            "client_parent_event_id": parent_event_id,
            "type": type,
            "occurred_at": occurred_at_str,
            "duration": kwargs.get("duration"),
            "tags": kwargs.get("tags", []),
            "metadata": kwargs.get("metadata", {}),
            "payload": payload,
        }

        # Queue for background processing and return immediately
        self._event_queue.queue_event(event_request)
        return client_event_id

    def update_event(self, event_id: str, type: Optional[str] = None, **kwargs) -> str:
        """Deprecated: events are immutable in the new model."""
        raise InvalidOperationError("update_event is no longer supported. Events are immutable.")

    def mask(self, data):
        if not self.masking_function:
            return data
        if not data:
            return data
        try:
            return self.masking_function(data)
        except Exception as e:
            logger = logging.getLogger('Lucidic')
            logger.error(f"Error in custom masking function: {repr(e)}")
            return "<Error in custom masking function, this is a fully-masked placeholder>"
    
    def initialize_telemetry(self, providers: list) -> bool:
        """
        Initialize telemetry with the given providers.
        This is a true singleton - only the first call creates the TracerProvider.
        Subsequent calls only add new instrumentors if needed.
        
        Args:
            providers: List of provider names to instrument
            
        Returns:
            True if telemetry was successfully initialized or already initialized
        """
        with self._telemetry_lock:
            try:
                # Create TracerProvider only once per process
                if self._tracer_provider is None:
                    logger.debug("[Telemetry] Creating TracerProvider (first initialization)")
                    
                    from opentelemetry import trace
                    from opentelemetry.sdk.trace import TracerProvider
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor
                    from opentelemetry.sdk.resources import Resource
                    
                    resource = Resource.create({
                        "service.name": "lucidic-ai",
                        "service.version": "1.0.0",
                        "lucidic.agent_id": self.agent_id,
                    })
                    
                    # Create provider with shutdown_on_exit=False for our control
                    self._tracer_provider = TracerProvider(resource=resource, shutdown_on_exit=False)
                    
                    # Add context capture processor FIRST
                    from .telemetry.context_capture_processor import ContextCaptureProcessor
                    context_processor = ContextCaptureProcessor()
                    self._tracer_provider.add_span_processor(context_processor)
                    
                    # Add exporter processor for sending spans to Lucidic
                    from .telemetry.lucidic_exporter import LucidicSpanExporter
                    exporter = LucidicSpanExporter()
                    # Configure for faster export: 100ms interval instead of default 5000ms
                    # This matches the TypeScript SDK's flush interval pattern
                    export_processor = BatchSpanProcessor(
                        exporter,
                        schedule_delay_millis=100,  # Export every 100ms
                        max_export_batch_size=512,  # Reasonable batch size
                        max_queue_size=2048         # Larger queue for burst handling
                    )
                    self._tracer_provider.add_span_processor(export_processor)
                    
                    # Set as global provider (only happens once)
                    try:
                        trace.set_tracer_provider(self._tracer_provider)
                        logger.debug("[Telemetry] Set global TracerProvider")
                    except Exception as e:
                        # This is OK - might already be set
                        logger.debug(f"[Telemetry] Global provider already set: {e}")
                    
                    self._telemetry_initialized = True
                
                # Now instrument the requested providers (can happen multiple times)
                if providers:
                    from .telemetry.telemetry_init import instrument_providers
                    new_instrumentors = instrument_providers(providers, self._tracer_provider, self._instrumentors)
                    # Update our tracking dict
                    self._instrumentors.update(new_instrumentors)
                    logger.debug(f"[Telemetry] Instrumented providers: {list(new_instrumentors.keys())}")
                
                return True
                
            except Exception as e:
                logger.error(f"[Telemetry] Failed to initialize: {e}")
                return False
    
    def flush_telemetry(self, timeout_seconds: float = 2.0) -> bool:
        """
        Flush all OpenTelemetry spans to ensure they're exported.
        
        This method blocks until all buffered spans in the TracerProvider
        are exported or the timeout is reached. Critical for ensuring
        LLM generation events are not lost during shutdown.
        
        Handles both active and shutdown TracerProviders gracefully.
        
        Args:
            timeout_seconds: Maximum time to wait for flush completion
            
        Returns:
            True if flush succeeded, False if timeout occurred
        """
        try:
            if self._tracer_provider:
                # Check if provider is already shutdown
                if hasattr(self._tracer_provider, '_shutdown') and self._tracer_provider._shutdown:
                    logger.debug("[Telemetry] TracerProvider already shutdown, skipping flush")
                    return True
                    
                # Convert seconds to milliseconds for OpenTelemetry
                timeout_millis = int(timeout_seconds * 1000)
                success = self._tracer_provider.force_flush(timeout_millis)
                if success:
                    logger.debug(f"[Telemetry] Successfully flushed spans (timeout={timeout_seconds}s)")
                else:
                    logger.warning(f"[Telemetry] Flush timed out after {timeout_seconds}s")
                return success
            return True  # No provider = nothing to flush = success
        except Exception as e:
            logger.error(f"[Telemetry] Failed to flush spans: {e}")
            return False