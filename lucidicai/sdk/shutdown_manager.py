"""Shutdown manager for graceful cleanup.

Coordinates shutdown across all active sessions, ensuring proper cleanup
on process exit. Inspired by TypeScript SDK's shutdown-manager.ts.
"""
import atexit
import signal
import sys
import threading
import time
from typing import Dict, Optional, Set, Callable, TYPE_CHECKING
from dataclasses import dataclass

from ..utils.logger import debug, info, warning, error, truncate_id

if TYPE_CHECKING:
    from ..client import LucidicAI


@dataclass
class SessionState:
    """State information for an active session."""
    session_id: str
    http_client: Optional[object] = None
    is_shutting_down: bool = False
    auto_end: bool = True


class ShutdownManager:
    """Singleton manager for coordinating shutdown across all active sessions.
    
    Ensures process listeners are only registered once and all sessions
    are properly ended on exit.
    """
    
    _instance: Optional['ShutdownManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # only initialize once
        if self._initialized:
            return

        self._initialized = True
        self.active_sessions: Dict[str, SessionState] = {}
        self.is_shutting_down = False
        self.shutdown_complete = threading.Event()
        self.listeners_registered = False
        self._session_lock = threading.Lock()

        # Client registry for multi-client support
        self._clients: Dict[str, "LucidicAI"] = {}
        self._client_lock = threading.Lock()

        debug("[ShutdownManager] Initialized")
    
    def register_session(self, session_id: str, state: SessionState) -> None:
        """Register a new active session.
        
        Args:
            session_id: Session identifier
            state: Session state information
        """
        with self._session_lock:
            debug(f"[ShutdownManager] Registering session {truncate_id(session_id)}, auto_end={state.auto_end}")
            self.active_sessions[session_id] = state
            
            # ensure listeners are registered
            self._ensure_listeners_registered()
    
    def unregister_session(self, session_id: str) -> None:
        """Unregister a session after it ends.
        
        Args:
            session_id: Session identifier
        """
        with self._session_lock:
            debug(f"[ShutdownManager] Unregistering session {truncate_id(session_id)}")
            self.active_sessions.pop(session_id, None)
    
    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        with self._session_lock:
            return len(self.active_sessions)
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if a session is active.

        Args:
            session_id: Session identifier

        Returns:
            True if session is active
        """
        with self._session_lock:
            return session_id in self.active_sessions

    def register_client(self, client: "LucidicAI") -> None:
        """Register a client for shutdown tracking.

        Args:
            client: The LucidicAI client to register
        """
        with self._client_lock:
            self._clients[client._client_id] = client
            debug(f"[ShutdownManager] Registered client {client._client_id[:8]}...")

            # Ensure listeners are registered
            self._ensure_listeners_registered()

    def unregister_client(self, client_id: str) -> None:
        """Unregister a client.

        Args:
            client_id: The client ID to unregister
        """
        with self._client_lock:
            self._clients.pop(client_id, None)
            debug(f"[ShutdownManager] Unregistered client {client_id[:8]}...")

    def _ensure_listeners_registered(self) -> None:
        """Register process exit listeners once."""
        if self.listeners_registered:
            return
            
        self.listeners_registered = True
        debug("[ShutdownManager] Registering global shutdown listeners (atexit, SIGINT, SIGTERM, uncaught exceptions)")
        
        # register atexit handler for normal termination
        atexit.register(self._handle_exit)
        
        # register signal handlers for interrupts
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # register uncaught exception handler
        sys.excepthook = self._exception_handler
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        info(f"[ShutdownManager] Received signal {signum}, initiating graceful shutdown")
        self._handle_shutdown(f"signal_{signum}")
        # exit after cleanup
        sys.exit(0)
    
    def _exception_handler(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        # log the exception
        error(f"[ShutdownManager] Uncaught exception: {exc_type.__name__}: {exc_value}")
        
        # Create an error event for the uncaught exception
        try:
            from ..sdk.event import create_event
            import traceback
            
            error_message = f"{exc_type.__name__}: {exc_value}"
            traceback_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            
            create_event(
                type="error_traceback",
                error=error_message,
                traceback=traceback_str
            )
            debug(f"[ShutdownManager] Created error_traceback event for uncaught exception")
        except Exception as e:
            debug(f"[ShutdownManager] Failed to create error_traceback event: {e}")
        
        # perform shutdown
        self._handle_shutdown("uncaught_exception")
        
        # call default handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    def _handle_exit(self):
        """Handle normal process exit."""
        debug("[ShutdownManager] Normal process exit triggered (atexit)")
        self._handle_shutdown("atexit")
    
    def _handle_shutdown(self, trigger: str) -> None:
        """Coordinate shutdown of all sessions.
        
        Args:
            trigger: What triggered the shutdown
        """
        if self.is_shutting_down:
            debug(f"[ShutdownManager] Already shutting down, ignoring {trigger}")
            return

        self.is_shutting_down = True

        # Check if there's anything to clean up
        with self._session_lock:
            session_count = len(self.active_sessions)
        with self._client_lock:
            client_count = len(self._clients)

        if session_count == 0 and client_count == 0:
            debug("[ShutdownManager] No active sessions or clients to clean up")
            # Reset flag so future shutdowns can proceed (e.g., if exception triggered
            # shutdown before any sessions were created, then user creates sessions)
            self.is_shutting_down = False
            self.shutdown_complete.set()
            return

        info(f"[ShutdownManager] Shutdown initiated by {trigger}, ending {session_count} active session(s), {client_count} client(s)")

        # perform shutdown in separate thread to avoid deadlocks
        import threading
        shutdown_thread = threading.Thread(
            target=self._perform_shutdown,
            name="ShutdownThread"
        )
        shutdown_thread.daemon = True
        shutdown_thread.start()

        # wait for shutdown with timeout - MUST be outside lock to avoid deadlock
        if not self.shutdown_complete.wait(timeout=30):
            warning("[ShutdownManager] Shutdown timeout after 30s")
    
    def _perform_shutdown(self) -> None:
        """Perform the actual shutdown of all sessions and clients."""
        debug("[ShutdownManager] _perform_shutdown thread started")
        try:
            # First, flush all pending background events before ending sessions
            # This ensures telemetry from the exporter is sent
            try:
                debug("[ShutdownManager] Flushing pending events before session cleanup")
                from ..sdk.event import flush
                flush(timeout=5.0)
                debug("[ShutdownManager] Event flush complete")
            except Exception as e:
                error(f"[ShutdownManager] Error flushing events: {e}")

            # Close all registered clients (this will also end their sessions)
            clients_to_close = []
            with self._client_lock:
                clients_to_close = list(self._clients.values())

            if clients_to_close:
                debug(f"[ShutdownManager] Closing {len(clients_to_close)} registered client(s)")
                for client in clients_to_close:
                    try:
                        debug(f"[ShutdownManager] Closing client {client._client_id[:8]}...")
                        client.close()
                    except Exception as e:
                        error(f"[ShutdownManager] Error closing client: {e}")

            # Also handle sessions registered directly (legacy support)
            sessions_to_end = []

            with self._session_lock:
                # collect sessions that need ending
                for session_id, state in self.active_sessions.items():
                    if state.auto_end and not state.is_shutting_down:
                        state.is_shutting_down = True
                        sessions_to_end.append((session_id, state))

            debug(f"[ShutdownManager] Found {len(sessions_to_end)} sessions to end")

            # end all sessions
            for session_id, state in sessions_to_end:
                try:
                    debug(f"[ShutdownManager] Ending session {truncate_id(session_id)}")
                    self._end_session(session_id, state)
                except Exception as e:
                    error(f"[ShutdownManager] Error ending session {truncate_id(session_id)}: {e}")
            
            # Final telemetry shutdown after all sessions are ended
            try:
                from ..sdk.init import get_tracer_provider
                tracer_provider = get_tracer_provider()
                if tracer_provider:
                    debug("[ShutdownManager] Final OpenTelemetry shutdown")
                    try:
                        # Final flush and shutdown with longer timeout
                        tracer_provider.force_flush(timeout_millis=5000)
                        tracer_provider.shutdown()
                        debug("[ShutdownManager] OpenTelemetry shutdown complete")
                    except Exception as e:
                        error(f"[ShutdownManager] Error in final telemetry shutdown: {e}")
            except ImportError:
                pass  # SDK not initialized
            
            info("[ShutdownManager] Shutdown complete")
            
        except Exception as e:
            error(f"[ShutdownManager] Unexpected error in _perform_shutdown: {e}")
            import traceback
            error(f"[ShutdownManager] Traceback: {traceback.format_exc()}")
        finally:
            debug("[ShutdownManager] Setting shutdown_complete event")
            self.shutdown_complete.set()
    
    def _end_session(self, session_id: str, state: SessionState) -> None:
        """End a single session with cleanup.
        
        Args:
            session_id: Session identifier
            state: Session state
        """
        # Flush OpenTelemetry spans first
        try:
            from ..sdk.init import get_tracer_provider
            tracer_provider = get_tracer_provider()
            if tracer_provider:
                debug(f"[ShutdownManager] Flushing OpenTelemetry spans for session {truncate_id(session_id)}")
                try:
                    # Force flush with 3 second timeout
                    tracer_provider.force_flush(timeout_millis=3000)
                except Exception as e:
                    error(f"[ShutdownManager] Error flushing spans: {e}")
        except ImportError:
            pass  # SDK not initialized
        
        # end session via API if http client present
        if state.http_client and session_id:
            try:
                debug(f"[ShutdownManager] Ending session {truncate_id(session_id)} via API")
                debug(f"[ShutdownManager] http_client type: {type(state.http_client)}, keys: {state.http_client.keys() if isinstance(state.http_client, dict) else 'not a dict'}")
                # state.http_client is a resources dict with 'sessions' key
                if isinstance(state.http_client, dict) and 'sessions' in state.http_client:
                    state.http_client['sessions'].end_session(
                        session_id,
                        is_successful=False,
                        session_eval_reason="Process shutdown"
                    )
                    debug(f"[ShutdownManager] Session {truncate_id(session_id)} ended via API")
                else:
                    debug(f"[ShutdownManager] Cannot end session - http_client not properly configured")
            except Exception as e:
                error(f"[ShutdownManager] Error ending session via API: {e}")
        
        # unregister the session
        self.unregister_session(session_id)
    
    def reset(self) -> None:
        """Reset shutdown manager (for testing)."""
        with self._session_lock:
            self.active_sessions.clear()
            self.is_shutting_down = False
            self.shutdown_complete.clear()
            # note: we don't reset listeners_registered as they persist


# global singleton instance
_shutdown_manager = ShutdownManager()


def get_shutdown_manager() -> ShutdownManager:
    """Get the global shutdown manager instance."""
    return _shutdown_manager