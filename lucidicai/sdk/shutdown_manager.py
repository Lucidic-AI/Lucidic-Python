"""Shutdown manager for graceful cleanup.

Coordinates shutdown across all active sessions, ensuring proper cleanup
on process exit. Inspired by TypeScript SDK's shutdown-manager.ts.
"""
import atexit
import logging
import signal
import sys
import threading
import time
from typing import Dict, Optional, Set, Callable
from dataclasses import dataclass

logger = logging.getLogger("Lucidic")


@dataclass
class SessionState:
    """State information for an active session."""
    session_id: str
    http_client: Optional[object] = None
    event_queue: Optional[object] = None
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
        
        logger.debug("[ShutdownManager] Initialized")
    
    def register_session(self, session_id: str, state: SessionState) -> None:
        """Register a new active session.
        
        Args:
            session_id: Session identifier
            state: Session state information
        """
        with self._session_lock:
            logger.debug(f"[ShutdownManager] Registering session {session_id}")
            self.active_sessions[session_id] = state
            
            # ensure listeners are registered
            self._ensure_listeners_registered()
    
    def unregister_session(self, session_id: str) -> None:
        """Unregister a session after it ends.
        
        Args:
            session_id: Session identifier
        """
        with self._session_lock:
            logger.debug(f"[ShutdownManager] Unregistering session {session_id}")
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
    
    def _ensure_listeners_registered(self) -> None:
        """Register process exit listeners once."""
        if self.listeners_registered:
            return
            
        self.listeners_registered = True
        logger.debug("[ShutdownManager] Registering global shutdown listeners")
        
        # register atexit handler for normal termination
        atexit.register(self._handle_exit)
        
        # register signal handlers for interrupts
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # register uncaught exception handler
        sys.excepthook = self._exception_handler
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"[ShutdownManager] Received signal {signum}")
        self._handle_shutdown(f"signal_{signum}")
        # exit after cleanup
        sys.exit(0)
    
    def _exception_handler(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        # log the exception
        logger.error(f"[ShutdownManager] Uncaught exception: {exc_type.__name__}: {exc_value}")
        
        # perform shutdown
        self._handle_shutdown("uncaught_exception")
        
        # call default handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    def _handle_exit(self):
        """Handle normal process exit."""
        logger.debug("[ShutdownManager] Normal exit triggered")
        self._handle_shutdown("atexit")
    
    def _handle_shutdown(self, trigger: str) -> None:
        """Coordinate shutdown of all sessions.
        
        Args:
            trigger: What triggered the shutdown
        """
        if self.is_shutting_down:
            logger.debug(f"[ShutdownManager] Already shutting down, ignoring {trigger}")
            return
        
        self.is_shutting_down = True
        
        with self._session_lock:
            session_count = len(self.active_sessions)
            if session_count == 0:
                logger.debug("[ShutdownManager] No active sessions to clean up")
                self.shutdown_complete.set()
                return
                
            logger.info(f"[ShutdownManager] Shutdown initiated by {trigger}, ending {session_count} active sessions")
            
            # perform shutdown in separate thread to avoid deadlocks
            import threading
            shutdown_thread = threading.Thread(
                target=self._perform_shutdown,
                name="ShutdownThread"
            )
            shutdown_thread.daemon = True
            shutdown_thread.start()
            
            # wait for shutdown with timeout
            if not self.shutdown_complete.wait(timeout=10):
                logger.warning("[ShutdownManager] Shutdown timeout after 10s")
    
    def _perform_shutdown(self) -> None:
        """Perform the actual shutdown of all sessions."""
        try:
            sessions_to_end = []
            
            with self._session_lock:
                # collect sessions that need ending
                for session_id, state in self.active_sessions.items():
                    if state.auto_end and not state.is_shutting_down:
                        state.is_shutting_down = True
                        sessions_to_end.append((session_id, state))
            
            # end all sessions
            for session_id, state in sessions_to_end:
                try:
                    logger.info(f"[ShutdownManager] Ending session {session_id}")
                    self._end_session(session_id, state)
                except Exception as e:
                    logger.debug(f"[ShutdownManager] Error ending session {session_id}: {e}")
            
            logger.info("[ShutdownManager] Shutdown complete")
            
        finally:
            self.shutdown_complete.set()
    
    def _end_session(self, session_id: str, state: SessionState) -> None:
        """End a single session with cleanup.
        
        Args:
            session_id: Session identifier
            state: Session state
        """
        # flush event queue if present
        if state.event_queue:
            try:
                logger.debug(f"[ShutdownManager] Flushing events for session {session_id}")
                # call flush method if it exists
                if hasattr(state.event_queue, 'flush'):
                    state.event_queue.flush(timeout_seconds=5.0)
                elif hasattr(state.event_queue, 'force_flush'):
                    state.event_queue.force_flush()
            except Exception as e:
                logger.debug(f"[ShutdownManager] Error flushing events: {e}")
        
        # end session via API if http client present
        if state.http_client and session_id:
            try:
                logger.debug(f"[ShutdownManager] Ending session {session_id} via API")
                # use the client's session resource
                if hasattr(state.http_client, 'sessions'):
                    state.http_client.sessions.end_session(
                        session_id,
                        is_successful=False,
                        is_successful_reason="Process shutdown"
                    )
            except Exception as e:
                logger.debug(f"[ShutdownManager] Error ending session via API: {e}")
        
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