"""Error suppression and safety utilities for the Lucidic SDK.

This module provides a comprehensive error handling system that prevents
SDK errors from propagating to user code, ensuring SDK failures don't
crash user applications.
"""
import os
import logging
import functools
import asyncio
import time
from typing import Any, Callable, Optional, TypeVar, Union
from collections.abc import Coroutine

logger = logging.getLogger("Lucidic")

# Type variable for preserving function signatures
F = TypeVar('F', bound=Callable[..., Any])


class SDKErrorHandler:
    """Centralized error handler for the Lucidic SDK.
    
    Provides error suppression, logging, and cleanup functionality
    to ensure SDK errors don't affect user code execution.
    """
    
    def __init__(self):
        # Configuration from environment variables
        self.suppress_errors = os.getenv("LUCIDIC_SUPPRESS_ERRORS", "true").lower() == "true"
        self.cleanup_on_error = os.getenv("LUCIDIC_CLEANUP_ON_ERROR", "true").lower() == "true"
        self.log_suppressed = os.getenv("LUCIDIC_LOG_SUPPRESSED", "true").lower() == "true"
        self.error_count = 0
        self.last_cleanup = 0
        self.cleanup_interval = 60  # Minimum seconds between cleanups
        
    def safe_execute(
        self, 
        return_default: Any = None,
        critical: bool = False,
        cleanup: bool = None
    ) -> Callable[[F], F]:
        """Decorator for safe execution of SDK functions.
        
        Args:
            return_default: Value to return if function fails (can be callable)
            critical: If True, always attempt cleanup on error
            cleanup: Override cleanup behavior (None uses global setting)
        
        Returns:
            Decorated function that suppresses errors based on configuration
        """
        def decorator(func: F) -> F:
            # Determine if function is async
            is_async = asyncio.iscoroutinefunction(func)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # If error suppression is disabled, run normally
                if not self.suppress_errors:
                    return func(*args, **kwargs)
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._handle_error(e, func.__name__, critical)
                    
                    # Determine if cleanup should run
                    should_cleanup = cleanup if cleanup is not None else self.cleanup_on_error
                    if should_cleanup or critical:
                        self._cleanup_resources()
                    
                    # Return default value
                    if callable(return_default):
                        try:
                            return return_default()
                        except Exception:
                            return None
                    return return_default
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # If error suppression is disabled, run normally
                if not self.suppress_errors:
                    return await func(*args, **kwargs)
                
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    self._handle_error(e, func.__name__, critical)
                    
                    # Determine if cleanup should run
                    should_cleanup = cleanup if cleanup is not None else self.cleanup_on_error
                    if should_cleanup or critical:
                        self._cleanup_resources()
                    
                    # Return default value
                    if callable(return_default):
                        try:
                            result = return_default()
                            if asyncio.iscoroutine(result):
                                return await result
                            return result
                        except Exception:
                            return None
                    return return_default
            
            # Return appropriate wrapper
            if is_async:
                return async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore
        
        return decorator
    
    def _handle_error(self, error: Exception, context: str, critical: bool = False):
        """Log error internally without propagating.
        
        Args:
            error: The exception that occurred
            context: Function name or context where error occurred
            critical: Whether this is a critical error
        """
        self.error_count += 1
        
        if self.log_suppressed:
            level = logging.WARNING if critical else logging.DEBUG
            logger.log(
                level,
                f"[SDK Error Suppressed] {context}: {error.__class__.__name__}: {error}"
            )
            
            # Log full traceback at debug level
            if logger.isEnabledFor(logging.DEBUG):
                import traceback
                logger.debug(f"[SDK Error Traceback]\n{traceback.format_exc()}")
    
    def _cleanup_resources(self):
        """Best-effort cleanup of SDK resources.
        
        Attempts to:
        1. Flush pending events
        2. End active sessions
        3. Close HTTP connections
        4. Clear telemetry resources
        """
        # Rate limit cleanup to avoid excessive attempts
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        self.last_cleanup = current_time
        
        try:
            # Import here to avoid circular dependency
            from .client import Client
            from .singleton import lai_inst
            
            # Get client if it exists
            client = lai_inst.get(Client)
            if not client or isinstance(client, type(client).__bases__):
                return
            
            logger.debug("[SDK Cleanup] Starting resource cleanup after error")
            
            # 1. Force flush events (short timeout to avoid hanging)
            if hasattr(client, '_event_queue') and client._event_queue:
                try:
                    logger.debug("[SDK Cleanup] Flushing event queue")
                    client._event_queue.force_flush(timeout_seconds=2.0)
                except Exception as e:
                    logger.debug(f"[SDK Cleanup] Event flush failed: {e}")
            
            # 2. Flush telemetry spans
            if hasattr(client, '_tracer_provider') and client._tracer_provider:
                try:
                    logger.debug("[SDK Cleanup] Flushing telemetry spans")
                    client._tracer_provider.force_flush(timeout_millis=2000)
                except Exception as e:
                    logger.debug(f"[SDK Cleanup] Telemetry flush failed: {e}")
            
            # 3. End active session (mark as unsuccessful)
            if hasattr(client, 'session') and client.session:
                try:
                    if not getattr(client.session, 'is_finished', False):
                        logger.debug("[SDK Cleanup] Ending active session")
                        client.session.update_session(
                            is_finished=True,
                            is_successful=False,
                            is_successful_reason="SDK error during execution"
                        )
                except Exception as e:
                    logger.debug(f"[SDK Cleanup] Session end failed: {e}")
            
            # 4. Close HTTP connections gracefully
            if hasattr(client, 'request_session') and client.request_session:
                try:
                    logger.debug("[SDK Cleanup] Closing HTTP session")
                    client.request_session.close()
                except Exception as e:
                    logger.debug(f"[SDK Cleanup] HTTP close failed: {e}")
            
            logger.debug("[SDK Cleanup] Resource cleanup completed")
            
        except Exception as e:
            # Cleanup itself must never raise
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[SDK Cleanup] Cleanup error: {e}")
    
    def wrap_class_methods(self, cls: type, exclude: list = None) -> type:
        """Wrap all public methods of a class with error suppression.
        
        Args:
            cls: Class whose methods to wrap
            exclude: List of method names to exclude from wrapping
        
        Returns:
            Class with wrapped methods
        """
        exclude = exclude or []
        
        for attr_name in dir(cls):
            # Skip private/magic methods and excluded methods
            if attr_name.startswith('_') or attr_name in exclude:
                continue
            
            attr = getattr(cls, attr_name)
            if callable(attr):
                # Wrap the method with safe_execute
                wrapped = self.safe_execute()(attr)
                setattr(cls, attr_name, wrapped)
        
        return cls
    
    def __call__(self, *args, **kwargs):
        """Allow using the handler as a decorator directly."""
        return self.safe_execute(*args, **kwargs)


# Global error handler instance
error_handler = SDKErrorHandler()

# Export convenience decorator
safe_execute = error_handler.safe_execute


def get_default_return(func_name: str) -> Any:
    """Get appropriate default return value for a function.
    
    Args:
        func_name: Name of the function
        
    Returns:
        Appropriate default value based on function name
    """
    # Functions that return IDs should return a placeholder
    if 'id' in func_name.lower() or func_name in ['init', 'create_experiment']:
        import uuid
        return str(uuid.uuid4())
    
    # Functions that return session objects
    if 'session' in func_name.lower():
        return None
    
    # Functions that return events
    if 'event' in func_name.lower():
        import uuid
        return str(uuid.uuid4())
    
    # Functions that return data
    if func_name.startswith('get_'):
        return {} if 'dataset' in func_name else None
    
    # Default
    return None


def install_error_handler():
    """Install global error handler for uncaught exceptions in SDK code.
    
    This is called during SDK initialization to set up error boundaries.
    """
    if not error_handler.suppress_errors:
        return
    
    logger.debug("[SDK Error Handler] Installed with error suppression enabled")
    
    # Log configuration
    logger.debug(f"[SDK Error Handler] Configuration:")
    logger.debug(f"  - Suppress errors: {error_handler.suppress_errors}")
    logger.debug(f"  - Cleanup on error: {error_handler.cleanup_on_error}")
    logger.debug(f"  - Log suppressed: {error_handler.log_suppressed}")