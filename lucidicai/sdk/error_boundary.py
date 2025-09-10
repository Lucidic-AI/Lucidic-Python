"""Error boundary pattern for SDK error suppression.

Inspired by the TypeScript SDK's error-boundary.ts, this module provides
a clean, centralized way to handle SDK errors without affecting user code.
"""
import functools
import logging
import threading
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

from ..core.config import get_config

logger = logging.getLogger("Lucidic")

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class ErrorContext:
    """Context information about an SDK error."""
    timestamp: datetime
    module: str
    function: str
    error_type: str
    error_message: str
    traceback: str
    suppressed: bool = True


class ErrorBoundary:
    """Centralized error boundary for the SDK.
    
    This class manages all error suppression, logging, and cleanup
    in a single place, similar to the TypeScript implementation.
    """
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.max_history_size = 100
        self.cleanup_handlers: List[Callable] = []
        self._lock = threading.Lock()
        self._config = None  # Lazy load config
    
    def wrap_function(self, func: F, module: str = "unknown") -> F:
        """Wrap a function with error boundary protection.
        
        Args:
            func: Function to wrap
            module: Module name for error context
            
        Returns:
            Wrapped function that suppresses errors based on config
        """
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not self.is_silent_mode():
                return func(*args, **kwargs)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return self._handle_error(e, module, func.__name__, args, kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not self.is_silent_mode():
                return await func(*args, **kwargs)
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return self._handle_error(e, module, func.__name__, args, kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    def wrap_module(self, module_dict: Dict[str, Any], module_name: str) -> Dict[str, Any]:
        """Wrap all functions in a module dictionary.
        
        Args:
            module_dict: Dictionary of module exports
            module_name: Name of the module
            
        Returns:
            Dictionary with all functions wrapped
        """
        if not self.is_silent_mode():
            return module_dict
        
        wrapped = {}
        for name, obj in module_dict.items():
            if callable(obj) and not name.startswith('_'):
                wrapped[name] = self.wrap_function(obj, module_name)
            else:
                wrapped[name] = obj
        
        return wrapped
    
    @property
    def config(self):
        """Lazy load configuration."""
        if self._config is None:
            self._config = get_config()
        return self._config
    
    def is_silent_mode(self) -> bool:
        """Check if SDK is in silent mode (error suppression enabled)."""
        return self.config.error_handling.suppress_errors
    
    def _handle_error(
        self,
        error: Exception,
        module: str,
        function: str,
        args: tuple,
        kwargs: dict
    ) -> Any:
        """Handle an error that occurred in SDK code.
        
        Args:
            error: The exception that occurred
            module: Module where error occurred
            function: Function where error occurred
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Default return value for the function
        """
        # Create error context
        context = ErrorContext(
            timestamp=datetime.now(),
            module=module,
            function=function,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            suppressed=True
        )
        
        # Add to history
        with self._lock:
            self.error_history.append(context)
            if len(self.error_history) > self.max_history_size:
                self.error_history.pop(0)
        
        # Log if configured
        if self.config.error_handling.log_suppressed:
            logger.debug(
                f"[ErrorBoundary] Suppressed {context.error_type} in {module}.{function}: "
                f"{context.error_message}"
            )
            if self.config.debug:
                logger.debug(f"[ErrorBoundary] Traceback:\n{context.traceback}")
        
        # Perform cleanup if configured
        if self.config.error_handling.cleanup_on_error:
            self._perform_cleanup()
        
        # Return appropriate default
        return self._get_default_return(function)
    
    def _get_default_return(self, function_name: str) -> Any:
        """Get appropriate default return value for a function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            Appropriate default value based on function name
        """
        # init function should return a fallback session ID
        if function_name == 'init':
            return f'fallback-session-{uuid.uuid4()}'
        
        # Functions that return IDs
        if any(x in function_name.lower() for x in ['id', 'create_experiment']):
            return str(uuid.uuid4())
        
        # Functions that return booleans
        if any(x in function_name.lower() for x in ['is_', 'has_', 'can_', 'should_']):
            return False
        
        # Functions that return data
        if function_name.lower().startswith('get_'):
            if 'dataset' in function_name.lower():
                return {}
            elif 'prompt' in function_name.lower():
                return ""
            return None
        
        # Default
        return None
    
    def _perform_cleanup(self) -> None:
        """Perform cleanup after an error."""
        # Run registered cleanup handlers
        for handler in self.cleanup_handlers:
            try:
                handler()
            except Exception as e:
                logger.debug(f"[ErrorBoundary] Cleanup handler failed: {e}")
    
    def register_cleanup_handler(self, handler: Callable) -> None:
        """Register a cleanup handler to run on errors.
        
        Args:
            handler: Cleanup function to register
        """
        self.cleanup_handlers.append(handler)
    
    def get_error_history(self) -> List[ErrorContext]:
        """Get the error history.
        
        Returns:
            List of error contexts
        """
        with self._lock:
            return list(self.error_history)
    
    def clear_error_history(self) -> None:
        """Clear the error history."""
        with self._lock:
            self.error_history.clear()
    
    def reset_config(self) -> None:
        """Reset cached configuration (for testing)."""
        self._config = None


# Global error boundary instance
_error_boundary: Optional[ErrorBoundary] = None


def get_error_boundary() -> ErrorBoundary:
    """Get the global error boundary instance."""
    global _error_boundary
    if _error_boundary is None:
        _error_boundary = ErrorBoundary()
    return _error_boundary


def wrap_sdk_function(func: F, module: str = "unknown") -> F:
    """Wrap an SDK function with error boundary protection.
    
    Args:
        func: Function to wrap
        module: Module name for error context
        
    Returns:
        Wrapped function
    """
    return get_error_boundary().wrap_function(func, module)


def wrap_sdk_module(module_dict: Dict[str, Any], module_name: str) -> Dict[str, Any]:
    """Wrap all functions in an SDK module.
    
    Args:
        module_dict: Dictionary of module exports
        module_name: Name of the module
        
    Returns:
        Dictionary with wrapped functions
    """
    return get_error_boundary().wrap_module(module_dict, module_name)


def is_silent_mode() -> bool:
    """Check if SDK is in silent mode."""
    return get_error_boundary().is_silent_mode()




def get_error_history() -> List[ErrorContext]:
    """Get the SDK error history."""
    return get_error_boundary().get_error_history()


def clear_error_history() -> None:
    """Clear the SDK error history."""
    get_error_boundary().clear_error_history()


def register_cleanup_handler(handler: Callable) -> None:
    """Register a cleanup handler.
    
    Args:
        handler: Cleanup function to run on errors
    """
    get_error_boundary().register_cleanup_handler(handler)