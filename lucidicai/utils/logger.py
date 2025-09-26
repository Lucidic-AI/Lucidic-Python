"""Centralized logging utilities for Lucidic SDK.

This module provides consistent logging functions that respect
LUCIDIC_DEBUG and LUCIDIC_VERBOSE environment variables.
"""
import os
import logging
from typing import Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure base logger
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.WARNING  # Default to WARNING, will be overridden by env vars
)

# Get the logger
logger = logging.getLogger("Lucidic")


def _env_true(value: Optional[str]) -> bool:
    """Check if environment variable is truthy."""
    if value is None:
        return False
    return value.lower() in ('true', '1', 'yes')


def is_debug() -> bool:
    """Check if debug mode is enabled."""
    return _env_true(os.getenv('LUCIDIC_DEBUG'))


def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    return _env_true(os.getenv('LUCIDIC_VERBOSE'))


def debug(message: str, *args: Any, **kwargs: Any) -> None:
    """Log debug message if LUCIDIC_DEBUG is enabled.
    
    Args:
        message: Log message with optional formatting
        *args: Positional arguments for message formatting
        **kwargs: Keyword arguments for logging
    """
    if is_debug():
        logger.debug(f"[DEBUG] {message}", *args, **kwargs)


def info(message: str, *args: Any, **kwargs: Any) -> None:
    """Log info message (always visible).
    
    Args:
        message: Log message with optional formatting
        *args: Positional arguments for message formatting
        **kwargs: Keyword arguments for logging
    """
    logger.info(message, *args, **kwargs)


def warning(message: str, *args: Any, **kwargs: Any) -> None:
    """Log warning message (always visible).
    
    Args:
        message: Log message with optional formatting
        *args: Positional arguments for message formatting
        **kwargs: Keyword arguments for logging
    """
    logger.warning(message, *args, **kwargs)


def error(message: str, *args: Any, **kwargs: Any) -> None:
    """Log error message (always visible).
    
    Args:
        message: Log message with optional formatting
        *args: Positional arguments for message formatting
        **kwargs: Keyword arguments for logging
    """
    logger.error(message, *args, **kwargs)


def verbose(message: str, *args: Any, **kwargs: Any) -> None:
    """Log verbose message if LUCIDIC_VERBOSE or LUCIDIC_DEBUG is enabled.
    
    Args:
        message: Log message with optional formatting
        *args: Positional arguments for message formatting
        **kwargs: Keyword arguments for logging
    """
    if is_debug() or is_verbose():
        logger.info(f"[VERBOSE] {message}", *args, **kwargs)


def truncate_id(id_str: Optional[str], length: int = 8) -> str:
    """Truncate UUID for logging.
    
    Args:
        id_str: UUID string to truncate
        length: Number of characters to keep
        
    Returns:
        Truncated ID with ellipsis
    """
    if not id_str:
        return "None"
    if len(id_str) <= length:
        return id_str
    return f"{id_str[:length]}..."


def mask_sensitive(data: dict, sensitive_keys: set = None) -> dict:
    """Mask sensitive data in dictionary for logging.
    
    Args:
        data: Dictionary potentially containing sensitive data
        sensitive_keys: Set of keys to mask (default: common sensitive keys)
        
    Returns:
        Dictionary with sensitive values masked
    """
    if sensitive_keys is None:
        sensitive_keys = {
            'api_key', 'apikey', 'api-key',
            'token', 'auth', 'authorization',
            'password', 'secret', 'key',
            'x-api-key', 'x-auth-token'
        }
    
    masked = {}
    for key, value in data.items():
        if any(k in key.lower() for k in sensitive_keys):
            if value:
                # Show first few chars for debugging
                masked[key] = f"{str(value)[:4]}...MASKED" if len(str(value)) > 4 else "MASKED"
            else:
                masked[key] = value
        else:
            masked[key] = value
    return masked


def truncate_data(data: Any, max_length: int = 500) -> str:
    """Truncate long data for logging.
    
    Args:
        data: Data to truncate
        max_length: Maximum length
        
    Returns:
        Truncated string representation
    """
    str_data = str(data)
    if len(str_data) <= max_length:
        return str_data
    return f"{str_data[:max_length]}... (truncated)"


# Configure logger level based on environment
if is_debug():
    logger.setLevel(logging.DEBUG)
elif is_verbose():
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.WARNING)