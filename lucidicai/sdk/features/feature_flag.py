import os
import logging
import time
from typing import Union, List, Dict, Any, Optional, overload, Tuple, Literal
from dotenv import load_dotenv

from ..init import get_http
from ...core.errors import APIKeyVerificationError, FeatureFlagError

logger = logging.getLogger("Lucidic")

# Cache implementation
class FeatureFlagCache:
    def __init__(self):
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._default_ttl = 300  # 5 minutes
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        if ttl is None:
            ttl = self._default_ttl
        if ttl > 0:
            self._cache[key] = (value, time.time() + ttl)
    
    def clear(self):
        self._cache.clear()

# Global cache instance
_flag_cache = FeatureFlagCache()

# Sentinel value to distinguish None from missing
MISSING = object()

# Function overloads for type safety
@overload
def get_feature_flag(
    flag_name: str,
    default: Any = ...,
    *,
    return_missing: Literal[False] = False,
    cache_ttl: Optional[int] = 300,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Any:
    """Get a single feature flag."""
    ...

@overload
def get_feature_flag(
    flag_name: str,
    default: Any = ...,
    *,
    return_missing: Literal[True],
    cache_ttl: Optional[int] = 300,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Tuple[Any, List[str]]:
    """Get a single feature flag with missing info."""
    ...

@overload
def get_feature_flag(
    flag_name: List[str],
    defaults: Optional[Dict[str, Any]] = None,
    *,
    return_missing: Literal[False] = False,
    cache_ttl: Optional[int] = 300,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Get multiple feature flags."""
    ...

@overload
def get_feature_flag(
    flag_name: List[str],
    defaults: Optional[Dict[str, Any]] = None,
    *,
    return_missing: Literal[True],
    cache_ttl: Optional[int] = 300,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Get multiple feature flags with missing info."""
    ...

def get_feature_flag(
    flag_name: Union[str, List[str]],
    default_or_defaults: Any = MISSING,
    *,
    return_missing: bool = False,
    cache_ttl: Optional[int] = 300,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Union[Any, Tuple[Any, List[str]], Dict[str, Any], Tuple[Dict[str, Any], List[str]]]:
    """
    Get feature flag(s) from backend. Raises FeatureFlagError on failure unless default provided.
    
    Args:
        flag_name: Single flag name (str) or list of flag names
        default_or_defaults: 
            - If flag_name is str: default value for that flag (optional)
            - If flag_name is List[str]: dict of defaults {flag_name: default_value}
        cache_ttl: Cache time-to-live in seconds (0 to disable, -1 for forever)
        api_key: Optional API key
        agent_id: Optional agent ID
    
    Returns:
        - If flag_name is str: The flag value (or tuple with missing list if return_missing=True)
        - If flag_name is List[str]: Dict mapping flag_name -> value (or tuple with missing list if return_missing=True)
    
    Raises:
        FeatureFlagError: If fetch fails and no default provided
        APIKeyVerificationError: If credentials missing
    
    Examples:
        # Single flag with default
        retries = lai.get_feature_flag("max_retries", default=3)
        
        # Single flag without default (can raise)
        retries = lai.get_feature_flag("max_retries")
        
        # Multiple flags
        flags = lai.get_feature_flag(
            ["max_retries", "timeout"],
            defaults={"max_retries": 3}
        )
    """

    load_dotenv()
    
    # Determine if single or batch
    is_single = isinstance(flag_name, str)
    flag_names = [flag_name] if is_single else flag_name
    
    # Parse defaults
    if is_single:
        has_default = default_or_defaults is not MISSING
        defaults = {flag_name: default_or_defaults} if has_default else {}
    else:
        defaults = default_or_defaults if default_or_defaults not in (None, MISSING) else {}
    
    # Track missing flags
    missing_flags = []
    
    # Check cache first
    uncached_flags = []
    cached_results = {}
    
    if cache_ttl != 0:
        for name in flag_names:
            cache_key = f"{agent_id}:{name}"
            cached_value = _flag_cache.get(cache_key)
            if cached_value is not None:
                cached_results[name] = cached_value
            else:
                uncached_flags.append(name)
    else:
        uncached_flags = flag_names
    
    # Fetch uncached flags if needed
    if uncached_flags:
        # Get credentials
        if api_key is None:
            api_key = os.getenv("LUCIDIC_API_KEY", None)
            if api_key is None:
                raise APIKeyVerificationError(
                    "Make sure to either pass your API key or set the LUCIDIC_API_KEY environment variable."
                )
        
        if agent_id is None:
            agent_id = os.getenv("LUCIDIC_AGENT_ID", None)
            if agent_id is None:
                raise APIKeyVerificationError(
                    "Lucidic agent ID not specified. Make sure to either pass your agent ID or set the LUCIDIC_AGENT_ID environment variable."
                )
        
        # Get HTTP client
        http = get_http()
        if not http:
            from ..init import create_session
            create_session(api_key=api_key, agent_id=agent_id)
            http = get_http()

        # check for active session
        from ..init import get_session_id
        session_id = get_session_id()

        try:
            if len(uncached_flags) == 1:
                # Single flag evaluation
                if session_id:
                    # Use session-based evaluation for consistency
                    response = http.post('evaluatefeatureflag', {
                        'session_id': session_id,
                        'flag_name': uncached_flags[0],
                        'context': {},
                        'default': defaults.get(uncached_flags[0])
                    })
                else:
                    # Use stateless evaluation as fallback
                    response = http.post('evaluatefeatureflagstateless', {
                        'agent_id': agent_id,
                        'flag_name': uncached_flags[0],
                        'context': {},
                        'default': defaults.get(uncached_flags[0])
                    })

                # Extract value from response
                if 'value' in response:
                    value = response['value']
                    cached_results[uncached_flags[0]] = value

                    # Cache the result
                    if cache_ttl != 0:
                        cache_key = f"{agent_id}:{uncached_flags[0]}"
                        _flag_cache.set(cache_key, value, ttl=cache_ttl if cache_ttl > 0 else None)
                elif 'error' in response:
                    # Flag not found or error
                    logger.warning(f"Feature flag error: {response['error']}")
                    missing_flags.append(uncached_flags[0])

            else:
                # Batch evaluation
                if session_id:
                    # Use session-based batch evaluation
                    response = http.post('evaluatebatchfeatureflags', {
                        'session_id': session_id,
                        'flag_names': uncached_flags,
                        'context': {},
                        'defaults': {k: v for k, v in defaults.items() if k in uncached_flags}
                    })
                else:
                    # Use stateless batch evaluation
                    response = http.post('evaluatebatchfeatureflagsstateless', {
                        'agent_id': agent_id,
                        'flag_names': uncached_flags,
                        'context': {},
                        'defaults': {k: v for k, v in defaults.items() if k in uncached_flags}
                    })

                # Process batch response
                if 'flags' in response:
                    for name in uncached_flags:
                        flag_data = response['flags'].get(name)
                        if flag_data and 'value' in flag_data:
                            value = flag_data['value']
                            cached_results[name] = value

                            # Cache it
                            if cache_ttl != 0:
                                cache_key = f"{agent_id}:{name}"
                                _flag_cache.set(cache_key, value, ttl=cache_ttl if cache_ttl > 0 else None)
                        else:
                            missing_flags.append(name)

        except Exception as e:
            # HTTP client raises on errors, fall back to defaults
            logger.error(f"Failed to fetch feature flags: {e}")

            # Use defaults for all uncached flags
            for name in uncached_flags:
                if name in defaults:
                    cached_results[name] = defaults[name]
                else:
                    missing_flags.append(name)
                    if is_single and not return_missing:
                        raise FeatureFlagError(f"'{name}': {e}") from e
    
    # Build final result
    result = {}
    for name in flag_names:
        if name in cached_results:
            result[name] = cached_results[name]
        elif name in defaults:
            result[name] = defaults[name]
        else:
            # No value and no default
            missing_flags.append(name)
            if is_single and not return_missing:
                raise FeatureFlagError(f"'{name}' not found and no default provided")
            else:
                result[name] = None
    
    # Return based on input type and return_missing flag
    if return_missing:
        return (result[flag_names[0]] if is_single else result, missing_flags)
    else:
        return result[flag_names[0]] if is_single else result


# Typed convenience functions
def get_bool_flag(flag_name: str, default: Optional[bool] = None, **kwargs) -> bool:
    """
    Get a boolean feature flag with type validation.

    Raises:
        FeatureFlagError: If fetch fails and no default provided
        TypeError: If flag value is not a boolean
    """
    value = get_feature_flag(flag_name, default=default if default is not None else MISSING, **kwargs)
    if not isinstance(value, bool):
        if default is not None:
            logger.warning(f"Feature flag '{flag_name}' is not a boolean, using default")
            return default
        raise TypeError(f"Feature flag '{flag_name}' expected boolean, got {type(value).__name__}")
    return value


def get_int_flag(flag_name: str, default: Optional[int] = None, **kwargs) -> int:
    """
    Get an integer feature flag with type validation.

    Raises:
        FeatureFlagError: If fetch fails and no default provided
        TypeError: If flag value is not an integer
    """
    value = get_feature_flag(flag_name, default=default if default is not None else MISSING, **kwargs)
    if not isinstance(value, int) or isinstance(value, bool):  # bool is subclass of int
        if default is not None:
            logger.warning(f"Feature flag '{flag_name}' is not an integer, using default")
            return default
        raise TypeError(f"Feature flag '{flag_name}' expected integer, got {type(value).__name__}")
    return value


def get_float_flag(flag_name: str, default: Optional[float] = None, **kwargs) -> float:
    """
    Get a float feature flag with type validation.

    Raises:
        FeatureFlagError: If fetch fails and no default provided
        TypeError: If flag value is not a float
    """
    value = get_feature_flag(flag_name, default=default if default is not None else MISSING, **kwargs)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        if default is not None:
            logger.warning(f"Feature flag '{flag_name}' is not a float, using default")
            return default
        raise TypeError(f"Feature flag '{flag_name}' expected float, got {type(value).__name__}")
    return float(value)


def get_string_flag(flag_name: str, default: Optional[str] = None, **kwargs) -> str:
    """
    Get a string feature flag with type validation.

    Raises:
        FeatureFlagError: If fetch fails and no default provided
        TypeError: If flag value is not a string
    """
    value = get_feature_flag(flag_name, default=default if default is not None else MISSING, **kwargs)
    if not isinstance(value, str):
        if default is not None:
            logger.warning(f"Feature flag '{flag_name}' is not a string, using default")
            return default
        raise TypeError(f"Feature flag '{flag_name}' expected string, got {type(value).__name__}")
    return value


def get_json_flag(flag_name: str, default: Optional[dict] = None, **kwargs) -> dict:
    """
    Get a JSON object feature flag.

    Raises:
        FeatureFlagError: If fetch fails and no default provided
    """
    value = get_feature_flag(flag_name, default=default if default is not None else MISSING, **kwargs)
    return value


def clear_feature_flag_cache():
    """Clear the feature flag cache."""
    _flag_cache.clear()
    logger.debug("Feature flag cache cleared")


# ==================== Asynchronous Functions ====================


# Async function overloads for type safety
@overload
async def aget_feature_flag(
    flag_name: str,
    default: Any = ...,
    *,
    return_missing: Literal[False] = False,
    cache_ttl: Optional[int] = 300,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Any:
    """Get a single feature flag (asynchronous)."""
    ...

@overload
async def aget_feature_flag(
    flag_name: str,
    default: Any = ...,
    *,
    return_missing: Literal[True],
    cache_ttl: Optional[int] = 300,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Tuple[Any, List[str]]:
    """Get a single feature flag with missing info (asynchronous)."""
    ...

@overload
async def aget_feature_flag(
    flag_name: List[str],
    defaults: Optional[Dict[str, Any]] = None,
    *,
    return_missing: Literal[False] = False,
    cache_ttl: Optional[int] = 300,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Get multiple feature flags (asynchronous)."""
    ...

@overload
async def aget_feature_flag(
    flag_name: List[str],
    defaults: Optional[Dict[str, Any]] = None,
    *,
    return_missing: Literal[True],
    cache_ttl: Optional[int] = 300,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Get multiple feature flags with missing info (asynchronous)."""
    ...

async def aget_feature_flag(
    flag_name: Union[str, List[str]],
    default_or_defaults: Any = MISSING,
    *,
    return_missing: bool = False,
    cache_ttl: Optional[int] = 300,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Union[Any, Tuple[Any, List[str]], Dict[str, Any], Tuple[Dict[str, Any], List[str]]]:
    """
    Get feature flag(s) from backend (asynchronous). Raises FeatureFlagError on failure unless default provided.
    
    Args:
        flag_name: Single flag name (str) or list of flag names
        default_or_defaults: 
            - If flag_name is str: default value for that flag (optional)
            - If flag_name is List[str]: dict of defaults {flag_name: default_value}
        cache_ttl: Cache time-to-live in seconds (0 to disable, -1 for forever)
        api_key: Optional API key
        agent_id: Optional agent ID
    
    Returns:
        - If flag_name is str: The flag value (or tuple with missing list if return_missing=True)
        - If flag_name is List[str]: Dict mapping flag_name -> value (or tuple with missing list if return_missing=True)
    
    Raises:
        FeatureFlagError: If fetch fails and no default provided
        APIKeyVerificationError: If credentials missing
    
    Examples:
        # Single flag with default
        retries = await lai.aget_feature_flag("max_retries", default=3)
        
        # Single flag without default (can raise)
        retries = await lai.aget_feature_flag("max_retries")
        
        # Multiple flags
        flags = await lai.aget_feature_flag(
            ["max_retries", "timeout"],
            defaults={"max_retries": 3}
        )
    """

    load_dotenv()
    
    # Determine if single or batch
    is_single = isinstance(flag_name, str)
    flag_names = [flag_name] if is_single else flag_name
    
    # Parse defaults
    if is_single:
        has_default = default_or_defaults is not MISSING
        defaults = {flag_name: default_or_defaults} if has_default else {}
    else:
        defaults = default_or_defaults if default_or_defaults not in (None, MISSING) else {}
    
    # Track missing flags
    missing_flags = []
    
    # Check cache first
    uncached_flags = []
    cached_results = {}
    
    if cache_ttl != 0:
        for name in flag_names:
            cache_key = f"{agent_id}:{name}"
            cached_value = _flag_cache.get(cache_key)
            if cached_value is not None:
                cached_results[name] = cached_value
            else:
                uncached_flags.append(name)
    else:
        uncached_flags = flag_names
    
    # Fetch uncached flags if needed
    if uncached_flags:
        # Get credentials
        if api_key is None:
            api_key = os.getenv("LUCIDIC_API_KEY", None)
            if api_key is None:
                raise APIKeyVerificationError(
                    "Make sure to either pass your API key or set the LUCIDIC_API_KEY environment variable."
                )
        
        if agent_id is None:
            agent_id = os.getenv("LUCIDIC_AGENT_ID", None)
            if agent_id is None:
                raise APIKeyVerificationError(
                    "Lucidic agent ID not specified. Make sure to either pass your agent ID or set the LUCIDIC_AGENT_ID environment variable."
                )
        
        # Get HTTP client
        http = get_http()
        if not http:
            from ..init import create_session
            create_session(api_key=api_key, agent_id=agent_id)
            http = get_http()

        # check for active session
        from ..init import get_session_id
        session_id = get_session_id()

        try:
            if len(uncached_flags) == 1:
                # Single flag evaluation
                if session_id:
                    # Use session-based evaluation for consistency
                    response = await http.apost('evaluatefeatureflag', {
                        'session_id': session_id,
                        'flag_name': uncached_flags[0],
                        'context': {},
                        'default': defaults.get(uncached_flags[0])
                    })
                else:
                    # Use stateless evaluation as fallback
                    response = await http.apost('evaluatefeatureflagstateless', {
                        'agent_id': agent_id,
                        'flag_name': uncached_flags[0],
                        'context': {},
                        'default': defaults.get(uncached_flags[0])
                    })

                # Extract value from response
                if 'value' in response:
                    value = response['value']
                    cached_results[uncached_flags[0]] = value

                    # Cache the result
                    if cache_ttl != 0:
                        cache_key = f"{agent_id}:{uncached_flags[0]}"
                        _flag_cache.set(cache_key, value, ttl=cache_ttl if cache_ttl > 0 else None)
                elif 'error' in response:
                    # Flag not found or error
                    logger.warning(f"Feature flag error: {response['error']}")
                    missing_flags.append(uncached_flags[0])

            else:
                # Batch evaluation
                if session_id:
                    # Use session-based batch evaluation
                    response = await http.apost('evaluatebatchfeatureflags', {
                        'session_id': session_id,
                        'flag_names': uncached_flags,
                        'context': {},
                        'defaults': {k: v for k, v in defaults.items() if k in uncached_flags}
                    })
                else:
                    # Use stateless batch evaluation
                    response = await http.apost('evaluatebatchfeatureflagsstateless', {
                        'agent_id': agent_id,
                        'flag_names': uncached_flags,
                        'context': {},
                        'defaults': {k: v for k, v in defaults.items() if k in uncached_flags}
                    })

                # Process batch response
                if 'flags' in response:
                    for name in uncached_flags:
                        flag_data = response['flags'].get(name)
                        if flag_data and 'value' in flag_data:
                            value = flag_data['value']
                            cached_results[name] = value

                            # Cache it
                            if cache_ttl != 0:
                                cache_key = f"{agent_id}:{name}"
                                _flag_cache.set(cache_key, value, ttl=cache_ttl if cache_ttl > 0 else None)
                        else:
                            missing_flags.append(name)

        except Exception as e:
            # HTTP client raises on errors, fall back to defaults
            logger.error(f"Failed to fetch feature flags: {e}")

            # Use defaults for all uncached flags
            for name in uncached_flags:
                if name in defaults:
                    cached_results[name] = defaults[name]
                else:
                    missing_flags.append(name)
                    if is_single and not return_missing:
                        raise FeatureFlagError(f"'{name}': {e}") from e
    
    # Build final result
    result = {}
    for name in flag_names:
        if name in cached_results:
            result[name] = cached_results[name]
        elif name in defaults:
            result[name] = defaults[name]
        else:
            # No value and no default
            missing_flags.append(name)
            if is_single and not return_missing:
                raise FeatureFlagError(f"'{name}' not found and no default provided")
            else:
                result[name] = None
    
    # Return based on input type and return_missing flag
    if return_missing:
        return (result[flag_names[0]] if is_single else result, missing_flags)
    else:
        return result[flag_names[0]] if is_single else result


# Async typed convenience functions
async def aget_bool_flag(flag_name: str, default: Optional[bool] = None, **kwargs) -> bool:
    """
    Get a boolean feature flag with type validation (asynchronous).

    Raises:
        FeatureFlagError: If fetch fails and no default provided
        TypeError: If flag value is not a boolean
    """
    value = await aget_feature_flag(flag_name, default=default if default is not None else MISSING, **kwargs)
    if not isinstance(value, bool):
        if default is not None:
            logger.warning(f"Feature flag '{flag_name}' is not a boolean, using default")
            return default
        raise TypeError(f"Feature flag '{flag_name}' expected boolean, got {type(value).__name__}")
    return value


async def aget_int_flag(flag_name: str, default: Optional[int] = None, **kwargs) -> int:
    """
    Get an integer feature flag with type validation (asynchronous).

    Raises:
        FeatureFlagError: If fetch fails and no default provided
        TypeError: If flag value is not an integer
    """
    value = await aget_feature_flag(flag_name, default=default if default is not None else MISSING, **kwargs)
    if not isinstance(value, int) or isinstance(value, bool):  # bool is subclass of int
        if default is not None:
            logger.warning(f"Feature flag '{flag_name}' is not an integer, using default")
            return default
        raise TypeError(f"Feature flag '{flag_name}' expected integer, got {type(value).__name__}")
    return value


async def aget_float_flag(flag_name: str, default: Optional[float] = None, **kwargs) -> float:
    """
    Get a float feature flag with type validation (asynchronous).

    Raises:
        FeatureFlagError: If fetch fails and no default provided
        TypeError: If flag value is not a float
    """
    value = await aget_feature_flag(flag_name, default=default if default is not None else MISSING, **kwargs)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        if default is not None:
            logger.warning(f"Feature flag '{flag_name}' is not a float, using default")
            return default
        raise TypeError(f"Feature flag '{flag_name}' expected float, got {type(value).__name__}")
    return float(value)


async def aget_string_flag(flag_name: str, default: Optional[str] = None, **kwargs) -> str:
    """
    Get a string feature flag with type validation (asynchronous).

    Raises:
        FeatureFlagError: If fetch fails and no default provided
        TypeError: If flag value is not a string
    """
    value = await aget_feature_flag(flag_name, default=default if default is not None else MISSING, **kwargs)
    if not isinstance(value, str):
        if default is not None:
            logger.warning(f"Feature flag '{flag_name}' is not a string, using default")
            return default
        raise TypeError(f"Feature flag '{flag_name}' expected string, got {type(value).__name__}")
    return value


async def aget_json_flag(flag_name: str, default: Optional[dict] = None, **kwargs) -> dict:
    """
    Get a JSON object feature flag (asynchronous).

    Raises:
        FeatureFlagError: If fetch fails and no default provided
    """
    value = await aget_feature_flag(flag_name, default=default if default is not None else MISSING, **kwargs)
    return value