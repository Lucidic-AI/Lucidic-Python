"""Lucidic AI SDK - Clean Export-Only Entry Point

This file only contains exports, with all logic moved to appropriate modules.
"""

# Import core modules
from .sdk import init as init_module
from .sdk import event as event_module
from .sdk import error_boundary
from .core.config import get_config

# Import raw functions
from .sdk.init import (
    init as _init,
    get_session_id as _get_session_id,
    clear_state as _clear_state,
)

from .sdk.event import (
    create_event as _create_event,
    create_error_event as _create_error_event,
    flush as _flush,
)

# Context management exports
from .sdk.context import (
    set_active_session,
    clear_active_session,
    bind_session,
    bind_session_async,
    session,
    session_async,
    run_session,
    run_in_session,
    current_session_id,
    current_parent_event_id,
)

# Decorators
from .sdk.decorators import event, event as step  # step is deprecated alias

# Error types
from .core.errors import (
    LucidicError,
    LucidicNotInitializedError,
    APIKeyVerificationError,
    InvalidOperationError,
    PromptError,
    FeatureFlagError,
)

# Import functions that need to be implemented
def _update_session(
    task=None,
    session_eval=None,
    session_eval_reason=None,
    is_successful=None,
    is_successful_reason=None
):
    """Update the current session."""
    from .sdk.init import get_resources, get_session_id
    
    session_id = get_session_id()
    if not session_id:
        return
    
    resources = get_resources()
    if resources and 'sessions' in resources:
        updates = {}
        if task is not None:
            updates['task'] = task
        if session_eval is not None:
            updates['session_eval'] = session_eval
        if session_eval_reason is not None:
            updates['session_eval_reason'] = session_eval_reason
        if is_successful is not None:
            updates['is_successful'] = is_successful
        if is_successful_reason is not None:
            updates['is_successful_reason'] = is_successful_reason
            
        if updates:
            resources['sessions'].update_session(session_id, updates)


def _end_session(
    session_eval=None,
    session_eval_reason=None,
    is_successful=None,
    is_successful_reason=None,
    wait_for_flush=True
):
    """End the current session."""
    from .sdk.init import get_resources, get_session_id, get_event_queue
    
    session_id = get_session_id()
    if not session_id:
        return
    
    # Flush events if requested
    if wait_for_flush:
        flush(timeout_seconds=5.0)
    
    # End session via API
    resources = get_resources()
    if resources and 'sessions' in resources:
        resources['sessions'].end_session(
            session_id,
            is_successful=is_successful,
            session_eval=session_eval,
            is_successful_reason=is_successful_reason,
            session_eval_reason=session_eval_reason
        )
    
    # Clear session context
    clear_active_session()


def _get_session():
    """Get the current session object."""
    from .sdk.init import get_session_id
    return get_session_id()


def _create_experiment(
    experiment_name,
    LLM_boolean_evaluators=None,
    LLM_numeric_evaluators=None,
    description=None,
    tags=None,
    api_key=None,
    agent_id=None,
):
    """Create a new experiment."""
    from .sdk.init import get_http
    from .core.config import SDKConfig, get_config
    
    # Get or create HTTP client
    http = get_http()
    config = get_config()
    
    if not http:
        config = SDKConfig.from_env(api_key=api_key, agent_id=agent_id)
        from .api.client import HttpClient
        http = HttpClient(config)
    
    # Use provided agent_id or fall back to config
    final_agent_id = agent_id or config.agent_id
    if not final_agent_id:
        raise ValueError("Agent ID is required for creating experiments")
    
    evaluator_names = []
    if LLM_boolean_evaluators:
        evaluator_names.extend(LLM_boolean_evaluators)
    if LLM_numeric_evaluators:
        evaluator_names.extend(LLM_numeric_evaluators)
    
    # Create experiment via API (matching TypeScript exactly)
    response = http.post('createexperiment', {
        'agent_id': final_agent_id,
        'experiment_name': experiment_name,
        'description': description or '',
        'tags': tags or [],
        'evaluator_names': evaluator_names
    })
    
    return response.get('experiment_id')


def _get_prompt(
    prompt_name,
    variables=None,
    cache_ttl=300,
    label='production'
):
    """Get a prompt from the prompt database."""
    from .sdk.init import get_http
    
    http = get_http()
    if not http:
        return ""
    
    # Get prompt from API
    try:
        response = http.get('getprompt', {
            'prompt_name': prompt_name,
            'label': label
        })
        
        # TypeScript SDK expects 'prompt_content' field
        prompt = response.get('prompt_content', '')
        
        # Replace variables if provided
        if variables:
            for key, value in variables.items():
                prompt = prompt.replace(f"{{{key}}}", str(value))
        
        return prompt
    except Exception:
        return ""


def _get_dataset(dataset_id, api_key=None, agent_id=None):
    """Get a dataset by ID."""
    from .sdk.features.dataset import get_dataset as __get_dataset
    return __get_dataset(dataset_id, api_key, agent_id)


def _get_dataset_items(dataset_id, api_key=None, agent_id=None):
    """Get dataset items."""
    from .sdk.features.dataset import get_dataset_items as __get_dataset_items
    return __get_dataset_items(dataset_id, api_key, agent_id)


def _list_datasets(api_key=None, agent_id=None):
    """List all datasets."""
    from .sdk.features.dataset import list_datasets as __list_datasets
    return __list_datasets(api_key, agent_id)


def _create_dataset(name, description=None, tags=None, suggested_flag_config=None, api_key=None, agent_id=None):
    """Create a new dataset."""
    from .sdk.features.dataset import create_dataset as __create_dataset
    return __create_dataset(name, description, tags, suggested_flag_config, api_key, agent_id)


def _update_dataset(dataset_id, name=None, description=None, tags=None, suggested_flag_config=None, api_key=None, agent_id=None):
    """Update dataset metadata."""
    from .sdk.features.dataset import update_dataset as __update_dataset
    return __update_dataset(dataset_id, name, description, tags, suggested_flag_config, api_key, agent_id)


def _delete_dataset(dataset_id, api_key=None, agent_id=None):
    """Delete a dataset."""
    from .sdk.features.dataset import delete_dataset as __delete_dataset
    return __delete_dataset(dataset_id, api_key, agent_id)


def _create_dataset_item(dataset_id, name, input_data, expected_output=None, description=None, tags=None, metadata=None, flag_overrides=None, api_key=None, agent_id=None):
    """Create a dataset item."""
    from .sdk.features.dataset import create_dataset_item as __create_dataset_item
    return __create_dataset_item(dataset_id, name, input_data, expected_output, description, tags, metadata, flag_overrides, api_key, agent_id)


def _get_dataset_item(dataset_id, item_id, api_key=None, agent_id=None):
    """Get a specific dataset item."""
    from .sdk.features.dataset import get_dataset_item as __get_dataset_item
    return __get_dataset_item(dataset_id, item_id, api_key, agent_id)


def _update_dataset_item(dataset_id, item_id, name=None, input_data=None, expected_output=None, description=None, tags=None, metadata=None, flag_overrides=None, api_key=None, agent_id=None):
    """Update a dataset item."""
    from .sdk.features.dataset import update_dataset_item as __update_dataset_item
    return __update_dataset_item(dataset_id, item_id, name, input_data, expected_output, description, tags, metadata, flag_overrides, api_key, agent_id)


def _delete_dataset_item(dataset_id, item_id, api_key=None, agent_id=None):
    """Delete a dataset item."""
    from .sdk.features.dataset import delete_dataset_item as __delete_dataset_item
    return __delete_dataset_item(dataset_id, item_id, api_key, agent_id)


def _list_dataset_item_sessions(dataset_id, item_id, api_key=None, agent_id=None):
    """List all sessions for a dataset item."""
    from .sdk.features.dataset import list_dataset_item_sessions as __list_dataset_item_sessions
    return __list_dataset_item_sessions(dataset_id, item_id, api_key, agent_id)


# Feature flags
from .sdk.features.feature_flag import (
    get_feature_flag,
    get_bool_flag,
    get_int_flag,
    get_float_flag,
    get_string_flag,
    get_json_flag,
    clear_feature_flag_cache,
)

# Error boundary utilities
is_silent_mode = error_boundary.is_silent_mode
get_error_history = error_boundary.get_error_history
clear_error_history = error_boundary.clear_error_history

# Version
__version__ = "2.0.0"

# Apply error boundary wrapping to all SDK functions
from .sdk.error_boundary import wrap_sdk_function

# Wrap main SDK functions
init = wrap_sdk_function(_init, "init")
get_session_id = wrap_sdk_function(_get_session_id, "init")
clear_state = wrap_sdk_function(_clear_state, "init")
create_event = wrap_sdk_function(_create_event, "event")
create_error_event = wrap_sdk_function(_create_error_event, "event")
flush = wrap_sdk_function(_flush, "event")

# Wrap session functions
update_session = wrap_sdk_function(_update_session, "session")
end_session = wrap_sdk_function(_end_session, "session")
get_session = wrap_sdk_function(_get_session, "session")

# Wrap feature functions
create_experiment = wrap_sdk_function(_create_experiment, "experiment")
get_prompt = wrap_sdk_function(_get_prompt, "prompt")

# Dataset management - complete CRUD
list_datasets = wrap_sdk_function(_list_datasets, "dataset")
create_dataset = wrap_sdk_function(_create_dataset, "dataset")
get_dataset = wrap_sdk_function(_get_dataset, "dataset")
update_dataset = wrap_sdk_function(_update_dataset, "dataset")
delete_dataset = wrap_sdk_function(_delete_dataset, "dataset")

# Dataset item management
create_dataset_item = wrap_sdk_function(_create_dataset_item, "dataset")
get_dataset_item = wrap_sdk_function(_get_dataset_item, "dataset")
update_dataset_item = wrap_sdk_function(_update_dataset_item, "dataset")
delete_dataset_item = wrap_sdk_function(_delete_dataset_item, "dataset")
get_dataset_items = wrap_sdk_function(_get_dataset_items, "dataset")
list_dataset_item_sessions = wrap_sdk_function(_list_dataset_item_sessions, "dataset")

# All exports
__all__ = [
    # Main functions
    'init',
    'get_session_id',
    'clear_state',
    'update_session',
    'end_session',
    'get_session',
    'create_event',
    'create_error_event',
    'flush',
    
    # Decorators
    'event',
    'step',
    
    # Features
    'create_experiment',
    'get_prompt',

    # Dataset management
    'list_datasets',
    'create_dataset',
    'get_dataset',
    'update_dataset',
    'delete_dataset',
    'create_dataset_item',
    'get_dataset_item',
    'update_dataset_item',
    'delete_dataset_item',
    'get_dataset_items',
    'list_dataset_item_sessions',

    # Feature flags
    'get_feature_flag',
    'get_bool_flag',
    'get_int_flag',
    'get_float_flag',
    'get_string_flag',
    'get_json_flag',
    'clear_feature_flag_cache',
    
    # Context management
    'set_active_session',
    'clear_active_session',
    'bind_session',
    'bind_session_async',
    'session',
    'session_async',
    'run_session',
    'run_in_session',
    'current_session_id',
    'current_parent_event_id',
    
    # Error types
    'LucidicError',
    'LucidicNotInitializedError',
    'APIKeyVerificationError',
    'InvalidOperationError',
    'PromptError',
    'FeatureFlagError',
    
    # Error boundary
    'is_silent_mode',
    'get_error_history',
    'clear_error_history',
    
    # Version
    '__version__',
]