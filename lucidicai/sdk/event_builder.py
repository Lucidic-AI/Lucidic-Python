"""Event builder for flexible parameter handling and field normalization.

Inspired by TypeScript SDK's EventBuilder, this module provides a clean way
to build events from various parameter formats and normalize field names.
"""
from typing import Any, Dict, Optional, List, Union
from datetime import datetime, timezone


# field mappings from TypeScript SDK
FIELD_MAPPINGS = {
    'completion': 'output',
    'response': 'output',
    'prompt': 'messages',
    'functionName': 'function_name',
    'args': 'arguments',
    'returnValue': 'return_value',
    'result': 'return_value',
    'stack': 'traceback',
    'stackTrace': 'traceback',
    'exception': 'error',
    'description': 'details',
    'message': 'details',
}


class EventBuilder:
    """Build events from flexible parameters with field normalization."""
    
    # field sets for different event types
    BASE_FIELDS = {
        'type', 'event_id', 'parent_event_id', 'occurred_at',
        'duration', 'tags', 'metadata'
    }
    
    LLM_FIELDS = {
        'provider', 'model', 'messages', 'prompt', 'output', 'completion',
        'response', 'input_tokens', 'output_tokens', 'cache', 'cost',
        'tool_calls', 'thinking', 'status', 'error', 'raw', 'params'
    }
    
    FUNCTION_FIELDS = {
        'function_name', 'functionName', 'arguments', 'args', 
        'return_value', 'returnValue', 'result'
    }
    
    ERROR_FIELDS = {
        'error', 'traceback', 'stack', 'stackTrace', 'exception'
    }
    
    GENERIC_FIELDS = {
        'details', 'description', 'message', 'data'
    }
    
    @classmethod
    def build(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build a normalized event from flexible parameters.
        
        Args:
            params: Flexible event parameters
            
        Returns:
            Normalized event request dictionary
        """
        # check if already in strict format
        if cls._is_strict_format(params):
            return params
        
        # normalize field names
        normalized = cls._normalize_fields(params)
        
        # detect event type if not provided
        event_type = normalized.get('type') or cls._detect_type(normalized)
        
        # build event based on type
        if event_type == 'llm_generation':
            return cls._build_llm_event(normalized)
        elif event_type == 'function_call':
            return cls._build_function_event(normalized)
        elif event_type == 'error_traceback':
            return cls._build_error_event(normalized)
        else:
            return cls._build_generic_event(normalized)
    
    @classmethod
    def _is_strict_format(cls, params: Dict[str, Any]) -> bool:
        """Check if params are already in strict format."""
        return 'payload' in params and isinstance(params.get('payload'), dict)
    
    @classmethod
    def _normalize_fields(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize field names using mappings."""
        normalized = {}
        for key, value in params.items():
            canonical = FIELD_MAPPINGS.get(key, key)
            normalized[canonical] = value
        return normalized
    
    @classmethod
    def _detect_type(cls, params: Dict[str, Any]) -> str:
        """Detect event type from parameters."""
        # llm generation indicators
        if any(key in params for key in ['provider', 'model', 'messages', 'prompt', 
                                          'input_tokens', 'output_tokens']):
            return 'llm_generation'
        
        # function call indicators
        if 'function_name' in params or ('arguments' in params and 'error' not in params):
            return 'function_call'
        
        # error traceback indicators
        if any(key in params for key in ['error', 'traceback', 'stack', 'exception']):
            return 'error_traceback'
        
        return 'generic'
    
    @classmethod
    def _extract_base_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract base event parameters."""
        base = {}
        
        # map common base fields
        if 'event_id' in params:
            base['client_event_id'] = params['event_id']
        if 'parent_event_id' in params:
            base['client_parent_event_id'] = params['parent_event_id']
        if 'session_id' in params:
            base['session_id'] = params['session_id']
        if 'occurred_at' in params:
            base['occurred_at'] = params['occurred_at']
        else:
            base['occurred_at'] = datetime.now(timezone.utc).isoformat()
        if 'duration' in params:
            base['duration'] = params['duration']
        if 'tags' in params:
            base['tags'] = params['tags']
        if 'metadata' in params:
            base['metadata'] = params['metadata']

        return base
    
    @classmethod
    def _extract_misc_fields(cls, params: Dict[str, Any], known_fields: set) -> Optional[Dict[str, Any]]:
        """Extract miscellaneous fields not in known sets."""
        misc = {}
        all_known = known_fields | cls.BASE_FIELDS
        
        for key, value in params.items():
            if key not in all_known and value is not None:
                misc[key] = value
        
        return misc if misc else None
    
    @classmethod
    def _build_llm_event(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build LLM generation event."""
        base = cls._extract_base_params(params)
        base['type'] = 'llm_generation'
        
        # build payload
        payload = {
            'request': {
                'provider': params.get('provider', 'unknown'),
                'model': params.get('model', 'unknown'),
            },
            'response': {},
            'usage': {}
        }
        
        # handle messages/prompt
        if 'messages' in params:
            payload['request']['messages'] = params['messages']
        elif 'prompt' in params:
            # convert prompt to messages format
            payload['request']['messages'] = [{'role': 'user', 'content': params['prompt']}]
        
        # request params
        if 'params' in params:
            payload['request']['params'] = params['params']
        
        # response fields
        if 'output' in params:
            payload['response']['output'] = params['output']
        if 'tool_calls' in params:
            payload['response']['tool_calls'] = params['tool_calls']
        if 'thinking' in params:
            payload['response']['thinking'] = params['thinking']
        if 'raw' in params:
            payload['response']['raw'] = params['raw']
        
        # usage fields
        if 'input_tokens' in params:
            payload['usage']['input_tokens'] = params['input_tokens']
        if 'output_tokens' in params:
            payload['usage']['output_tokens'] = params['output_tokens']
        if 'cache' in params:
            payload['usage']['cache'] = params['cache']
        if 'cost' in params:
            payload['usage']['cost'] = params['cost']
        
        # status and error
        if 'status' in params:
            payload['status'] = params['status']
        if 'error' in params:
            payload['error'] = str(params['error'])
        
        # misc fields
        misc = cls._extract_misc_fields(params, cls.LLM_FIELDS)
        if misc:
            payload['misc'] = misc
        
        base['payload'] = payload
        return base
    
    @classmethod
    def _build_function_event(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build function call event."""
        base = cls._extract_base_params(params)
        base['type'] = 'function_call'
        
        payload = {
            'function_name': params.get('function_name', 'unknown')
        }
        
        if 'arguments' in params:
            payload['arguments'] = params['arguments']
        if 'return_value' in params:
            payload['return_value'] = params['return_value']
        if 'error' in params:
            payload['error'] = params['error']
        
        # misc fields
        misc = cls._extract_misc_fields(params, cls.FUNCTION_FIELDS)
        if misc:
            payload['misc'] = misc
        
        base['payload'] = payload
        return base
    
    @classmethod
    def _build_error_event(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build error traceback event."""
        base = cls._extract_base_params(params)
        base['type'] = 'error_traceback'
        
        # handle error
        error_str = ''
        traceback_str = ''
        
        error_val = params.get('error')
        if isinstance(error_val, Exception):
            error_str = str(error_val)
            import traceback
            traceback_str = traceback.format_exc()
        else:
            error_str = str(error_val or 'Unknown error')
            traceback_str = params.get('traceback', '')
        
        payload = {
            'error': error_str,
            'traceback': traceback_str
        }
        
        # context
        if 'context' in params:
            payload['context'] = params['context']
        
        # misc fields
        misc = cls._extract_misc_fields(params, cls.ERROR_FIELDS)
        if misc:
            payload['misc'] = misc
        
        base['payload'] = payload
        return base
    
    @classmethod
    def _build_generic_event(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build generic event."""
        base = cls._extract_base_params(params)
        base['type'] = params.get('type', 'generic')
        
        # build payload from non-base fields
        payload = {}
        
        # common generic fields
        if 'details' in params:
            payload['details'] = params['details']
        elif 'data' in params:
            payload['data'] = params['data']
        
        # include all other fields in payload
        for key, value in params.items():
            if key not in cls.BASE_FIELDS and key not in payload:
                payload[key] = value
        
        # misc fields
        misc = cls._extract_misc_fields(params, cls.GENERIC_FIELDS)
        if misc:
            payload['misc'] = misc
        
        base['payload'] = payload
        return base