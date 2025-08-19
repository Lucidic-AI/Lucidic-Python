"""Compatibility layer for telemetry modules

This file provides backward compatibility for telemetry modules that still
import Client. This should be removed once telemetry is fully refactored.
"""

from lucidicai.sdk.state import SDKState
from lucidicai.util.singleton import singleton


@singleton  
class Client:
    """Legacy Client wrapper for backward compatibility"""
    
    def __init__(self, api_key=None, agent_id=None):
        """Initialize compatibility client"""
        self._state = SDKState()
        if api_key:
            self._state.api_key = api_key
        if agent_id:
            self._state.agent_id = agent_id
    
    @property
    def agent_id(self):
        return self._state.agent_id
    
    @agent_id.setter
    def agent_id(self, value):
        self._state.agent_id = value
    
    @property
    def api_key(self):
        return self._state.api_key
    
    @property
    def session(self):
        # Return a mock session object if needed
        if self._state.session_id:
            return type('Session', (), {
                'session_id': self._state.session_id,
                'active_step': None
            })()
        return None
    
    @property
    def masking_function(self):
        return self._state.masking_function
    
    @masking_function.setter
    def masking_function(self, value):
        self._state.masking_function = value
    
    def mask(self, data):
        return self._state.mask(data)
    
    @property
    def initialized(self):
        return self._state.is_initialized()
    
    @property
    def providers(self):
        return self._state.providers
    
    def set_provider(self, provider):
        """Legacy provider setup"""
        if provider not in self._state.providers:
            self._state.providers.append(provider)
            provider.override()
    
    def clear(self):
        """Legacy clear method"""
        self._state.reset()