from abc import ABC, abstractmethod
from typing import Optional

class BaseProvider(ABC):
    def __init__(self):
        self._provider_name = None
    
    @abstractmethod
    def handle_response(self, response, kwargs, session: Optional = None):
        """Handle responses from the LLM provider"""
        pass
        
    @abstractmethod
    def override(self):
        """Override the provider's API methods"""
        pass
        
    @abstractmethod
    def undo_override(self):
        """Restore original API methods"""
        pass