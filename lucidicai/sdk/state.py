"""SDK state management"""

from typing import Optional, Dict, Any, Callable, List
from lucidicai.util.singleton import singleton
from lucidicai.util.cache import LRUCache
from lucidicai.util.logger import logger


@singleton
class SDKState:
    """Central state management for Lucidic SDK"""
    
    def __init__(self):
        """Initialize SDK state"""
        self.initialized = False
        self.session_id: Optional[str] = None
        self.agent_id: Optional[str] = None
        self.api_key: Optional[str] = None
        self.masking_function: Optional[Callable[[str], str]] = None
        self.auto_end: bool = True
        self.providers: List[Any] = []
        
        # Caching
        self.previous_sessions = LRUCache(500)
        self.custom_session_id_translations = LRUCache(500)
        self.prompt_cache: Dict[tuple, tuple] = {}
        
        # HTTP client reference
        self.http_client = None
        
        # Session reference
        self.session = None
    
    def is_initialized(self) -> bool:
        """Check if SDK is initialized"""
        return self.initialized
    
    def reset(self):
        """Reset SDK state"""
        self.initialized = False
        self.session_id = None
        self.session = None
        self.providers.clear()
        self.http_client = None
        logger.info("SDK state reset")
    
    def mask(self, data: Any) -> Any:
        """Apply masking function to data
        
        Args:
            data: Data to mask
            
        Returns:
            Masked data or original if no masking function
        """
        if not self.masking_function or not data:
            return data
        
        try:
            return self.masking_function(data)
        except Exception as e:
            logger.error(f"Error in masking function: {e}")
            return "<Error in masking function>"