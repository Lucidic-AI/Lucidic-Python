"""Prompt API resource handler"""

from typing import Optional, Dict, Any
from lucidicai.client.http_client import HttpClient


class PromptResource:
    """Handles prompt-related API operations"""
    
    def __init__(self, http_client: HttpClient):
        """Initialize prompt resource with HTTP client
        
        Args:
            http_client: HTTP client instance for API requests
        """
        self.http = http_client
        self.cache: Dict[tuple, tuple] = {}  # (prompt_name, label) -> (prompt, expiration)
    
    def get_prompt(
        self,
        agent_id: str,
        prompt_name: str,
        label: str = "production",
        cache_ttl: int = 300
    ) -> str:
        """Get prompt from API with caching
        
        Args:
            agent_id: Agent ID
            prompt_name: Name of the prompt
            label: Prompt label/version
            cache_ttl: Cache time-to-live in seconds (-1 for forever, 0 to disable)
            
        Returns:
            Prompt content string
        """
        import time
        
        # Check cache
        cache_key = (prompt_name, label)
        if cache_key in self.cache:
            prompt, expiration = self.cache[cache_key]
            if expiration == float('inf') or time.time() < expiration:
                return prompt
        
        # Fetch from API
        params = {
            "agent_id": agent_id,
            "prompt_name": prompt_name,
            "label": label
        }
        response = self.http.get("getprompt", params)
        prompt = response.get("prompt_content", "")
        
        # Update cache
        if cache_ttl != 0:
            if cache_ttl == -1:
                expiration = float('inf')
            else:
                expiration = time.time() + cache_ttl
            self.cache[cache_key] = (prompt, expiration)
        
        return prompt
    
    def substitute_variables(self, prompt: str, variables: Dict[str, Any]) -> str:
        """Substitute variables in prompt template
        
        Args:
            prompt: Prompt template with {{variable}} placeholders
            variables: Dictionary of variable values
            
        Returns:
            Prompt with variables substituted
        """
        result = prompt
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        return result