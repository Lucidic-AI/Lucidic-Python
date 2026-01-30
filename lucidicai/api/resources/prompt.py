"""Prompt resource API operations."""
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from ..client import HttpClient

if TYPE_CHECKING:
    from ...core.config import SDKConfig

logger = logging.getLogger("Lucidic")


@dataclass
class Prompt:
    """Represents a prompt retrieved from the Lucidic prompt database."""

    raw_content: str
    content: str
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        return self.content

    def replace_variables(self, variables: Dict[str, Any]) -> "Prompt":
        """Replace template variables in the prompt content.

        Replaces {{key}} placeholders in raw_content with the provided
        variable values and updates content.

        Args:
            variables: Dictionary mapping variable names to their values.

        Returns:
            self, for method chaining.
        """
        content = self.raw_content
        for key, value in variables.items():
            content = content.replace(f"{{{{{key}}}}}", str(value))
        self.content = content
        return self


class PromptResource:
    """Handle prompt-related API operations."""

    def __init__(self, http: HttpClient, config: "SDKConfig", production: bool = False):
        """Initialize prompt resource.

        Args:
            http: HTTP client instance
            config: SDK configuration
            production: Whether to suppress errors in production mode
        """
        self.http = http
        self._config = config
        self._production = production
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _is_cache_valid(self, cache_key: Tuple[str, str], cache_ttl: int) -> bool:
        """Check if a cached prompt is still valid.

        Args:
            cache_key: The (prompt_name, label) tuple
            cache_ttl: Cache TTL in seconds (-1 = indefinite, 0 = no cache)

        Returns:
            True if cache is valid, False otherwise
        """
        if cache_ttl == 0:
            return False
        if cache_key not in self._cache:
            return False
        if cache_ttl == -1:
            return True
        cached = self._cache[cache_key]
        return (time.time() - cached["timestamp"]) < cache_ttl

    def get(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]] = None,
        label: str = "production",
        cache_ttl: int = 0,
    ) -> Prompt:
        """Get a prompt from the prompt database.

        Args:
            prompt_name: Name of the prompt.
            variables: Variables to interpolate into the prompt.
            label: Prompt version label (default: "production").
            cache_ttl: Cache TTL in seconds. 0 = no cache, -1 = cache indefinitely,
                       positive value = seconds before refetching.

        Returns:
            A Prompt object with raw_content, content (with variables replaced),
            and metadata. Use str(prompt) for backward-compatible string access.
        """
        try:
            cache_key = (prompt_name, label)

            # Check cache
            if self._is_cache_valid(cache_key, cache_ttl):
                raw_content = self._cache[cache_key]["content"]
                metadata = self._cache[cache_key]["metadata"]
            else:
                response = self.http.get(
                    "getprompt",
                    {"prompt_name": prompt_name, "label": label, "agent_id": self._config.agent_id},
                )
                raw_content = response.get("prompt_content", "")
                metadata = response.get("metadata", {})

                # Store in cache if caching is enabled
                if cache_ttl != 0:
                    self._cache[cache_key] = {
                        "content": raw_content,
                        "metadata": metadata,
                        "timestamp": time.time(),
                    }

            prompt = Prompt(raw_content=raw_content, content=raw_content, metadata=metadata)
            if variables:
                prompt.replace_variables(variables)
            return prompt
        except Exception as e:
            if self._production:
                logger.error(f"[PromptResource] Failed to get prompt: {e}")
                return Prompt(raw_content="", content="", metadata={})
            raise

    async def aget(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]] = None,
        label: str = "production",
        cache_ttl: int = 0,
    ) -> Prompt:
        """Get a prompt from the prompt database (asynchronous).

        See get() for full documentation.
        """
        try:
            cache_key = (prompt_name, label)

            # Check cache
            if self._is_cache_valid(cache_key, cache_ttl):
                raw_content = self._cache[cache_key]["content"]
                metadata = self._cache[cache_key]["metadata"]
            else:
                response = await self.http.aget(
                    "getprompt",
                    {"prompt_name": prompt_name, "label": label, "agent_id": self._config.agent_id},
                )
                raw_content = response.get("prompt_content", "")
                metadata = response.get("metadata", {})

                # Store in cache if caching is enabled
                if cache_ttl != 0:
                    self._cache[cache_key] = {
                        "content": raw_content,
                        "metadata": metadata,
                        "timestamp": time.time(),
                    }

            prompt = Prompt(raw_content=raw_content, content=raw_content, metadata=metadata)
            if variables:
                prompt.replace_variables(variables)
            return prompt
        except Exception as e:
            if self._production:
                logger.error(f"[PromptResource] Failed to get prompt: {e}")
                return Prompt(raw_content="", content="", metadata={})
            raise
