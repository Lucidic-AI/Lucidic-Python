"""Prompt resource API operations."""
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

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

    def _invalidate_cache(self, prompt_name: str, label: Optional[str] = None) -> None:
        """Invalidate cached prompt entries.

        Args:
            prompt_name: Name of the prompt to invalidate.
            label: If provided, only invalidate the specific (prompt_name, label) entry.
                   If None, invalidate all entries matching prompt_name.
        """
        if label is not None:
            self._cache.pop((prompt_name, label), None)
        else:
            keys_to_remove = [k for k in self._cache if k[0] == prompt_name]
            for k in keys_to_remove:
                del self._cache[k]

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
                    "sdk/prompts",
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
                    "sdk/prompts",
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

    def update(
        self,
        prompt_name: str,
        prompt_content: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        labels: Optional[List[str]] = None,
    ) -> Prompt:
        """Update a prompt, creating a new immutable version.

        Args:
            prompt_name: Name of the prompt to update.
            prompt_content: New content for the prompt.
            description: Optional description for the prompt version.
            metadata: Optional metadata dict to attach to the prompt version.
            labels: Optional list of labels to assign to the new version.

        Returns:
            A Prompt object with the new content and metadata from the response.
        """
        try:
            body: Dict[str, Any] = {
                "agent_id": self._config.agent_id,
                "prompt_name": prompt_name,
                "prompt_content": prompt_content,
            }
            if description is not None:
                body["description"] = description
            if metadata is not None:
                body["metadata"] = metadata
            if labels is not None:
                body["labels"] = labels

            response = self.http.put("sdk/prompts", data=body)
            response_metadata = response.get("metadata", {})

            self._invalidate_cache(prompt_name)

            return Prompt(raw_content=prompt_content, content=prompt_content, metadata=response_metadata)
        except Exception as e:
            if self._production:
                logger.error(f"[PromptResource] Failed to update prompt: {e}")
                return Prompt(raw_content="", content="", metadata={})
            raise

    async def aupdate(
        self,
        prompt_name: str,
        prompt_content: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        labels: Optional[List[str]] = None,
    ) -> Prompt:
        """Update a prompt, creating a new immutable version (asynchronous).

        See update() for full documentation.
        """
        try:
            body: Dict[str, Any] = {
                "agent_id": self._config.agent_id,
                "prompt_name": prompt_name,
                "prompt_content": prompt_content,
            }
            if description is not None:
                body["description"] = description
            if metadata is not None:
                body["metadata"] = metadata
            if labels is not None:
                body["labels"] = labels

            response = await self.http.aput("sdk/prompts", data=body)
            response_metadata = response.get("metadata", {})

            self._invalidate_cache(prompt_name)

            return Prompt(raw_content=prompt_content, content=prompt_content, metadata=response_metadata)
        except Exception as e:
            if self._production:
                logger.error(f"[PromptResource] Failed to update prompt: {e}")
                return Prompt(raw_content="", content="", metadata={})
            raise

    def update_metadata(
        self,
        prompt_name: str,
        label: str,
        metadata: Dict[str, Any],
    ) -> Prompt:
        """Update metadata on an existing prompt version.

        Sends a PATCH request to update only the metadata for the prompt version
        identified by (prompt_name, label). The prompt content is not returned
        by this endpoint, so the returned Prompt will have empty content fields.

        Args:
            prompt_name: Name of the prompt.
            label: Label identifying the prompt version to update.
            metadata: Metadata dict to set on the prompt version.

        Returns:
            A Prompt object with empty content and the updated metadata.
        """
        try:
            body: Dict[str, Any] = {
                "agent_id": self._config.agent_id,
                "prompt_name": prompt_name,
                "label": label,
                "metadata": metadata,
            }

            response = self.http.patch("sdk/prompts", data=body)
            response_metadata = response.get("metadata", {})

            self._invalidate_cache(prompt_name, label)

            return Prompt(raw_content="", content="", metadata=response_metadata)
        except Exception as e:
            if self._production:
                logger.error(f"[PromptResource] Failed to update prompt metadata: {e}")
                return Prompt(raw_content="", content="", metadata={})
            raise

    async def aupdate_metadata(
        self,
        prompt_name: str,
        label: str,
        metadata: Dict[str, Any],
    ) -> Prompt:
        """Update metadata on an existing prompt version (asynchronous).

        See update_metadata() for full documentation.
        """
        try:
            body: Dict[str, Any] = {
                "agent_id": self._config.agent_id,
                "prompt_name": prompt_name,
                "label": label,
                "metadata": metadata,
            }

            response = await self.http.apatch("sdk/prompts", data=body)
            response_metadata = response.get("metadata", {})

            self._invalidate_cache(prompt_name, label)

            return Prompt(raw_content="", content="", metadata=response_metadata)
        except Exception as e:
            if self._production:
                logger.error(f"[PromptResource] Failed to update prompt metadata: {e}")
                return Prompt(raw_content="", content="", metadata={})
            raise
