"""Prompt resource API operations."""
import logging
from typing import Any, Dict, Optional

from ..client import HttpClient

logger = logging.getLogger("Lucidic")


class PromptResource:
    """Handle prompt-related API operations."""

    def __init__(self, http: HttpClient, production: bool = False):
        """Initialize prompt resource.

        Args:
            http: HTTP client instance
            production: Whether to suppress errors in production mode
        """
        self.http = http
        self._production = production

    def get(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]] = None,
        label: str = "production",
    ) -> str:
        """Get a prompt from the prompt database.

        Args:
            prompt_name: Name of the prompt.
            variables: Variables to interpolate into the prompt.
            label: Prompt version label (default: "production").

        Returns:
            The prompt content with variables interpolated.
        """
        try:
            response = self.http.get(
                "getprompt",
                {"prompt_name": prompt_name, "label": label},
            )
            prompt = response.get("prompt_content", "")

            # Replace variables
            if variables:
                for key, value in variables.items():
                    prompt = prompt.replace(f"{{{key}}}", str(value))

            return prompt
        except Exception as e:
            if self._production:
                logger.error(f"[PromptResource] Failed to get prompt: {e}")
                return ""
            raise

    async def aget(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]] = None,
        label: str = "production",
    ) -> str:
        """Get a prompt from the prompt database (asynchronous).

        See get() for full documentation.
        """
        try:
            response = await self.http.aget(
                "getprompt",
                {"prompt_name": prompt_name, "label": label},
            )
            prompt = response.get("prompt_content", "")

            if variables:
                for key, value in variables.items():
                    prompt = prompt.replace(f"{{{key}}}", str(value))

            return prompt
        except Exception as e:
            if self._production:
                logger.error(f"[PromptResource] Failed to get prompt: {e}")
                return ""
            raise
