"""Feature flag resource API operations."""
import logging
from typing import Any, Dict, Optional

from ..client import HttpClient

logger = logging.getLogger("Lucidic")


class FeatureFlagResource:
    """Handle feature flag-related API operations."""

    def __init__(
        self,
        http: HttpClient,
        agent_id: Optional[str] = None,
        production: bool = False,
    ):
        """Initialize feature flag resource.

        Args:
            http: HTTP client instance
            agent_id: Default agent ID for feature flags
            production: Whether to suppress errors in production mode
        """
        self.http = http
        self._agent_id = agent_id
        self._production = production

    def get(
        self,
        flag_name: str,
        default: Any = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Get a feature flag value.

        Args:
            flag_name: Name of the feature flag.
            default: Default value if flag is not found.
            context: Optional context for flag evaluation.

        Returns:
            The flag value or default.
        """
        try:
            response = self.http.get(
                "featureflags",
                {"flag_name": flag_name, "agent_id": self._agent_id},
            )
            return response.get("value", default)
        except Exception as e:
            if self._production:
                logger.error(f"[FeatureFlagResource] Failed to get feature flag: {e}")
                return default
            raise

    async def aget(
        self,
        flag_name: str,
        default: Any = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Get a feature flag value (asynchronous).

        See get() for full documentation.
        """
        try:
            response = await self.http.aget(
                "featureflags",
                {"flag_name": flag_name, "agent_id": self._agent_id},
            )
            return response.get("value", default)
        except Exception as e:
            if self._production:
                logger.error(f"[FeatureFlagResource] Failed to get feature flag: {e}")
                return default
            raise
