"""client.usage — org-aggregated usage / quota counters (LUC-911).

A single-endpoint namespace. Data-bearing read — does NOT swallow in production.
"""
from ..client import HttpClient
from ..models.usage import Usage


class UsageResource:
    """Handle for ``GET /sdk/v2/usage``."""

    def __init__(self, http: HttpClient):
        self.http = http

    def get(self) -> Usage:
        """Read the key's org aggregated usage counters (scope ``usage:read``)."""
        return Usage.from_dict(self.http.get("sdk/v2/usage"))

    async def aget(self) -> Usage:
        """Async sibling of ``get``."""
        return Usage.from_dict(await self.http.aget("sdk/v2/usage"))
