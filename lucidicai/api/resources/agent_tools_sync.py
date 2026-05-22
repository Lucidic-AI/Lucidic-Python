"""Agent-tools-sync resource — ``POST /sdk/agent-tools/sync``.

Sister of ``MockCallResource``. Consumed by ``ToolsResource.sync()``
(LUC-608) to push the captured tool registry to the backend so the
dashboard's Tools tab (LUC-586) and the ``mock_call`` dispatch
(LUC-584) have the latest source_hash + signature + docstring per tool.

Backend contract (``api/views/sdk_agent_tools.py``, frozen):

- Request body: ``{agent_id, tools: [{name, signature{params, return_type},
  docstring, return_shape, source_hash}, ...]}``
- 200: ``{synced: true, stats: {...}}``
- 422: ``{errors: {...}}`` when Tool.clean rejects a payload (non-identifier
  name, malformed source_hash, signature shape mismatch, etc.). The
  service rolls back the whole batch — partial syncs aren't a thing.
- 503: ``{error: str}`` when the per-agent sync lock can't be acquired
  within the backend's timeout. SDK should back off + retry.
"""
import logging
from typing import Any, Dict, List

import httpx

from ..client import HttpClient
from ...core.errors import LucidicError

logger = logging.getLogger("Lucidic")


class SyncAgentToolsResource:
    """Thin wrapper over ``POST /sdk/agent-tools/sync``.

    Constructed once per ``LucidicAI`` (held by ``ToolsResource``).
    Doesn't know about ``ToolSurface`` — caller (``ToolsResource.sync``)
    converts the registry into the wire shape before calling here. Keeps
    this module HTTP-only.
    """

    def __init__(self, http: HttpClient):
        self.http = http

    def sync(
        self,
        *,
        agent_id: str,
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """POST the tool catalog and return the backend's stats.

        Returns the parsed body on 200 — typically
        ``{"synced": True, "stats": {"created": int, "updated": int,
        "drift_count": int, ...}}`` (exact shape is informational; the
        SDK only checks synced=True).

        Raises ``LucidicError`` on any non-2xx. The validation-failure
        path (422) carries a structured error body in
        ``exc.response.json()["errors"]`` — surfaced inline in the
        exception message so callers see the bad payload field without
        manual parsing.
        """
        body = {"agent_id": agent_id, "tools": tools}
        logger.debug(
            "[SyncAgentToolsResource] syncing %d tool(s) for agent %s",
            len(tools), agent_id[:8] + "...",
        )
        try:
            return self.http.post("sdk/agent-tools/sync", body)
        except httpx.HTTPStatusError as exc:
            raise LucidicError(_format_sync_error(exc)) from exc

    async def async_sync(
        self,
        *,
        agent_id: str,
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Async sibling of ``sync``."""
        body = {"agent_id": agent_id, "tools": tools}
        try:
            return await self.http.apost("sdk/agent-tools/sync", body)
        except httpx.HTTPStatusError as exc:
            raise LucidicError(_format_sync_error(exc)) from exc


def _format_sync_error(exc: httpx.HTTPStatusError) -> str:
    """Best-effort human-readable error from a non-2xx /sdk/agent-tools/sync.

    422 carries ``{"errors": {<field>: [...messages]}}`` from
    Tool.clean; 503 carries ``{"error": <str>}``; other shapes fall
    back to status + raw body.
    """
    try:
        body = exc.response.json()
    except ValueError:
        return f"HTTP {exc.response.status_code}: {exc.response.text or 'no body'}"
    if isinstance(body, dict):
        if "errors" in body:
            return f"HTTP {exc.response.status_code} validation: {body['errors']}"
        if "error" in body:
            return f"HTTP {exc.response.status_code}: {body['error']}"
    return f"HTTP {exc.response.status_code}: {body!r}"
