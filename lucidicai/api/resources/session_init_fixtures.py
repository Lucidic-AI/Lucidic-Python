"""Session-init-fixtures resource — ``POST /sdk/session-init-fixtures``.

Backend prerequisite for any subsequent ``mock_call`` against a
tool-backed session: materializes the per-session DuckDB file (LUC-574)
and writes ``Session.tool_version_snapshot`` (LUC-585). Without this
the mock_call view returns 400 ``session_not_initialized``.

The backend is gracefully tolerant of non-tool-backed sessions:
returns ``{"initialized": False, "reason": "..."}`` when the session
has no DatasetItem. The SDK reads ``initialized`` to decide whether to
establish a ``MockContext`` for the session — see
``sdk/tools/resource.py::ToolsResource._init_session``.

Backend contract (``api/views/sdk_session_fixtures.py``):

- Request body: ``{session_id}``
- 200: ``{initialized: bool, reason?: str, fixture_ids: [...],
  tools_snapshotted: [...], already_initialized: bool}``
- 400: ``{error: <str>}`` for malformed input or session lookup failure
- 404: ``{error: "Specified session not found"}``
- 503: ``{error: <str>}`` couldn't acquire init lock — retry with backoff
"""
import logging
from typing import Any, Dict

import httpx

from ..client import HttpClient
from ...core.errors import LucidicError

logger = logging.getLogger("Lucidic")


class SessionInitFixturesResource:
    """Thin wrapper over ``POST /sdk/session-init-fixtures``."""

    def __init__(self, http: HttpClient):
        self.http = http

    def init(self, session_id: str) -> Dict[str, Any]:
        """Initialize per-session fixture state + tool version snapshot.

        Returns the parsed body. Caller inspects ``initialized``:
        True → session is tool-backed and ready for mock_call; False →
        no-op (session has no DatasetItem or no Resources/tools).

        Raises ``LucidicError`` on 4xx/5xx with the backend's error
        string. The session_not_found path (404) is a real bug — the
        session_id was wrong or already finished — but we surface it
        as a plain ``LucidicError`` since the caller in
        ``SessionResource.create()`` already swallows + warns.
        """
        body = {"session_id": session_id}
        logger.debug(
            "[SessionInitFixturesResource] init session %s",
            session_id[:8] + "..." if len(session_id) > 8 else session_id,
        )
        try:
            return self.http.post("sdk/session-init-fixtures", body)
        except httpx.HTTPStatusError as exc:
            raise LucidicError(_format_init_error(exc)) from exc

    async def ainit(self, session_id: str) -> Dict[str, Any]:
        """Async sibling of ``init``."""
        body = {"session_id": session_id}
        try:
            return await self.http.apost("sdk/session-init-fixtures", body)
        except httpx.HTTPStatusError as exc:
            raise LucidicError(_format_init_error(exc)) from exc


def _format_init_error(exc: httpx.HTTPStatusError) -> str:
    """Best-effort human-readable error from a non-2xx response."""
    try:
        body = exc.response.json()
    except ValueError:
        return f"HTTP {exc.response.status_code}: {exc.response.text or 'no body'}"
    if isinstance(body, dict) and "error" in body:
        return f"HTTP {exc.response.status_code}: {body['error']}"
    return f"HTTP {exc.response.status_code}: {body!r}"
