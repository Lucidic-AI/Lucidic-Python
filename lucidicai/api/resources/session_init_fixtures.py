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

from ..client import HttpClient

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

        Raises a typed ``LucidicError`` (e.g. ``NotFoundError`` on a wrong /
        finished session_id, ``ServiceUnavailableError`` after the transport
        exhausts its 503 retry budget) on non-2xx. The caller in
        ``SessionResource.create()`` already swallows + warns, so these don't
        surface to the user.
        """
        body = {"session_id": session_id}
        logger.debug(
            "[SessionInitFixturesResource] init session %s",
            session_id[:8] + "..." if len(session_id) > 8 else session_id,
        )
        return self.http.post("sdk/session-init-fixtures", body)

    async def ainit(self, session_id: str) -> Dict[str, Any]:
        """Async sibling of ``init``."""
        body = {"session_id": session_id}
        return await self.http.apost("sdk/session-init-fixtures", body)
