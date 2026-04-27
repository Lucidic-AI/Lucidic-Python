"""Mock call resource — runtime fixture-backed tool call dispatch (LUC-483).

The backend (`POST /sdk/mock-call`, LUC-481) resolves the active session through
its DatasetItem → Dataset → Resource → Fixture, runs the `tool_name` against
the fixture, and returns the result. This resource is the SDK-side counterpart.

Typical usage in a client agent:

    if os.getenv("LUCIDIC_TEST_MODE"):
        rows = client.mock_calls.create("query_sql", sql=user_query)
    else:
        rows = self.db.execute(user_query)

The `create_or_none` variant returns `None` instead of raising on a 422
(unsupported SQL) — useful for agents that prefer sentinel handling over
exceptions.
"""
import logging
from typing import Any, Dict, Optional

import httpx

from ..client import HttpClient
from ...core.errors import LucidicError, LucidicUnsupportedSQLError

logger = logging.getLogger("Lucidic")


def _truncate_id(id_str: Optional[str]) -> str:
    if not id_str:
        return "None"
    return f"{id_str[:8]}..." if len(id_str) > 8 else id_str


def _parse_unsupported_sql(exc: httpx.HTTPStatusError) -> Optional[LucidicUnsupportedSQLError]:
    """If the response is a 422 with the documented `unsupported_sql` envelope,
    build a typed exception. Otherwise return None so the caller can fall
    through to a generic LucidicError.

    The backend contract (LUC-481) guarantees:
        HTTP 422 → {"error": "unsupported_sql", "detail": str, "source_dialect": str}
    """
    if exc.response.status_code != 422:
        return None
    try:
        body = exc.response.json()
    except ValueError:
        return None
    if not isinstance(body, dict) or body.get("error") != "unsupported_sql":
        return None
    return LucidicUnsupportedSQLError(
        detail=body.get("detail", ""),
        source_dialect=body.get("source_dialect", ""),
    )


def _server_error_message(exc: httpx.HTTPStatusError) -> str:
    """Best-effort extraction of a human-readable error message from a non-422
    HTTP error. Falls back to the raw status code if the body isn't JSON-shaped.
    """
    try:
        body = exc.response.json()
        if isinstance(body, dict):
            for key in ("error", "detail", "message"):
                if isinstance(body.get(key), str):
                    return body[key]
    except ValueError:
        pass
    return f"HTTP {exc.response.status_code}: {exc.response.text or 'no body'}"


class MockCallResource:
    """Handle SDK mock-call dispatch against fixture-backed datasets."""

    def __init__(self, http: HttpClient, production: bool = False):
        self.http = http
        # production flag is accepted for parity with other resources; mock_call
        # never silently swallows errors — the response body IS the user's data,
        # not observability, so failures must surface even in production mode.
        self._production = production

    # ==================== Sync ====================

    def create(
        self,
        tool_name: str,
        *,
        session_id: Optional[str] = None,
        client_event_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Dispatch a single mock call against the active session's fixture.

        Args:
            tool_name: Identifier matching `Resource.tool_name` on the backend.
            session_id: Override the session id pulled from context. Useful for
                cross-thread or cross-async-task dispatch where the ContextVar
                isn't propagated.
            client_event_id: Override the auto-generated client-side event id.
                Lets the SDK retry idempotently — backend uses this as the
                FUNCTION_CALL event's idempotency key.
            **kwargs: Handler-specific arguments. For `SQLHandler` (the only
                handler in M2), pass `sql="SELECT ..."`.

        Returns:
            Handler output. For SQL: `{"columns": [...], "rows": [[...]],
            "row_count": int}`.

        Raises:
            LucidicUnsupportedSQLError: backend returned HTTP 422 (parse,
                transpile, READ_ONLY mutation, or oversize result).
            LucidicError: any other 4xx/5xx (missing session, no fixture,
                ambiguous tool_name, server error, etc.).
        """
        body = self._build_body(tool_name, session_id, client_event_id, kwargs)
        if body is None:
            return {}

        try:
            return self.http.post("sdk/mock-call", body)
        except httpx.HTTPStatusError as exc:
            unsupported = _parse_unsupported_sql(exc)
            if unsupported is not None:
                raise unsupported from exc
            raise LucidicError(_server_error_message(exc)) from exc

    def create_or_none(
        self,
        tool_name: str,
        *,
        session_id: Optional[str] = None,
        client_event_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Same as `create` but returns `None` instead of raising on HTTP 422.

        Useful for agents that prefer sentinel handling over exceptions:

            rows = client.mock_calls.create_or_none("query_sql", sql=user_sql)
            if rows is None:
                return "I can't run that query against the test fixture."
            return summarize(rows)

        Other 4xx/5xx errors still raise `LucidicError` — the silent path is
        only for "this SQL isn't supported," not for misconfiguration.
        """
        try:
            return self.create(
                tool_name,
                session_id=session_id,
                client_event_id=client_event_id,
                **kwargs,
            )
        except LucidicUnsupportedSQLError:
            return None

    # ==================== Async ====================

    async def acreate(
        self,
        tool_name: str,
        *,
        session_id: Optional[str] = None,
        client_event_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async version of `create`. See `create` for full documentation."""
        body = self._build_body(tool_name, session_id, client_event_id, kwargs)
        if body is None:
            return {}

        try:
            return await self.http.apost("sdk/mock-call", body)
        except httpx.HTTPStatusError as exc:
            unsupported = _parse_unsupported_sql(exc)
            if unsupported is not None:
                raise unsupported from exc
            raise LucidicError(_server_error_message(exc)) from exc

    async def acreate_or_none(
        self,
        tool_name: str,
        *,
        session_id: Optional[str] = None,
        client_event_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Async version of `create_or_none`."""
        try:
            return await self.acreate(
                tool_name,
                session_id=session_id,
                client_event_id=client_event_id,
                **kwargs,
            )
        except LucidicUnsupportedSQLError:
            return None

    # ==================== Internals ====================

    def _build_body(
        self,
        tool_name: str,
        session_id: Optional[str],
        client_event_id: Optional[str],
        kwargs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Resolve session_id from context if not given, build the request body.

        Returns None when there's no active session — the caller short-circuits
        with an empty result rather than hitting the backend with a guaranteed
        4xx. Mirrors EvalsResource.emit's "no session, no-op" behavior.
        """
        from ...sdk.context import current_session_id

        resolved_session_id = session_id or current_session_id.get(None)
        if not resolved_session_id:
            logger.debug("[MockCallResource] No active session — skipping dispatch")
            if not self._production:
                logger.warning(
                    "mock_calls.create() called with no active session and no "
                    "session_id override; returning empty result"
                )
            return None

        logger.debug(
            f"[MockCallResource] dispatch tool_name={tool_name!r} "
            f"session={_truncate_id(resolved_session_id)}"
        )
        body: Dict[str, Any] = {
            "session_id": resolved_session_id,
            "tool_name": tool_name,
            "kwargs": kwargs,
        }
        if client_event_id is not None:
            body["client_event_id"] = client_event_id
        return body
