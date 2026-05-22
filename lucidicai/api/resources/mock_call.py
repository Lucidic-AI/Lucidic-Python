"""Mock call resource — `POST /sdk/mock-call` HTTP layer.

Two public methods per envelope-style preference:

- **`call()` / `acall()`** (new in LUC-607) — thin transport. Returns the
  parsed v2 success body ``{return_value, tier, was_mocked}`` so the
  caller can inspect `was_mocked` and branch. Raises typed
  `LucidicMockCallError` subclasses on non-2xx. Consumed by the internal
  `sdk/tools/transport.py::emit_call_through_backend` helper that backs
  `@mockable` and the framework adapters.

- **`create()` / `acreate()`** (legacy, LUC-483-era) — kept for the
  explicit-mock pattern customers built around M2:

      if os.getenv("LUCIDIC_TEST_MODE"):
          rows = client.mock_calls.create("query_sql", sql=user_query)

  Returns `body["return_value"]` directly (matching the original
  "you get the tool's output" semantic). Logs a WARNING + returns None
  on PASS_THROUGH — getting fallback when you explicitly asked for a
  mock means the tool's dashboard tier is misconfigured.

Backend contract (`api/views/sdk_mock_call.py`, frozen):

- Request body: ``{session_id, tool_name, kwargs, client_event_id?}``
- Success (200): ``{return_value, tier, was_mocked}`` uniform shape across
  all 200 paths (PASS_THROUGH carries ``return_value=null, was_mocked=false``).
- Errors: ``{"error": {"code": str, "detail": str, ...code-specific}}``
  at the corresponding HTTP status. See ``core/errors.py`` for the full
  code→class lookup.
"""
import logging
from typing import Any, Dict, Optional

import httpx

from ..client import HttpClient
from ...core.errors import (
    LucidicMockCallError,
    error_class_for_code,
)

logger = logging.getLogger("Lucidic")


def _truncate_id(id_str: Optional[str]) -> str:
    if not id_str:
        return "None"
    return f"{id_str[:8]}..." if len(id_str) > 8 else id_str


def _exception_from_http_error(exc: httpx.HTTPStatusError) -> LucidicMockCallError:
    """Translate a non-2xx response into the appropriate typed exception.

    Backend envelope (LUC-584) is uniform: ``{"error": {"code": str,
    "detail": str, ...code-specific keys...}}``. Look up the class by
    `code` via ``error_class_for_code`` (returns the base class for
    unknown codes — forward-compat with backend additions).

    Falls back to ``LucidicMockCallError`` with a synthetic detail when
    the body isn't parseable as the documented envelope (truncated
    responses, proxy errors that don't pass the JSON through, etc.).
    """
    try:
        body = exc.response.json()
    except ValueError:
        return LucidicMockCallError(
            code="malformed_response",
            detail=f"HTTP {exc.response.status_code}: {exc.response.text or 'no body'}",
        )

    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict) or "code" not in err:
        return LucidicMockCallError(
            code="malformed_response",
            detail=f"HTTP {exc.response.status_code}: {body!r}",
        )

    code = err["code"]
    detail = err.get("detail", "")
    extra = {k: v for k, v in err.items() if k not in ("code", "detail")}
    cls = error_class_for_code(code)

    # Each typed subclass that adds attributes does so via its own
    # __init__ keyword params. Pass the extra dict through; classes that
    # don't recognize a key still accept it via the base's **extra
    # channel and stash it as an attribute.
    #
    # When `cls is LucidicMockCallError` (unknown code → forward-compat
    # fallback), pass the actual code through so the exception carries
    # the backend's wire value instead of the class-level default.
    try:
        if cls is LucidicMockCallError:
            return cls(detail, code=code, **extra)
        return cls(detail, **extra)
    except TypeError:
        # Defensive: if a subclass added a stricter __init__ in the
        # future and rejects an unknown kwarg, fall back to the base.
        return LucidicMockCallError(code=code, detail=detail, **extra)


class MockCallResource:
    """SDK-side handle for POST /sdk/mock-call.

    Constructed once per `LucidicAI` client (see `client.py:_resources`).
    Thread-safe through the underlying `HttpClient`.
    """

    def __init__(self, http: HttpClient, production: bool = False):
        self.http = http
        # production flag is accepted for parity with other resources;
        # mock_call never silently swallows errors — the response body
        # IS the user's data, not observability, so failures must
        # surface even in production mode.
        self._production = production

    # ==================== call() / acall() — thin transport (new) ====================

    def call(
        self,
        *,
        session_id: str,
        tool_name: str,
        kwargs: Dict[str, Any],
        client_event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dispatch one mock-call and return the parsed v2 success body.

        Always returns a dict with keys ``return_value``, ``tier``,
        ``was_mocked``. Raises a typed ``LucidicMockCallError`` subclass
        on any non-2xx response.

        Unlike ``create()``, ``session_id`` is required (the transport
        layer in ``sdk/tools/transport.py`` resolves it from the bound
        mock context and passes it explicitly). Use ``create()`` for the
        legacy contextvar-lookup behavior.
        """
        body: Dict[str, Any] = {
            "session_id": session_id,
            "tool_name": tool_name,
            "kwargs": kwargs,
        }
        if client_event_id is not None:
            body["client_event_id"] = client_event_id

        logger.debug(
            "[MockCallResource] dispatch tool_name=%r session=%s",
            tool_name, _truncate_id(session_id),
        )

        try:
            return self.http.post("sdk/mock-call", body)
        except httpx.HTTPStatusError as exc:
            raise _exception_from_http_error(exc) from exc

    async def acall(
        self,
        *,
        session_id: str,
        tool_name: str,
        kwargs: Dict[str, Any],
        client_event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of ``call``."""
        body: Dict[str, Any] = {
            "session_id": session_id,
            "tool_name": tool_name,
            "kwargs": kwargs,
        }
        if client_event_id is not None:
            body["client_event_id"] = client_event_id

        logger.debug(
            "[MockCallResource] dispatch (async) tool_name=%r session=%s",
            tool_name, _truncate_id(session_id),
        )

        try:
            return await self.http.apost("sdk/mock-call", body)
        except httpx.HTTPStatusError as exc:
            raise _exception_from_http_error(exc) from exc

    # ==================== create() / acreate() — legacy LUC-483 API ====================

    def create(
        self,
        tool_name: str,
        *,
        session_id: Optional[str] = None,
        client_event_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Legacy explicit-mock dispatch.

        Returns the tool's ``return_value`` directly (matching the
        original LUC-483 "you get the tool's output" semantic). For
        tier=SQL_TEMPLATE the return shape is
        ``{"columns": [...], "rows": [[...]], "row_count": int}``;
        other tiers return whatever the executor produces.

        PASS_THROUGH handling: backend returns ``was_mocked=false`` and
        ``return_value=null`` when the tool's dashboard tier is
        PASS_THROUGH. Legacy callers explicitly asked for a mock, so
        getting fallback is almost certainly a config mismatch — we log
        a WARNING and return None. If you want fallback-with-real-execution
        semantics, use ``@mockable`` instead; the transport layer there
        runs the real function for PASS_THROUGH.

        Args:
            tool_name: Identifier matching ``Tool.name`` for the active
                session's agent.
            session_id: Override the session id from the SDK context.
                Useful for cross-thread or cross-async-task dispatch
                where the ContextVar isn't propagated.
            client_event_id: Override the auto-generated client-side
                event id. Backend uses this as the FUNCTION_CALL event's
                idempotency key.
            **kwargs: Tool-specific arguments matching ``Tool.signature``.

        Returns:
            The tool's return value (any JSON-serializable shape), or
            None when there's no active session or the call hit
            PASS_THROUGH.

        Raises:
            LucidicMockCallError or subclass: any non-2xx from the
                backend. Common subclasses: ``LucidicToolDriftError``,
                ``LucidicUnknownToolError``, ``LucidicUnsupportedSQLError``,
                ``LucidicToolBlockedError``, ``LucidicSessionNotInitializedError``.
        """
        resolved_session_id = self._resolve_session_id(session_id)
        if resolved_session_id is None:
            return None

        body = self.call(
            session_id=resolved_session_id,
            tool_name=tool_name,
            kwargs=dict(kwargs),
            client_event_id=client_event_id,
        )
        return self._unwrap_legacy(tool_name, body)

    async def acreate(
        self,
        tool_name: str,
        *,
        session_id: Optional[str] = None,
        client_event_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Async version of ``create``. See ``create`` for full docs."""
        resolved_session_id = self._resolve_session_id(session_id)
        if resolved_session_id is None:
            return None

        body = await self.acall(
            session_id=resolved_session_id,
            tool_name=tool_name,
            kwargs=dict(kwargs),
            client_event_id=client_event_id,
        )
        return self._unwrap_legacy(tool_name, body)

    # ==================== internals ====================

    def _resolve_session_id(self, session_id: Optional[str]) -> Optional[str]:
        """Resolve session_id from the legacy contextvar fallback.

        Mirrors the LUC-483 behavior: if no session is bound and no
        explicit ``session_id`` was passed, log + short-circuit with
        None rather than hitting the backend with a guaranteed 4xx.
        """
        from ...sdk.context import current_session_id

        resolved = session_id or current_session_id.get(None)
        if not resolved:
            logger.debug("[MockCallResource] No active session — skipping dispatch")
            if not self._production:
                logger.warning(
                    "mock_calls.create() called with no active session and no "
                    "session_id override; returning empty result"
                )
            return None
        return resolved

    def _unwrap_legacy(self, tool_name: str, body: Dict[str, Any]) -> Any:
        """Extract the tool return value for the legacy ``create()`` API.

        Logs a WARNING when the backend served PASS_THROUGH — explicit
        callers shouldn't be hitting that path. Future-proofing: if the
        backend ever omits ``was_mocked``, treat its absence as "mocked"
        for backwards compat (no false-positive warnings on stale
        backends).
        """
        was_mocked = body.get("was_mocked", True)
        if not was_mocked:
            tier = body.get("tier", "?")
            logger.warning(
                "mock_calls.create(%r) hit tier=%s and was not mocked; "
                "return_value is None. Check the tool's tier in the dashboard "
                "or use @mockable for fallback-with-real-execution semantics.",
                tool_name, tier,
            )
        return body.get("return_value")
