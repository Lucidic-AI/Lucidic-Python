from typing import Any, Optional, Type
import sys
import traceback


class LucidicError(Exception):
    """Base exception for all Lucidic SDK errors"""
    pass


class WaitTimeout(LucidicError):
    """Raised by the ``wait_for`` / ``await_for`` poll helpers (LUC-920) when a
    trigger-then-poll workflow does not reach a terminal state before the deadline.

    ``last_state`` is the most recently polled state (so a caller can still inspect
    partial progress); ``timeout`` is the budget in seconds that elapsed.
    """

    def __init__(self, last_state: Any = None, timeout: Optional[float] = None):
        self.last_state = last_state
        self.timeout = timeout
        detail = f" within {timeout:g}s" if timeout is not None else ""
        super().__init__(f"Polling did not reach a terminal state{detail}.")


# ---------------------------------------------------------------------------
# HTTP / API errors (LUC-900)
# ---------------------------------------------------------------------------
#
# Every non-2xx response from the Lucidic API is decoded once, centrally, in
# ``HttpClient._handle_response`` and raised as one of the typed exceptions
# below (all subclasses of ``LucidicError``, so ``except LucidicError`` still
# catches everything). The backend error envelope is one of:
#
#   {"error": "<message>"}                       -> message
#   {"error": "Validation failed", "details": …} -> message + details
#   {"errors": {<field>: [...]}}                 -> ValidationError(details=…)
#   {"detail": "<message>"}                      -> message (DRF default)
#   {"error": {"code": str, ...}}                -> mock-call family (below)
#
# ``exception_from_response`` does the decoding; ``_STATUS_TO_CLASS`` maps
# status codes to the generic classes.


class LucidicAPIError(LucidicError):
    """Base for a typed non-2xx response from the Lucidic API.

    Carries the HTTP ``status_code``, the decoded ``response_body`` (when the
    body was JSON), the raw ``response_text``, and any field-level ``details``
    so callers can inspect a failure without re-parsing the response.
    """

    # Subclasses pin their canonical status; the base leaves it None.
    status_code: Optional[int] = None

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        details: Any = None,
        response_body: Any = None,
        response_text: Optional[str] = None,
    ):
        if status_code is not None:
            self.status_code = status_code
        self.details = details
        self.response_body = response_body
        self.response_text = response_text
        super().__init__(message)


class ValidationError(LucidicAPIError):
    """400 / 422 — invalid input. ``details`` carries field-level errors when
    the backend returns ``{"errors": {...}}`` or ``{"details": {...}}``."""
    status_code = 400


class AuthError(LucidicAPIError):
    """401 — missing or invalid API key."""
    status_code = 401


class APIKeyVerificationError(AuthError):
    """Exception for API key verification errors.

    Kept as a distinct type + message for the init/verify path and for
    back-compat with callers that ``except APIKeyVerificationError``; it is now
    an ``AuthError`` subclass so ``except AuthError`` catches it too.
    """
    def __init__(self, message, **kwargs: Any):
        super().__init__(f"Could not verify Lucidic API key: {message}", **kwargs)


class InsufficientScopeError(LucidicAPIError):
    """403 — the API key is not authorized for this action.

    Usually a missing capability scope: ``required_scope`` names the missing
    ``resource:verb`` when the backend reports it. When ``required_scope`` is
    ``None`` the key HAS the scope but a narrower rule blocked the action (e.g.
    an agent-bound key can't create new agents) — read the message rather than
    assuming a scope is missing.
    """
    status_code = 403

    def __init__(self, message: str, *, required_scope: Optional[str] = None, **kwargs: Any):
        self.required_scope = required_scope
        super().__init__(message, **kwargs)


class NotFoundError(LucidicAPIError):
    """404 — resource not found, **or** an agent-bound key that can't see it
    (the backend returns 404 rather than 403 to avoid leaking existence)."""
    status_code = 404


class ConflictError(LucidicAPIError):
    """409 — conflict, typically a duplicate (unique-constraint) violation."""
    status_code = 409


class RateLimitError(LucidicAPIError):
    """429 — throttled. Raised only after the transport's retry budget is
    exhausted; ``retry_after`` carries the server's hint when present."""
    status_code = 429

    def __init__(self, message: str, *, retry_after: Optional[float] = None, **kwargs: Any):
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class ServiceUnavailableError(LucidicAPIError):
    """503 — temporarily unavailable (Temporal / S3 degraded, or a workflow
    couldn't start). Retryable; raised only after the transport's retry budget
    is exhausted. ``retry_after`` carries the server's hint when present."""
    status_code = 503

    def __init__(self, message: str, *, retry_after: Optional[float] = None, **kwargs: Any):
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class APIError(LucidicAPIError):
    """Any other non-2xx (5xx, unmapped codes, or an unparseable/malformed
    body). The catch-all so callers always get a typed ``LucidicAPIError``."""

class LucidicNotInitializedError(LucidicError):
    """Exception for calling Lucidic functions before Lucidic Client is initialized (lai.init())"""
    def __init__(self):
        super().__init__("Client is not initialized. Make sure to call lai.init() to initialize the client before calling other functions.")

class PromptError(LucidicError):
    "Exception for errors related to prompt management"
    def __init__(self, message: str):
        super().__init__(f"Error getting Lucidic prompt: {message}")

class InvalidOperationError(LucidicError):
    "Exception for errors resulting from attempting an invalid operation"
    def __init__(self, message: str):
        super().__init__(f"An invalid Lucidic operation was attempted: {message}")


class FeatureFlagError(LucidicError):
    """Exception for feature flag fetch failures"""
    def __init__(self, message: str):
        super().__init__(f"Failed to fetch feature flag: {message}")


class AgentIdRequiredError(LucidicError):
    """Raised when an operation needs an ``agent_id`` but the client was created
    without one (LUC-926).

    ``agent_id`` is optional at construction so org-/id-scoped work (agents,
    projects, usage, and any get-by-id) can run without one. Operations that are
    inherently agent-scoped — telemetry ingestion, prompt fetch, and the
    list-by-agent reads — raise this instead. It is raised **even in production**
    (not routed through the telemetry error-swallow): a missing agent_id is a
    deterministic setup mistake, so silently no-op'ing would just lose data.
    """

    def __init__(self, operation: str):
        super().__init__(
            f"{operation} requires an agent_id, but this client was created "
            f"without one. Provide it via LucidicAI(agent_id=...), the "
            f"LUCIDIC_AGENT_ID env var, or (where the method accepts it) pass "
            f"agent_id=... to the call."
        )


def require_agent_id(agent_id: Optional[str], operation: str) -> str:
    """Return ``agent_id`` if present, else raise ``AgentIdRequiredError``.

    The single guard for every agent-scoped operation. Call at the top of the
    operation, before any HTTP request or error-swallowing, so the failure is
    immediate and loud.
    """
    if not agent_id:
        raise AgentIdRequiredError(operation)
    return agent_id


# ---------------------------------------------------------------------------
# mock_call dispatch errors (LUC-607)
# ---------------------------------------------------------------------------
#
# All non-2xx responses from POST /sdk/mock-call carry the envelope
# `{"error": {"code": str, "detail": str, ...code-specific keys...}}`.
# The SDK turns each `code` into a typed subclass below so callers can
# branch with `except LucidicToolDriftError` rather than parsing strings.
#
# The base class accepts arbitrary `extra` kwargs and exposes them as
# attributes — most subclasses just inherit; a couple override `__init__`
# for code-specific fields (e.g. `source_hash` on drift) where attribute
# names are part of the public API.
#
# Network failures use the synthetic code `"network_error"` so callers
# can handle "couldn't reach backend" uniformly with the typed family.


class LucidicMockCallError(LucidicError):
    """Base for POST /sdk/mock-call dispatch failures.

    Subclasses are keyed by the backend's `code` field via
    `_CODE_TO_CLASS`. Use the lookup helper `error_class_for_code(code)`
    when constructing from a parsed envelope.
    """

    # Subclasses override at class level. The base value is what bare
    # `LucidicMockCallError` instances report when none of the typed
    # paths match (forward-compat safety net for codes the SDK doesn't
    # recognize yet — the `code` kwarg shadows on the instance).
    code: str = "mock_call_error"

    def __init__(self, detail: str, *, code: Optional[str] = None, **extra: Any):
        # Allow the base class to be raised with an explicit code (e.g.
        # for unknown codes from a newer backend); subclasses just pass
        # `detail` and rely on the class-level code. Store on instance
        # so the class-level value isn't mutated.
        if code is not None:
            self.code = code
        self.detail = detail
        # Stash every extra field as a public attribute — keeps backwards
        # compat if the backend adds new code-specific fields without an
        # SDK release.
        for k, v in extra.items():
            setattr(self, k, v)
        super().__init__(f"[{self.code}] {detail}")


class LucidicToolDriftError(LucidicMockCallError):
    """Backend 409 — the tool's source_hash drifted from the session's
    `tool_version_snapshot`. The transport layer's drift-fallback path
    (LUC-607's emit_call_through_backend) catches this and runs the
    user's local function instead of raising. Direct callers of the
    resource see the exception.

    Attributes:
        session_hash: hash recorded at session start (None on
            add-mid-session — tool didn't exist when the snapshot was
            written).
        current_hash: the tool's current source_hash on the backend.
    """
    code = "tool_drift"

    def __init__(
        self,
        detail: str,
        *,
        session_hash: Optional[str] = None,
        current_hash: Optional[str] = None,
        **extra: Any,
    ):
        super().__init__(detail, **extra)
        self.session_hash = session_hash
        self.current_hash = current_hash


class LucidicToolBlockedError(LucidicMockCallError):
    """Backend 422 — Tool.tier=BLOCKED. Dashboard configured this tool
    to be un-callable through mock_call. No fallback; user must change
    the tier or stop calling the tool."""
    code = "tool_blocked"


class LucidicUnknownToolError(LucidicMockCallError):
    """Backend 404 — no Tool row matches (agent, name). Sync the
    registry via `client.tools.sync()` (LUC-608) or check the tool's
    name matches the dashboard."""
    code = "unknown_tool"


class LucidicSessionNotInitializedError(LucidicMockCallError):
    """Backend 400 — session exists but `tool_version_snapshot` is None.
    Caller must POST /sdk/session-init-fixtures first (LUC-608 wires
    this into `create_session` automatically)."""
    code = "session_not_initialized"


class LucidicSessionNotFoundError(LucidicMockCallError):
    """Backend 404 — session_id doesn't resolve under the auth key's org."""
    code = "session_not_found"


class LucidicMissingDatasetItemError(LucidicMockCallError):
    """Backend 400 — session has no DatasetItem link. Session must be
    started under a tool-backed Dataset for mock_call to resolve a Tool."""
    code = "missing_datasetitem"


class LucidicToolConfigError(LucidicMockCallError):
    """Backend 422 — tier executor rejected the Tool's config (e.g. bad
    impl_body JSON, wrong resource count). Dashboard-side fix."""
    code = "tool_config_error"


class LucidicImplError(LucidicMockCallError):
    """Backend 422 — Tier 3 Python impl_body raised at runtime.

    Attributes:
        cause_type: name of the original exception class (e.g. "ValueError").
    """
    code = "impl_error"

    def __init__(self, detail: str, *, cause_type: Optional[str] = None, **extra: Any):
        super().__init__(detail, **extra)
        self.cause_type = cause_type


class LucidicMissingImplError(LucidicError):
    """Raised by the transport layer when the backend returns PASS_THROUGH
    or tool_drift and no `real_fn` is available to fall back to.

    Common in adapter contexts (OpenAI / Anthropic) where the user
    forgot to include the tool in their `impls` dict. The transport
    can't run the real function, so the call has no resolution.

    Not a LucidicMockCallError subclass — this is SDK-side, not a
    backend-reported failure. Distinct base so callers can catch
    `except LucidicMockCallError` for backend errors without
    accidentally swallowing missing-impl bugs.
    """
    def __init__(self, tool_name: str, reason: str):
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(
            f"No local implementation for tool {tool_name!r} ({reason}). "
            f"Provide one via the `impls` dict (adapters) or by decorating "
            f"the function with @mockable."
        )


class LucidicUnsupportedSQLError(LucidicMockCallError):
    """Backend 422 with code=unsupported_sql — tier1 SQL executor rejected
    the rendered statement (parse / transpile failure, READ_ONLY mutation,
    oversize result, etc.).

    Attributes:
        source_dialect: the dialect the SQL was authored in
            (e.g. "POSTGRES"). Useful for branching: "this is
            Postgres-only syntax we can't run against the fixture."

    Note: this class was originally defined for the M2-era legacy
    `mock_calls.create()` (LUC-483) with a different __init__ signature
    (`detail` + `source_dialect` as positional kwargs). The base-class
    signature here remains compatible — `source_dialect` is exposed as
    an attribute via the `**extra` channel of LucidicMockCallError.
    """
    code = "unsupported_sql"

    def __init__(
        self,
        detail: str = "",
        source_dialect: str = "",
        **extra: Any,
    ):
        # Backwards-compat: old call sites pass `detail` and
        # `source_dialect` as positional kwargs. New call sites coming
        # from the v2 envelope path do the same via the factory.
        super().__init__(detail, source_dialect=source_dialect, **extra)


# Code-to-class lookup. Lives in errors.py (not mock_call.py) so the
# resource and transport modules import from the same source.
_CODE_TO_CLASS: dict[str, Type[LucidicMockCallError]] = {
    LucidicToolDriftError.code: LucidicToolDriftError,
    LucidicToolBlockedError.code: LucidicToolBlockedError,
    LucidicUnknownToolError.code: LucidicUnknownToolError,
    LucidicSessionNotInitializedError.code: LucidicSessionNotInitializedError,
    LucidicSessionNotFoundError.code: LucidicSessionNotFoundError,
    LucidicMissingDatasetItemError.code: LucidicMissingDatasetItemError,
    LucidicToolConfigError.code: LucidicToolConfigError,
    LucidicImplError.code: LucidicImplError,
    LucidicUnsupportedSQLError.code: LucidicUnsupportedSQLError,
}


def error_class_for_code(code: str) -> Type[LucidicMockCallError]:
    """Look up the typed exception subclass for a backend error code.

    Returns the base class for unrecognized codes — newer backends may
    introduce codes the SDK doesn't know about yet; we want callers to
    still get a typed `LucidicMockCallError` they can branch on, not a
    generic `LucidicError`.
    """
    return _CODE_TO_CLASS.get(code, LucidicMockCallError)


# ---------------------------------------------------------------------------
# Central non-2xx decoding (LUC-900)
# ---------------------------------------------------------------------------

# Status -> generic typed exception. The mock-call ``{"error": {"code"}}``
# family is matched by *shape*, not status, before this map is consulted.
_STATUS_TO_CLASS: dict[int, Type[LucidicAPIError]] = {
    400: ValidationError,
    401: APIKeyVerificationError,  # back-compat: keep the verify-key type on 401
    403: InsufficientScopeError,
    404: NotFoundError,
    409: ConflictError,
    422: ValidationError,
    429: RateLimitError,
    503: ServiceUnavailableError,
}


def _mock_error_from_envelope(err: dict) -> LucidicMockCallError:
    """Build the typed mock-call exception from a decoded
    ``{"code": str, "detail": str, ...code-specific}`` error object.

    Mirrors the forward-compat handling of the former
    ``mock_call._exception_from_http_error``: unknown codes fall back to the
    base ``LucidicMockCallError`` carrying the wire ``code``.
    """
    # Caller guarantees "code" is present (matched by shape before dispatch).
    code = err["code"]
    detail = err.get("detail", "")
    extra = {k: v for k, v in err.items() if k not in ("code", "detail")}
    cls = error_class_for_code(code)
    try:
        if cls is LucidicMockCallError:
            return cls(detail, code=code, **extra)
        return cls(detail, **extra)
    except TypeError:
        # Defensive: a subclass with a stricter __init__ rejecting an unknown
        # kwarg still degrades to the base class.
        return LucidicMockCallError(code=code, detail=detail, **extra)


def exception_from_response(
    status_code: int,
    body: Any,
    text: str = "",
    *,
    retry_after: Optional[float] = None,
) -> LucidicError:
    """Decode a non-2xx response into a typed exception.

    Shapes handled, in priority order: the mock-call ``{"error": {"code"}}``
    family, ``{"errors": {...}}`` validation dicts, ``{"error": "..."}`` (with
    optional ``details``), DRF ``{"detail": "..."}``, then a text fallback.
    Unknown/absent status codes yield ``APIError`` so callers always get a
    typed ``LucidicAPIError``.
    """
    details: Any = None
    message: Optional[str] = None

    if isinstance(body, dict):
        err = body.get("error")
        # Mock-call family is keyed by shape (a nested dict with a "code"),
        # which no generic endpoint returns — safe to match regardless of path.
        if isinstance(err, dict) and "code" in err:
            return _mock_error_from_envelope(err)
        if isinstance(body.get("errors"), dict):
            details = body["errors"]
            message = "Validation failed"
        elif isinstance(err, str):
            message = err
            if isinstance(body.get("details"), (dict, list)):
                details = body["details"]
        elif isinstance(body.get("detail"), str):
            message = body["detail"]

    if message is None:
        message = text or f"HTTP {status_code}"

    cls = _STATUS_TO_CLASS.get(status_code, APIError)
    kwargs: dict = {
        "status_code": status_code,
        "details": details,
        "response_body": body if isinstance(body, dict) else None,
        "response_text": text,
    }
    if cls in (RateLimitError, ServiceUnavailableError):
        kwargs["retry_after"] = retry_after
    if cls is InsufficientScopeError and isinstance(body, dict):
        kwargs["required_scope"] = body.get("required_scope")
    return cls(message, **kwargs)


def install_error_handler():
    """Install global handler to create ERROR_TRACEBACK events for uncaught exceptions."""
    from ..sdk.event import create_event
    from ..sdk.init import get_session_id
    from ..sdk.context import current_parent_event_id

    def handle_exception(exc_type, exc_value, exc_traceback):
        try:
            if get_session_id():
                tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                create_event(
                    type="error_traceback",
                    error=str(exc_value),
                    traceback=tb
                )
        except Exception:
            pass
        try:
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        except Exception:
            pass

    sys.excepthook = handle_exception
