from typing import Any, Optional, Type
import sys
import traceback


class LucidicError(Exception):
    """Base exception for all Lucidic SDK errors"""
    pass


class APIKeyVerificationError(LucidicError):
    """Exception for API key verification errors"""
    def __init__(self, message):
        super().__init__(f"Could not verify Lucidic API key: {message}")

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
