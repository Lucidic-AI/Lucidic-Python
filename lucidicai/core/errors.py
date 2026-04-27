from typing import Optional
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


class LucidicUnsupportedSQLError(LucidicError):
    """Raised when the backend's mock_call dispatch rejects a SQL statement
    (parse / transpile failure, READ_ONLY mutation, oversize result, etc.).

    Maps to HTTP 422 from /sdk/mock-call. Carries the structured error body so
    user code can branch on `source_dialect` (e.g., "this is a Postgres-only
    syntax we can't run against the fixture") without parsing the message string.

    Use `client.mock_calls.create_or_none(...)` if you'd rather get `None` than
    catch this exception — useful for agents that prefer sentinel-style handling.
    """
    def __init__(self, detail: str, source_dialect: str):
        super().__init__(f"Unsupported SQL ({source_dialect}): {detail}")
        self.detail = detail
        self.source_dialect = source_dialect


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
