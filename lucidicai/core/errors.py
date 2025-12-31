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
