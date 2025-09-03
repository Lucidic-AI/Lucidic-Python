from typing import Optional
import sys
import traceback

class APIKeyVerificationError(Exception):
    """Exception for API key verification errors"""
    def __init__(self, message):
        super().__init__(f"Could not verify Lucidic API key: {message}")

class LucidicNotInitializedError(Exception):
    """Exception for calling Lucidic functions before Lucidic Client is initialized (lai.init())"""
    def __init__(self):
        super().__init__("Client is not initialized. Make sure to call lai.init() to initialize the client before calling other functions.")

class PromptError(Exception):
    "Exception for errors related to prompt management"
    def __init__(self, message: str):
        super().__init__(f"Error getting Lucidic prompt: {message}")

class InvalidOperationError(Exception):
    "Exception for errors resulting from attempting an invalid operation"
    def __init__(self, message: str):
        super().__init__(f"An invalid Lucidic operation was attempted: {message}")


class FeatureFlagError(Exception):
    """Exception for feature flag fetch failures"""
    def __init__(self, message: str):
        super().__init__(f"Failed to fetch feature flag: {message}")


def install_error_handler():
    """Install global handler to create ERROR_TRACEBACK events for uncaught exceptions."""
    from .client import Client
    from .context import current_parent_event_id

    def handle_exception(exc_type, exc_value, exc_traceback):
        try:
            client = Client()
            if client and client.session:
                tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                parent_id = None
                try:
                    parent_id = current_parent_event_id.get(None)
                except Exception:
                    parent_id = None
                client.create_event(
                    type="error_traceback",
                    error=str(exc_value),
                    traceback=tb,
                    parent_event_id=parent_id
                )
        except Exception:
            pass
        try:
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        except Exception:
            pass

    sys.excepthook = handle_exception
