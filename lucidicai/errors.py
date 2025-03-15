from typing import Optional

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
