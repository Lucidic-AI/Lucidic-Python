"""Lucidic AI SDK - Instance-based client for AI observability.

This SDK provides observability for AI applications, tracking workflows,
costs, and performance across multiple LLM providers.

Example:
    from lucidicai import LucidicAI

    client = LucidicAI(api_key="...", agent_id="...", providers=["openai"])

    with client.create_session(session_name="My Session") as session:
        @client.event
        def my_function():
            # LLM calls are automatically tracked
            pass
        my_function()

    client.close()
"""

# Main client class
from .client import LucidicAI

# Session object
from .session_obj import Session

# Error types
from .core.errors import (
    LucidicError,
    LucidicNotInitializedError,
    APIKeyVerificationError,
    InvalidOperationError,
    PromptError,
    FeatureFlagError,
)

# Prompt object
from .api.resources.prompt import Prompt

# Integrations
from .integrations.livekit import setup_livekit

# Version
__version__ = "3.4.1"

# All exports
__all__ = [
    # Main client
    "LucidicAI",
    # Session object
    "Session",
    # Error types
    "LucidicError",
    "LucidicNotInitializedError",
    "APIKeyVerificationError",
    "InvalidOperationError",
    "PromptError",
    "FeatureFlagError",
    # Prompt object
    "Prompt",
    # Integrations
    "setup_livekit",
    # Version
    "__version__",
]
