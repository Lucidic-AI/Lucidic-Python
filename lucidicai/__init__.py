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
    # Typed HTTP/API errors (LUC-900)
    LucidicAPIError,
    ValidationError,
    AuthError,
    InsufficientScopeError,
    NotFoundError,
    ConflictError,
    RateLimitError,
    ServiceUnavailableError,
    APIError,
    # mock_call dispatch error family (LUC-607)
    LucidicMockCallError,
    LucidicToolDriftError,
    LucidicToolBlockedError,
    LucidicUnknownToolError,
    LucidicSessionNotInitializedError,
    LucidicSessionNotFoundError,
    LucidicMissingDatasetItemError,
    LucidicToolConfigError,
    LucidicImplError,
    LucidicMissingImplError,
    LucidicUnsupportedSQLError,
)
from .sdk.training_modules import TrainingModulesResource

# Typed response-model base (LUC-905)
from .api.models import APIModel, CursorPage

# Prompt object
from .api.resources.prompt import Prompt

# Tool dispatch (v2 tool-backed)
from .sdk.tools import mockable
from .sdk.tools.adapters import (
    adispatch_anthropic_tool_call,
    adispatch_openai_tool_call,
    dispatch_anthropic_tool_call,
    dispatch_openai_tool_call,
    register_anthropic_tools,
    register_langchain_tools,
    register_openai_tools,
)

# Integrations
from .integrations.livekit import setup_livekit

# Version
__version__ = "3.7.2"

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
    # Typed HTTP/API errors (LUC-900)
    "LucidicAPIError",
    "ValidationError",
    "AuthError",
    "InsufficientScopeError",
    "NotFoundError",
    "ConflictError",
    "RateLimitError",
    "ServiceUnavailableError",
    "APIError",
    # mock_call dispatch error family (LUC-607)
    "LucidicMockCallError",
    "LucidicToolDriftError",
    "LucidicToolBlockedError",
    "LucidicUnknownToolError",
    "LucidicSessionNotInitializedError",
    "LucidicSessionNotFoundError",
    "LucidicMissingDatasetItemError",
    "LucidicToolConfigError",
    "LucidicImplError",
    "LucidicMissingImplError",
    "LucidicUnsupportedSQLError",
    "TrainingModulesResource",
    # Typed response-model base (LUC-905)
    "APIModel",
    "CursorPage",
    # Prompt object
    "Prompt",
    # Tool dispatch (v2 tool-backed)
    "mockable",
    # OpenAI adapter (LUC-579)
    "adispatch_openai_tool_call",
    "dispatch_openai_tool_call",
    "register_openai_tools",
    # Anthropic adapter (LUC-580)
    "adispatch_anthropic_tool_call",
    "dispatch_anthropic_tool_call",
    "register_anthropic_tools",
    # LangChain adapter (LUC-578)
    "register_langchain_tools",
    # Integrations
    "setup_livekit",
    # Version
    "__version__",
]
