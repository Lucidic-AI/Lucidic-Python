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
    # Typed HTTP/API errors
    LucidicAPIError,
    ValidationError,
    AuthError,
    InsufficientScopeError,
    NotFoundError,
    ConflictError,
    RateLimitError,
    ServiceUnavailableError,
    APIError,
    # mock_call dispatch error family
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
    # lucidicai.load() errors
    RunLoadError,
    RunSignatureError,
    RunPayloadError,
    RunSecretMissingError,
)
from .sdk.training_modules import TrainingModulesResource

# Typed response-model base
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

# Eval-run webhook signatures
from . import webhook

# One object per run
from .run import Run, RunIds, RunTools, load

# Webhook runner
from .runner import MemoryDedupe, Runner, serve

# Version
__version__ = "4.0.0"

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
    # Typed HTTP/API errors
    "LucidicAPIError",
    "ValidationError",
    "AuthError",
    "InsufficientScopeError",
    "NotFoundError",
    "ConflictError",
    "RateLimitError",
    "ServiceUnavailableError",
    "APIError",
    # mock_call dispatch error family
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
    # lucidicai.load() errors
    "RunLoadError",
    "RunSignatureError",
    "RunPayloadError",
    "RunSecretMissingError",
    "TrainingModulesResource",
    # Typed response-model base
    "APIModel",
    "CursorPage",
    # Prompt object
    "Prompt",
    # Tool dispatch (v2 tool-backed)
    "mockable",
    # OpenAI adapter
    "adispatch_openai_tool_call",
    "dispatch_openai_tool_call",
    "register_openai_tools",
    # Anthropic adapter
    "adispatch_anthropic_tool_call",
    "dispatch_anthropic_tool_call",
    "register_anthropic_tools",
    # LangChain adapter
    "register_langchain_tools",
    # Integrations
    "setup_livekit",
    # Eval-run webhook signatures
    "webhook",
    # One object per run
    "load",
    "Run",
    "RunIds",
    "RunTools",
    # Webhook runner
    "Runner",
    "MemoryDedupe",
    "serve",
    # Version
    "__version__",
]
