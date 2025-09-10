"""Type definitions for the Lucidic SDK."""
from enum import Enum
from typing import Literal


class EventType(Enum):
    """Supported event types."""
    LLM_GENERATION = "llm_generation"
    FUNCTION_CALL = "function_call"
    ERROR_TRACEBACK = "error_traceback"
    GENERIC = "generic"


# Provider type literals
ProviderType = Literal[
    "openai",
    "anthropic",
    "langchain",
    "pydantic_ai",
    "openai_agents",
    "litellm",
    "bedrock",
    "aws_bedrock",
    "amazon_bedrock",
    "vertexai",
    "vertex_ai",
    "google",
    "google_generativeai",
    "cohere",
    "groq",
]


# Deprecated type aliases (for backward compatibility)
StepType = EventType  # Steps are now events