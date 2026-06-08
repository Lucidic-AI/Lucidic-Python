"""Unified provider detection utilities.

Consolidates provider detection logic from:
- lucidic_exporter.py (_detect_provider_name)
- model_pricing.py (get_provider_from_model)
- litellm_bridge.py (_extract_provider)
"""
from typing import Any, Dict, Optional


# Provider patterns: provider name -> list of substrings to match
PROVIDER_PATTERNS = {
    "anthropic": ["claude", "anthropic"],
    "openai": ["gpt", "openai", "o1", "o3", "o4", "text-davinci", "code-davinci"],
    "google": ["gemini", "google", "gemma", "palm", "bison"],
    "meta": ["llama", "meta"],
    "mistral": ["mistral", "mixtral"],
    "cohere": ["command", "cohere"],
    "deepseek": ["deepseek"],
    "qwen": ["qwen", "qwq"],
    "together": ["together", "redpajama"],
    "perplexity": ["pplx", "perplexity"],
    "grok": ["grok", "xai"],
    "groq": ["groq"],
}


def _normalize_provider(raw: str) -> Optional[str]:
    """normalize a raw provider attribute value to a canonical provider name.

    handles the new gen_ai.provider.name values (e.g. "azure.ai.openai",
    "gcp.gemini") and the legacy gen_ai.system values. returns None when the
    value is too ambiguous to resolve on its own (e.g. bare "azure"), so the
    caller can fall through to model-name detection.
    """
    r = (raw or "").lower()
    if not r:
        return None
    if "openai" in r:
        return "openai"
    if "anthropic" in r or "claude" in r:
        return "anthropic"
    if "gemini" in r or "google" in r or "vertex" in r or "palm" in r:
        return "google"
    if "bedrock" in r:
        return "bedrock"
    # a known provider name on its own
    for provider, patterns in PROVIDER_PATTERNS.items():
        if r == provider or any(p == r for p in patterns):
            return provider
    # bare cloud names (e.g. "azure") are ambiguous - let the model decide
    if r in ("azure", "aws", "gcp", "cloud"):
        return None
    return r


def detect_provider(
    model: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> str:
    """Detect LLM provider from model name or span attributes.

    Checks in order:
    1. Span attributes (gen_ai.provider.name, gen_ai.system, service.name) - most reliable
    2. Model prefix (e.g., "anthropic/claude-3") - common in LiteLLM
    3. Model name pattern matching - fallback

    Args:
        model: Model name string (e.g., "gpt-4", "claude-3-opus")
        attributes: OpenTelemetry span attributes dict

    Returns:
        Provider name string (e.g., "openai", "anthropic") or "unknown"
    """
    # 1. Check attributes first (most reliable source)
    if attributes:
        # new semconv gen_ai.provider.name, then legacy gen_ai.system
        raw = attributes.get("gen_ai.provider.name") or attributes.get("gen_ai.system")
        if raw:
            normalized = _normalize_provider(str(raw))
            if normalized:
                return normalized

        # Service name may contain provider info
        if service := attributes.get("service.name"):
            service_lower = str(service).lower()
            for provider in PROVIDER_PATTERNS:
                if provider in service_lower:
                    return provider

    # 2. Check for provider prefix in model (e.g., "anthropic/claude-3")
    if model and "/" in model:
        prefix = model.split("/")[0].lower()
        # Validate it's a known provider
        if prefix in PROVIDER_PATTERNS:
            return prefix
        # Check if prefix matches any provider patterns
        for provider, patterns in PROVIDER_PATTERNS.items():
            if any(p in prefix for p in patterns):
                return provider

    # 3. Fall back to model name pattern matching
    if model:
        model_lower = model.lower()
        for provider, patterns in PROVIDER_PATTERNS.items():
            if any(pattern in model_lower for pattern in patterns):
                return provider

    return "unknown"
