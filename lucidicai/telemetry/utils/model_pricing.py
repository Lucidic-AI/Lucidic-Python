import logging

logger = logging.getLogger("Lucidic")

MODEL_PRICING = {

    # gpt 5.x pricing
    "gpt-5.2": {"input": 1.75, "output": 14.0},
    "gpt-5.1": {"input": 1.25, "output": 10.0},

    # OpenAI GPT-5 Series (Verified 2025)
    "gpt-5": {"input": 1.25, "output": 10.0},
    "gpt-5-mini": {"input": 0.250, "output": 2.0},
    "gpt-5-nano": {"input": 0.05, "output": 0.4},

    # OpenAI GPT-4o Series (Verified 2025)
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4o-realtime-preview": {"input": 5.0, "output": 20.0},  # Text pricing
    "gpt-4o-audio-preview": {"input": 100.0, "output": 200.0},  # Audio pricing per 1M tokens
    
    # OpenAI GPT-4.1 Series (2025)
    "gpt-4.1": {"input": 2.00, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4.1-nano": {"input": 0.2, "output": 0.8},
    
    # OpenAI GPT-4 Series
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
    "gpt-4-vision-preview": {"input": 10.0, "output": 30.0},
    "gpt-4-32k": {"input": 60.0, "output": 120.0},
    
    # OpenAI GPT-3.5 Series
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "gpt-3.5-turbo-16k": {"input": 3.0, "output": 4.0},
    "gpt-3.5-turbo-instruct": {"input": 1.5, "output": 2.0},
    
    # OpenAI o-Series (Reasoning Models) - Verified 2025
    "o1": {"input": 15.0, "output": 60.0},
    "o1-preview": {"input": 15.0, "output": 60.0},
    "o1-mini": {"input": 3.0, "output": 15.0}, 
    "o3": {"input": 15.0, "output": 60.0}, 
    "o3-mini": {"input": 1.1, "output": 4.4}, 
    "o4-mini": {"input": 4.00, "output": 16.0},
    
    # OpenAI Legacy Models
    "text-davinci-003": {"input": 20.0, "output": 20.0},
    "text-davinci-002": {"input": 20.0, "output": 20.0},
    "code-davinci-002": {"input": 20.0, "output": 20.0},

    # Claude 4.5 models
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "claude-opus-4-5": {"input": 5.0, "output": 25.0},
    
    # Claude 4 Models (2025) - Verified
    "claude-4-opus": {"input": 15.0, "output": 75.0},
    "claude-opus-4": {"input": 15.0, "output": 75.0},
    "claude-4-sonnet": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    
    # Claude 3.5 Models - Verified 2025
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-latest": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku": {"input": 1.0, "output": 5.0}, 
    "claude-3-5-haiku-latest": {"input": 1.0, "output": 5.0},
    "claude-3-7-sonnet": {"input": 3.0, "output": 15.0},  # Same as 3.5 sonnet
    "claude-3-7-sonnet-latest": {"input": 3.0, "output": 15.0},
    
    # Claude 3 Models
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-opus-latest": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    
    # Claude 2 Models
    "claude-2": {"input": 8.0, "output": 24.0},
    "claude-2.1": {"input": 8.0, "output": 24.0},
    "claude-2.0": {"input": 8.0, "output": 24.0},
    
    # Claude Instant
    "claude-instant": {"input": 0.8, "output": 2.4},
    "claude-instant-1": {"input": 0.8, "output": 2.4},
    "claude-instant-1.2": {"input": 0.8, "output": 2.4},

    # Gemini 3 series
    "gemini-3-flash-preview": {"input": 0.5, "output": 3.00},
    "gemini-3-pro-preview": {"input": 2.0, "output": 12.00}, # different pricing for different input sizes ????
    
    
    # Google Gemini 2.5 Series (2025) - Verified
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},  # Up to 200k tokens
    "gemini-2.5-pro-preview": {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.6},  # Non-thinking
    "gemini-2.5-flash-preview": {"input": 0.15, "output": 0.6},
    
    # Google Gemini 2.0 Series - Verified
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
    "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free experimental
    "gemini-2.0-flash-experimental": {"input": 0.0, "output": 0.0},
    
    # Google Gemini 1.5 Series - Verified
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0},  # Up to 128k tokens
    "gemini-1.5-pro-preview": {"input": 1.25, "output": 5.0},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.3},  # Up to 128k tokens
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
    
    # Google Gemini 1.0 Series
    "gemini-pro": {"input": 0.5, "output": 1.5},
    "gemini-pro-vision": {"input": 0.25, "output": 0.5},
    "gemini-1.0-pro": {"input": 0.5, "output": 1.5},
    
    # Google PaLM Series
    "text-bison": {"input": 1.0, "output": 1.0},
    "text-bison-32k": {"input": 1.0, "output": 1.0},
    "chat-bison": {"input": 1.0, "output": 1.0},
    "chat-bison-32k": {"input": 1.0, "output": 1.0},
    
    # Meta Llama 4 Series (2025)
    "llama-4-maverick-17b": {"input": 0.2, "output": 0.6},
    "llama-4-scout-17b": {"input": 0.11, "output": 0.34},
    "llama-guard-4-12b": {"input": 0.20, "output": 0.20},
    "meta-llama/llama-4-maverick-17b-128e-instruct": {"input": 0.2, "output": 0.6},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11, "output": 0.34},
    "meta-llama/llama-guard-4-12b-128k": {"input": 0.20, "output": 0.20},
    
    # Meta Llama 3.x Series - Verified 2025 (Together AI pricing)
    "llama-3.3-70b": {"input": 0.54, "output": 0.88}, 
    "llama-3.1-405b": {"input": 6.0, "output": 12.0},  
    "llama-3.1-70b": {"input": 0.54, "output": 0.88},  
    "llama-3.1-8b": {"input": 0.10, "output": 0.18},   
    "llama-3-70b": {"input": 0.54, "output": 0.88},    
    "llama-3-8b": {"input": 0.10, "output": 0.18},     
    "llama-guard-3-8b": {"input": 0.20, "output": 0.20},
    "meta-llama/llama-3.3-70b-versatile-128k": {"input": 0.54, "output": 0.88}, 
    "meta-llama/llama-3.1-8b-instant-128k": {"input": 0.10, "output": 0.18},     
    "meta-llama/llama-3-70b-8k": {"input": 0.54, "output": 0.88},                
    "meta-llama/llama-3-8b-8k": {"input": 0.10, "output": 0.18},                 
    "meta-llama/llama-guard-3-8b-8k": {"input": 0.20, "output": 0.20},
    
    # Mistral Models
    "mistral-large": {"input": 2.0, "output": 6.0},
    "mistral-medium": {"input": 2.7, "output": 8.1},
    "mistral-small": {"input": 0.1, "output": 0.3},
    "mistral-tiny": {"input": 0.14, "output": 0.42},
    "mistral-7b-instruct": {"input": 0.15, "output": 0.15},
    "mistral-8x7b-instruct": {"input": 0.24, "output": 0.24},
    "mistral-saba-24b": {"input": 0.79, "output": 0.79},
    "mistral/mistral-saba-24b": {"input": 0.79, "output": 0.79},
    
    # Cohere Models
    "command": {"input": 1.0, "output": 2.0},
    "command-light": {"input": 0.3, "output": 0.6},
    "command-nightly": {"input": 1.0, "output": 2.0},
    "command-r": {"input": 0.5, "output": 1.5},
    "command-r-plus": {"input": 3.0, "output": 15.0},
    
    # DeepSeek Models
    "deepseek-r1-distill-llama-70b": {"input": 0.75, "output": 0.99},
    "deepseek-ai/deepseek-r1-distill-llama-70b": {"input": 0.75, "output": 0.99},
    "deepseek-coder": {"input": 0.14, "output": 0.28},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek/deepseek-v3-0324": {"input": 0.14, "output": 0.28},
    
    # Qwen Models
    "qwen-qwq-32b": {"input": 0.29, "output": 0.39},
    "qwen/qwen-qwq-32b-preview-128k": {"input": 0.29, "output": 0.39},
    "qwen-turbo": {"input": 0.3, "output": 0.6},
    "qwen-plus": {"input": 0.5, "output": 2.0},
    "qwen-max": {"input": 2.0, "output": 6.0},
    "qwen2.5-32b-instruct": {"input": 0.7, "output": 2.8},
    "qwen2.5-max": {"input": 1.6, "output": 6.4},
    
    # Google Gemma Models
    "gemma-2-9b": {"input": 0.20, "output": 0.20},
    "gemma-2-27b": {"input": 0.27, "output": 0.27},
    "gemma-7b-it": {"input": 0.07, "output": 0.07},
    "google/gemma-2-9b-8k": {"input": 0.20, "output": 0.20},
    
    # Together AI Models
    "together-ai/redpajama-incite-7b-chat": {"input": 0.2, "output": 0.2},
    "together-ai/redpajama-incite-base-3b-v1": {"input": 0.1, "output": 0.1},
    
    # Perplexity Models
    "pplx-7b-chat": {"input": 0.07, "output": 0.28},
    "pplx-70b-chat": {"input": 0.7, "output": 2.8},
    "pplx-7b-online": {"input": 0.07, "output": 0.28},
    "pplx-70b-online": {"input": 0.7, "output": 2.8},

    # Grok Models
    "grok-3-latest": {"input": 3, "output": 15},
    "grok-3": {"input": 3, "output": 15},
    "grok-3-fast": {"input": 5, "output": 25},
    "grok-3-mini": {"input": 0.3, "output": 0.5},
    "grok-3-mini-fast": {"input": 0.6, "output": 4},

}

# Provider average pricing fallbacks
PROVIDER_AVERAGES = {
    "anthropic": {"input": 3.0, "output": 15.0},    # Average of Claude 3.5 Sonnet
    "openai": {"input": 2.5, "output": 10.0},       # GPT-4o pricing
    "google": {"input": 0.5, "output": 1.5},        # Gemini Pro average
    "meta": {"input": 0.3, "output": 0.5},          # Llama average
    "mistral": {"input": 0.5, "output": 1.5},       # Mistral average
    "cohere": {"input": 1.0, "output": 2.0},        # Command model average
    "deepseek": {"input": 0.3, "output": 0.5},      # DeepSeek average
    "qwen": {"input": 0.5, "output": 1.0},          # Qwen average
    "together": {"input": 0.15, "output": 0.15},    # Together AI average
    "perplexity": {"input": 0.4, "output": 1.5},    # Perplexity average
    "grok": {"input": 2.4, "output": 12},           # Grok average
    "groq": {"input": 0.3, "output": 0.6},          # Groq average (placeholder)
}

def get_provider_from_model(model: str) -> str:
    """Extract provider name from model string.

    This is a backward-compatible alias for detect_provider().
    """
    from .provider import detect_provider
    return detect_provider(model=model)

def normalize_model_name(model: str) -> str:
    """Normalize model name by stripping dates and provider prefixes"""
    import re
    
    model_lower = model.lower()
    # Remove provider prefixes (generalizable pattern: any_provider/)
    model_lower = re.sub(r'^[^/]+/', '', model_lower)
    # Strip Google/Vertex prefixes
    model_lower = model_lower.replace('publishers/google/models/', '').replace('models/', '')
    
    # Strip date suffixes (20240229, 20241022, etc.) but preserve model versions like o1-mini, o3-mini
    # Pattern: remove -YYYYMMDD or -YYYY-MM-DD at the end
    date_pattern = r'-\d{8}$|_\d{8}$|-\d{4}-\d{2}-\d{2}$'
    model_lower = re.sub(date_pattern, '', model_lower)
    
    return model_lower

def calculate_cost(model: str, token_usage: dict) -> float:
    model_lower = normalize_model_name(model)
    
    # Try exact match first, then longest prefix match
    pricing = (
        MODEL_PRICING.get(model_lower) or
        MODEL_PRICING.get(
            next((prefix for prefix in sorted(MODEL_PRICING.keys(), key=len, reverse=True) 
                  if model_lower.startswith(prefix)), None)
        ) or
        PROVIDER_AVERAGES.get(
            get_provider_from_model(model), 
            {"input": 2.5, "output": 10.0}
        )
    )
    
    # Print warning only if using fallback pricing
    if model_lower not in MODEL_PRICING:
        provider = get_provider_from_model(model)
        if provider in PROVIDER_AVERAGES:
            logger.warning(f"No pricing found for model: {model}, using {provider} average pricing")
        else:
            logger.warning(f"No pricing found for model: {model}, using default pricing")
    
    input_tokens = token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))
    output_tokens = token_usage.get("completion_tokens", token_usage.get("output_tokens", 0))
    
    cost = ((input_tokens * pricing["input"]) + (output_tokens * pricing["output"])) / 1_000_000
    return cost
