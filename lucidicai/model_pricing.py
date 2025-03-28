MODEL_PRICING = {
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-7-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-5-haiku": {"input": 0.25, "output": 1.25},
    "claude-2": {"input": 8.0, "output": 24.0},
    
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
    "gemini-2.0-flash-exp": {"input": 0.1, "output": 0.4},
    
    "o1": {"input": 15.0, "output": 60.0},
    "o3-mini": {"input": 1.1, "output": 4.4},
    "o1-mini": {"input": 1.1, "output": 4.4},
}

def calculate_cost(model: str, token_usage: dict) -> float:
    """
    Calculate cost based on model and token usage.
    
    Args:
        model: The model name (e.g., 'gpt-4o', 'claude-3-opus')
        token_usage: Dictionary with prompt_tokens and completion_tokens counts
    
    Returns:
        Cost in USD
    """
    # Normalize model name
    model_lower = model.lower()
    model_lower = model_lower.replace("anthropic/", "").replace("openai/", "").replace("google/", "").replace("models/", "")
    
    # Remove date suffix from Claude models if present
    for prefix in ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", 
                  "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-7-sonnet"]:
        if model_lower.startswith(prefix) and prefix in MODEL_PRICING:
            model_lower = prefix
            break
    
    # Get pricing
    if model_lower in MODEL_PRICING:
        pricing = MODEL_PRICING[model_lower]
    else:
        print(f"[Warning] No pricing found for model: {model}, using default pricing")
        pricing = {"input": 10.0, "output": 30.0}  # Conservative default
    
    # Get token counts
    input_tokens = token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))
    output_tokens = token_usage.get("completion_tokens", token_usage.get("output_tokens", 0))
    
    # Calculate cost
    cost = ((input_tokens * pricing["input"]) + (output_tokens * pricing["output"])) / 1_000_000
    print(f"")
    return cost
