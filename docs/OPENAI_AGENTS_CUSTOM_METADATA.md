# Custom Metadata with OpenAI Agents SDK Integration

## Current State

The OpenAI Agents handler (`lucidicai/providers/openai_agents_handler.py`) includes code to extract a `lucidic_metadata` parameter from the `Runner.run_sync()` kwargs:

```python
# Extract custom metadata if provided
metadata = kwargs.pop('lucidic_metadata', {})
```

However, this metadata is currently **not used** when creating steps. The metadata is extracted but not passed to the `lai.create_step()` function.

## Current Usage Pattern (Non-functional)

```python
# This syntax is accepted but metadata is not actually stored
result = Runner.run_sync(
    agent,
    "Your prompt here",
    lucidic_metadata={
        "user_id": "user_123",
        "request_type": "support",
        "priority": "high"
    }
)
```

## Limitation

The current implementation of `lai.create_step()` only accepts these parameters:
- `state`: State description
- `action`: Action description  
- `goal`: Goal description
- `eval_score`: Evaluation score
- `eval_description`: Evaluation description
- `screenshot`: Screenshot (base64 or path)

There is no support for arbitrary metadata fields.

## Workaround

Until custom metadata support is added, you can encode metadata in the existing fields:

```python
# Workaround 1: Encode metadata in the goal
user_id = "user_123"
priority = "high"
prompt = "What is the weather?"

# Manually create step with encoded metadata
step_id = lai.create_step(
    state=f"Processing request from user {user_id}",
    action=f"Weather query (priority: {priority})",
    goal=f"{prompt} | metadata: user_id={user_id}, priority={priority}"
)

# Then run the agent
result = Runner.run_sync(agent, prompt)
```

```python
# Workaround 2: Use eval_description for metadata
import json

metadata = {
    "user_id": "user_123",
    "request_type": "weather",
    "priority": "high"
}

step_id = lai.create_step(
    state="Processing user request",
    action="Weather query",
    goal=prompt,
    eval_description=f"Metadata: {json.dumps(metadata)}"
)
```

## Feature Request

To properly support custom metadata, the following changes would be needed:

1. Update `Step` class to accept a `metadata` parameter
2. Update the API endpoint to store metadata
3. Update the handler to pass metadata to `create_step()`

Example of desired usage:

```python
# Desired future API
result = Runner.run_sync(
    agent,
    "Your prompt",
    lucidic_metadata={
        "user_id": "user_123",
        "session_type": "support",
        "priority": "high",
        "custom_field": "any_value"
    }
)

# This would internally call:
lai.create_step(
    state="...",
    action="...", 
    goal="...",
    metadata={  # New parameter
        "user_id": "user_123",
        "session_type": "support",
        "priority": "high",
        "custom_field": "any_value"
    }
)
```

## Use Cases for Custom Metadata

1. **User Tracking**: Track user IDs, segments, preferences
2. **A/B Testing**: Track experiment IDs, variants, test groups
3. **Analytics**: Business metrics, conversion tracking, funnel analysis
4. **Debugging**: Request IDs, error codes, stack traces
5. **Compliance**: Audit trails, data governance, regulatory requirements
6. **Performance**: Response times, cache hits, resource usage
7. **Routing**: Priority levels, queue positions, SLA tracking

## Recommendation

For now, use the workaround methods above to encode metadata in existing fields. Contact the Lucidic AI team to request proper metadata support if this feature is critical for your use case.