# Lucidic AI Python SDK

The official Python SDK for [Lucidic AI](https://lucidic.ai), providing comprehensive observability and analytics for LLM-powered applications.

## Features

- **Session & Event Tracking** - Track complex AI workflows with typed, immutable events and automatic nesting
- **Multi-Provider Support** - Automatic instrumentation for OpenAI, Anthropic, LangChain, Google Generative AI (Gemini), Vertex AI, AWS Bedrock, Cohere, Groq, and more
- **Real-time Analytics** - Monitor costs, performance, and behavior of your AI applications
- **Data Privacy** - Built-in masking functions to protect sensitive information
- **Screenshot Support** - Capture and analyze visual context in your AI workflows
- **Production Ready** - OpenTelemetry-based instrumentation for enterprise-scale applications
- **Decorators** - Pythonic decorators for effortless function tracking with automatic nesting
- **Async Support** - Full support for async/await patterns and concurrent execution

## Installation

```bash
pip install lucidicai
```

## Quick Start

```python
import lucidicai as lai
from openai import OpenAI

# Initialize the SDK
session_id = lai.init(
    session_name="My AI Assistant",
    providers=["openai"]
)

# Use your LLM as normal - Lucidic automatically tracks the interaction
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

# Events are automatically created and queued for delivery
# Session automatically ends on process exit (auto_end=True by default)
```

### Quick Start with Context Manager

```python
import lucidicai as lai
from openai import OpenAI

# All-in-one lifecycle: init → bind → run → auto-end at context exit
with lai.session(session_name="My AI Assistant", providers=["openai"]):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello, how are you?"}]
    )
# Session automatically ends when exiting the context
```

## Configuration

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
LUCIDIC_API_KEY=your_api_key       # Required: Your Lucidic API key
LUCIDIC_AGENT_ID=your_agent_id     # Required: Your agent identifier
LUCIDIC_DEBUG=False                 # Optional: Enable debug logging
LUCIDIC_VERBOSE=False               # Optional: Enable verbose event logging
```

### Initialization Options

```python
lai.init(
    session_name="My Session",              # Optional: Name for this session
    api_key="...",                          # Optional: Override env var
    agent_id="...",                         # Optional: Override env var
    providers=["openai", "anthropic"],      # Optional: LLM providers to track
    task="Process customer request",        # Optional: High-level task description
    auto_end=True,                          # Optional: Auto-end session on exit (default: True)
    masking_function=my_mask_func,          # Optional: Custom PII masking
    tags=["customer-support", "v1.2"],      # Optional: Session tags
    evaluators=[...],                          # Optional: Evaluation criteria
    experiment_id="...",                    # Optional: Link to experiment
    capture_uncaught=True                   # Optional: Capture crash events (default: True)
)
```

## Core Concepts

### Sessions
A session represents a complete interaction or workflow. Sessions are automatically tracked and can be nested across threads and async tasks.

```python
# Start a new session
session_id = lai.init(session_name="Customer Support Chat")

# Update session metadata
lai.update_session(
    task="Resolved billing issue",
    session_eval=0.95,
    is_successful=True
)

# End session (or let auto_end handle it)
lai.end_session(is_successful=True, session_eval=0.9)
```

### Session Context Management (Thread & Async Safe)

Lucidic uses Python's `contextvars` to bind sessions to the current execution context, ensuring correct attribution in concurrent environments.

#### Pattern 1: Full Lifecycle Management

```python
import lucidicai as lai
from openai import OpenAI

# Synchronous version
with lai.session(session_name="order-flow", providers=["openai"]):
    OpenAI().chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Place order"}]
    )
# Session automatically ends at context exit
```

```python
# Async version
import asyncio
from openai import AsyncOpenAI

async def main():
    async with lai.session_async(session_name="async-flow", providers=["openai"]):
        await AsyncOpenAI().chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )

asyncio.run(main())
```

#### Pattern 2: Session Binding (Without Auto-End)

```python
# Create session without auto-end
sid = lai.init(session_name="long-running", providers=["openai"], auto_end=False)

# Bind for specific operations
with lai.bind_session(sid):
    # Operations here are attributed to this session
    OpenAI().chat.completions.create(...)

# Session remains open - end manually when ready
lai.end_session()
```

#### Pattern 3: Function Wrappers

```python
def process_request():
    from openai import OpenAI
    return OpenAI().chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Process this"}]
    )

# Full lifecycle wrapper
result = lai.run_session(
    process_request,
    init_params={"session_name": "wrapped", "providers": ["openai"]}
)

# Bind-only wrapper
sid = lai.init(session_name="manual", providers=["openai"], auto_end=False)
result = lai.run_in_session(sid, process_request)
lai.end_session()
```

### Automatic Session Management

By default, Lucidic automatically ends your session when your process exits:

```python
# Default behavior - session auto-ends on exit
lai.init(session_name="My Session")  # auto_end=True by default

# Disable auto-end for manual control
lai.init(session_name="My Session", auto_end=False)
# Must call lai.end_session() explicitly
```

### Events

Events are automatically created when using instrumented providers. All events are typed and immutable once created.

```python
# Manual event creation with typed payloads
event_id = lai.create_event(
    type="function_call",  # or "llm_generation", "error_traceback", "generic"
    function_name="process_data",
    arguments={"input": "data"},
    return_value={"result": "success"},
    duration=1.5
)
```

Event types and their payloads:
- **llm_generation**: LLM API calls with request/response/usage data
- **function_call**: Function executions with arguments and return values
- **error_traceback**: Errors with full traceback information
- **generic**: General events with custom details

## Provider Integration

### OpenAI
```python
from openai import OpenAI

lai.init(session_name="OpenAI Example", providers=["openai"])
client = OpenAI()

# All OpenAI API calls are automatically tracked
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a haiku about coding"}]
)
```

### Anthropic
```python
from anthropic import Anthropic

lai.init(session_name="Claude Example", providers=["anthropic"])
client = Anthropic()

# Anthropic API calls are automatically tracked
response = client.messages.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
```

### LangChain
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

lai.init(session_name="LangChain Example", providers=["langchain"])

# LangChain calls are automatically tracked
llm = ChatOpenAI(model="gpt-4")
response = llm.invoke([HumanMessage(content="Hello!")])
```

### Google Generative AI (Gemini)
```python
import google.generativeai as genai

lai.init(session_name="Gemini Example", providers=["google"])
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Write a haiku about clouds")
```

### Vertex AI
```python
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

lai.init(session_name="Vertex Example", providers=["vertexai"])
aiplatform.init(project=os.getenv("GCP_PROJECT"), location="us-central1")

model = GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Say hello")
```

### AWS Bedrock
```python
import boto3

lai.init(session_name="Bedrock Example", providers=["bedrock"])
client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = client.invoke_model(
    modelId="amazon.nova-lite-v1:0",
    body=b'{"inputText": "Hello from Bedrock"}',
    contentType="application/json",
    accept="application/json"
)
```

### Cohere
```python
import cohere

lai.init(session_name="Cohere Example", providers=["cohere"])
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
response = co.chat(
    model="command-r",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Groq
```python
from groq import Groq

lai.init(session_name="Groq Example", providers=["groq"])
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Hello from Groq"}]
)
```

## Advanced Features

### Function Tracking with Decorators

The `@lai.event` decorator automatically tracks function calls as nested events:

```python
@lai.event()
def process_data(input_data: dict) -> dict:
    # Function automatically tracked with arguments and return value
    result = transform(input_data)
    return result

# Creates a FUNCTION_CALL event with full tracking
output = process_data({"key": "value"})
```

#### Nested Event Tracking

Events automatically nest when functions call other tracked functions:

```python
@lai.event()
def outer_function(data: str) -> dict:
    # This creates a parent event
    result = inner_function(data)
    return {"processed": result}

@lai.event()
def inner_function(data: str) -> str:
    # This creates a child event nested under outer_function
    return data.upper()

# Creates nested events with parent-child relationship
output = outer_function("hello")
```

#### Error Tracking

The decorator automatically captures exceptions:

```python
@lai.event()
def risky_operation(value: int) -> int:
    if value < 0:
        raise ValueError("Value must be positive")
    return value * 2

try:
    risky_operation(-1)
except ValueError:
    pass  # Error is still tracked in the event
```

#### Async Function Support

```python
@lai.event()
async def async_process(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# Async functions are fully supported
result = await async_process("https://api.example.com/data")
```

### Data Masking

Protect sensitive information with custom masking functions:

```python
def mask_pii(text):
    # Your PII masking logic here
    import re
    # Example: mask email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Example: mask phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    return text

lai.init(
    session_name="Secure Session",
    masking_function=mask_pii
)
```

### Experiments

Create experiments to group and analyze multiple sessions:

```python
# Create an experiment
experiment_id = lai.create_experiment(
    experiment_name="Prompt Optimization Test",
    LLM_boolean_evaluators=["response_quality", "latency"],
    LLM_numeric_evaluators=["coherence", "relevance"],
    description="Testing different prompt strategies",
    tags=["A/B-test", "prompts"]
)

# Link sessions to the experiment
lai.init(
    session_name="Test Variant A",
    experiment_id=experiment_id,
    providers=["openai"]
)
```

### Prompt Management

Fetch and cache prompts from the Lucidic platform:

```python
prompt = lai.get_prompt(
    prompt_name="customer_support",
    variables={"issue_type": "billing", "customer_name": "John"},
    cache_ttl=3600,  # Cache for 1 hour
    label="v1.2"
)

# Variables are replaced in the prompt template
# {{issue_type}} → "billing"
# {{customer_name}} → "John"
```

### Manual Flush

Force flush all pending telemetry data:

```python
# Ensure all events are sent immediately
lai.flush(timeout_seconds=2.0)
```

## Error Handling

The SDK provides specific exceptions for different error scenarios:

```python
from lucidicai.errors import (
    APIKeyVerificationError,
    InvalidOperationError,
    LucidicNotInitializedError,
    PromptError
)

try:
    lai.init(session_name="My Session")
except APIKeyVerificationError:
    print("Invalid API key - check your credentials")
except LucidicNotInitializedError:
    print("SDK not initialized - call lai.init() first")
```

## Crash Event Capture

The SDK automatically captures uncaught exceptions and creates error events:

```python
lai.init(
    session_name="my-session",
    capture_uncaught=True  # Default: True
)

# If an uncaught exception occurs:
# 1. An error_traceback event is created with the full traceback
# 2. The session is ended as unsuccessful
# 3. Telemetry is flushed before exit
```

This feature also handles:
- **SIGINT/SIGTERM signals**: Graceful shutdown with event creation
- **Thread exceptions**: Main thread exceptions trigger full shutdown
- **Masking**: Error messages are masked if a masking_function is provided

## Performance & Architecture

### Non-Blocking Event Delivery
- Events are queued and delivered asynchronously
- Returns immediately with client-side UUID
- Background worker handles batching and retries

### Efficient Batching
- Events batched every 100ms or 100 events
- Large payloads (>64KB) automatically compressed with gzip
- Automatic blob storage for oversized events

### Thread & Async Safety
- Context-aware session binding using contextvars
- Thread-safe singleton pattern
- Full async/await support

### OpenTelemetry Integration
- Industry-standard observability
- Automatic span → event conversion
- Configurable export intervals

## Best Practices

1. **Initialize Once**: Call `lai.init()` at the start of your application
2. **Use Context Managers**: Prefer `with lai.session()` for automatic lifecycle management
3. **Enable Auto-End**: Let the SDK handle session cleanup (default behavior)
4. **Handle Errors**: Wrap SDK calls in try-except blocks for production
5. **Mask Sensitive Data**: Always use masking functions when handling PII
6. **Leverage Decorators**: Use `@lai.event` for automatic function tracking
7. **Group Related Work**: Use experiments to analyze A/B tests and variants

## Examples

### Customer Support Bot
```python
import lucidicai as lai
from openai import OpenAI

# Initialize with context manager for automatic cleanup
with lai.session(
    session_name="Customer Support",
    providers=["openai"],
    task="Handle customer inquiry",
    tags=["support", "chat"]
):
    @lai.event()
    def analyze_issue(customer_message: str) -> str:
        """Analyze the customer's issue"""
        # LLM call is automatically tracked
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a support analyst"},
                {"role": "user", "content": f"Categorize this issue: {customer_message}"}
            ]
        )
        return response.choices[0].message.content
    
    @lai.event()
    def generate_response(issue_category: str) -> str:
        """Generate a helpful response"""
        # Nested event tracking
        # ... response generation logic ...
        return "Response generated"
    
    # Process customer request with automatic nesting
    issue = analyze_issue("I can't login to my account")
    response = generate_response(issue)
```

### Data Analysis Pipeline
```python
import lucidicai as lai
import pandas as pd
from typing import Dict, Any

lai.init(
    session_name="Quarterly Sales Analysis",
    providers=["openai"],
    task="Generate sales insights",
    auto_end=True  # Session will end when process exits
)

@lai.event()
def load_data(file_path: str) -> pd.DataFrame:
    """Load and validate sales data"""
    df = pd.read_csv(file_path)
    # Data loading logic
    return df

@lai.event()
def analyze_with_llm(data_summary: Dict[str, Any]) -> str:
    """Generate insights using GPT-4"""
    from openai import OpenAI
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data analyst"},
            {"role": "user", "content": f"Analyze this sales data: {data_summary}"}
        ]
    )
    return response.choices[0].message.content

# Execute pipeline with automatic tracking
df = load_data("sales_q4.csv")
summary = df.describe().to_dict()
insights = analyze_with_llm(summary)

# Session automatically ends on process exit
```

### Concurrent Processing
```python
import lucidicai as lai
import asyncio
from typing import List

async def process_item(item_id: str, session_id: str) -> dict:
    """Process a single item with session binding"""
    # Bind this coroutine to the session
    with lai.bind_session(session_id):
        @lai.event()
        async def fetch_data(id: str) -> dict:
            # Async operation tracked as nested event
            await asyncio.sleep(0.1)
            return {"id": id, "data": "processed"}
        
        return await fetch_data(item_id)

async def main():
    # Create session
    session_id = lai.init(
        session_name="Batch Processing",
        providers=["openai"],
        auto_end=False  # Manual control for async
    )
    
    # Process items concurrently
    items = ["item1", "item2", "item3"]
    tasks = [process_item(item, session_id) for item in items]
    results = await asyncio.gather(*tasks)
    
    # End session manually
    lai.end_session(is_successful=True)
    return results

# Run async pipeline
asyncio.run(main())
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LUCIDIC_API_KEY` | Required | API authentication key |
| `LUCIDIC_AGENT_ID` | Required | Agent identifier |
| `LUCIDIC_AUTO_END` | True | Auto-end sessions on exit |

## Support

- **Documentation**: [https://docs.lucidic.ai](https://docs.lucidic.ai)
- **Issues**: [GitHub Issues](https://github.com/Lucidic-AI/Lucidic-Python/issues)

## License

This SDK is distributed under the MIT License.