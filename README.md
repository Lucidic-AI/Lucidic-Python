# Lucidic AI Python SDK

The official Python SDK for [Lucidic AI](https://lucidic.ai), providing comprehensive observability and analytics for LLM-powered applications.

## Features

- **Session & Step Tracking** - Track complex AI agent workflows with hierarchical session management
- **Multi-Provider Support** - Automatic instrumentation for OpenAI, Anthropic, LangChain, Google Generative AI (Gemini), Vertex AI, AWS Bedrock, Cohere, Groq, and more
- **Real-time Analytics** - Monitor costs, performance, and behavior of your AI applications
- **Data Privacy** - Built-in masking functions to protect sensitive information
- **Screenshot Support** - Capture and analyze visual context in your AI workflows
- **Production Ready** - OpenTelemetry-based instrumentation for enterprise-scale applications
- **Decorators** - Pythonic decorators for effortless step and event tracking

## Installation

```bash
pip install lucidicai
```

## Quick Start

```python
import lucidicai as lai
from openai import OpenAI

# Initialize the SDK
lai.init(
    session_name="My AI Assistant",
    providers=["openai"]
)

# Create a workflow step
lai.create_step(
    state="Processing user query",
    action="Generate response", 
    goal="Provide helpful answer"
)

# Use your LLM as normal - Lucidic automatically tracks the interaction
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

# End the step and session
lai.end_step()
lai.end_session(is_successful=True)
```

### Quick Start (context manager)

```python
import lucidicai as lai
from openai import OpenAI

# All-in-one lifecycle: init → bind → run → auto-end at context exit
with lai.session(session_name="My AI Assistant", providers=["openai"]):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": "Hello, how are you?"}]
    )
```

## Configuration

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
LUCIDIC_API_KEY=your_api_key       # Required: Your Lucidic API key
LUCIDIC_AGENT_ID=your_agent_id     # Required: Your agent identifier
```

### Initialization Options

```python
lai.init(
    session_name="My Session",              # Required: Name for this session
    api_key="...",                 # Optional: Override env var
    agent_id="...",                        # Optional: Override env var
    providers=["openai", "anthropic", "google", "vertexai", "bedrock", "cohere", "groq"],     # Optional: LLM providers to track
    task="Process customer request",       # Optional: High-level task description
    production_monitoring=False,           # Optional: Production mode flag
    auto_end=True,                         # Optional: Auto-end session on exit (default: True)
    masking_function=my_mask_func,         # Optional: Custom PII masking
    tags=["customer-support", "v1.2"],     # Optional: Session tags
    rubrics=[...]                          # Optional: Evaluation criteria
)
```

## Core Concepts

### Sessions
A session represents a complete interaction or workflow, containing multiple steps and events.

```python
# Start a new session
session_id = lai.init(session_name="Customer Support Chat")

# Continue an existing session
lai.continue_session(session_id="existing-session-id")

# Update session metadata
lai.update_session(
    task="Resolved billing issue",
    session_eval=0.95,
    is_successful=True
)

# End session
lai.end_session(is_successful=True, session_eval=0.9)
```

### Session Context (async-safe)

Lucidic uses Python contextvars to bind a session to the current execution context (threads/async tasks). This guarantees spans from concurrent requests are attributed to the correct session.

There are three recommended patterns:

1) Full lifecycle (auto-end on exit)

```python
import lucidicai as lai
from openai import OpenAI

with lai.session(session_name="order-flow", providers=["openai"]):
    OpenAI().chat.completions.create(
        model="gpt-5",
        messages=[{"role":"user","content":"Place order"}]
    )
# Session automatically ends at context exit.
# Note: any auto_end argument is ignored inside session(...).
```

Async variant:

```python
import lucidicai as lai
from openai import AsyncOpenAI
import asyncio

async def main():
    async with lai.session_async(session_name="async-flow", providers=["openai"]):
        await AsyncOpenAI().chat.completions.create(
            model="gpt-5",
            messages=[{"role":"user","content":"Hello"}]
        )

asyncio.run(main())
```

2) Bind-only (does NOT end the session)

```python
import lucidicai as lai
from openai import OpenAI

sid = lai.init(session_name="request-123", providers=["openai"], auto_end=False)
with lai.bind_session(sid):
    OpenAI().chat.completions.create(
        model="gpt-5",
        messages=[{"role":"user","content":"..."}]
    )
# Session remains open. End explicitly when ready:
lai.end_session()
```

Async variant:

```python
sid = lai.init(session_name="request-async", providers=["openai"], auto_end=False)

async def run():
    async with lai.bind_session_async(sid):
        await AsyncOpenAI().chat.completions.create(
            model="gpt-5",
            messages=[{"role":"user","content":"..."}]
        )

asyncio.run(run())
# End later
lai.end_session()
```

3) Fully manual

```python
sid = lai.init(session_name="manual", providers=["openai"], auto_end=True)
lai.set_active_session(sid)
# ... your workflow ...
lai.clear_active_session()
# End now, or rely on auto_end at process exit
lai.end_session()
```

Function wrappers are also provided:

```python
def do_work():
    from openai import OpenAI
    return OpenAI().chat.completions.create(model="gpt-5", messages=[{"role":"user","content":"wrapped"}])

# Full lifecycle in one call
result = lai.run_session(do_work, init_params={"session_name":"wrapped","providers":["openai"]})

# Bind-only wrapper
sid = lai.init(session_name="bound-only", providers=["openai"], auto_end=False)
result = lai.run_in_session(sid, do_work)
lai.end_session()
```

Notes:
- The context managers are safe for threads and asyncio tasks.
- `session(...)` always ends the session at context exit (ignores any provided auto_end).
- Existing single-threaded usage (plain `init` + provider calls) remains supported.

### Automatic Session Management (auto_end)

By default, Lucidic automatically ends your session when your process exits, ensuring no data is lost. This feature is enabled by default but can be controlled:

```python
# Default behavior - session auto-ends on exit
lai.init(session_name="My Session")  # auto_end=True by default

# Disable auto-end if you want manual control
lai.init(session_name="My Session", auto_end=False)
```

The auto_end feature:
- Automatically calls `end_session()` when your Python process exits
- Works with normal exits, crashes, and interrupts (Ctrl+C)
- Prevents data loss from forgotten `end_session()` calls
- Can be disabled for cases where you need explicit control

When using `session(...)` or `session_async(...)`, the session will end at context exit regardless of the `auto_end` setting. A debug warning is logged if `auto_end` is provided in that context.

### Steps
Steps break down complex workflows into discrete, trackable units.

```python
# Create a step
step_id = lai.create_step(
    state="Current context or state",
    action="What the agent is doing",
    goal="What the agent aims to achieve",
    screenshot_path="/path/to/screenshot.png"  # Optional
)

# Update step progress
lai.update_step(
    step_id=step_id,
    eval_score=0.8,
    eval_description="Partially completed task"
)

# End step
lai.end_step(step_id=step_id)
```

- NOTE: If no step exists when an LLM call is made (but Lucidic has already been initialized), Lucidic will automatically create a new step for that call. This step will contain exactly one event—the LLM call itself.

### Events
Events are automatically tracked when using instrumented providers, but can also be created manually.

```python
# Manual event creation
event_id = lai.create_event(
    description="Generated summary",
    result="Success",
    cost_added=0.002,
    model="gpt-4",
    screenshots=["/path/to/image1.png", "/path/to/image2.png"]
)
```

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

lai.init(session_name="Gemini Example", providers=["google"])  # or "google_generativeai"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")
resp = model.generate_content("Write a haiku about clouds")
```

### Vertex AI
```python
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

lai.init(session_name="Vertex Example", providers=["vertexai"])  # or "vertex_ai"
aiplatform.init(project=os.getenv("GCP_PROJECT"), location=os.getenv("GCP_REGION", "us-central1"))

model = GenerativeModel("gemini-1.5-flash")
resp = model.generate_content("Say hello")
```

### AWS Bedrock
```python
import boto3

lai.init(session_name="Bedrock Example", providers=["bedrock"])  # or "aws_bedrock", "amazon_bedrock"
client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))

resp = client.invoke_model(
    modelId=os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0"),
    body=b'{"inputText": "Hello from Bedrock"}',
    contentType="application/json",
    accept="application/json",
)
```

### Cohere
```python
import cohere

lai.init(session_name="Cohere Example", providers=["cohere"])
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
resp = co.chat(model="command-r", messages=[{"role":"user","content":"Hello"}])
```

### Groq
```python
from groq import Groq

lai.init(session_name="Groq Example", providers=["groq"])
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
resp = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role":"user","content":"Hello from Groq"}],
)
```

## Advanced Features

### Decorators
Simplify your code with Python decorators for automatic tracking:

#### Step Decorator
Wrap functions to automatically create and manage steps:

```python
@lai.step(
    # All parameters are optional and auto generated if not provided
    state="Processing data",    
    action="Transform input",
    goal="Generate output",
    eval_score=1,
    eval_description="Data succesfully processed",
    screenshot_path="/path/to/image"    # populates step image if provided. No image if not provided
)
def process_data(input_data: dict) -> dict:
    # Your processing logic here
    result = transform(input_data)
    return result

# The function automatically creates a step, executes, and ends the step
output = process_data({"key": "value"})
```

#### Event Decorator
Track function calls as events with automatic input/output capture:

```python
@lai.event(
    # All parameters are optional
    description="Calculate statistics",  # function inputs if not provided
    result="Stats calculated"           # function output if not provided
    model="stats-engine",               # Not shown if not provided
    cost_added=0.001                   # 0 if not provided
)
def calculate_stats(data: list) -> dict:
    return {
        'mean': sum(data) / len(data),
        'max': max(data),
        'min': min(data)
    }

# Creates an event with function inputs and outputs
stats = calculate_stats([1, 2, 3, 4, 5])
```

#### Accessing Created Steps and Events
Within decorated functions, you can access and update the created step:

```python
from lucidicai.decorators import get_decorator_step

@lai.step(state="Initial state", action="Process")
def process_with_updates(data: dict) -> dict:
    # Access the current step ID
    step_id = get_decorator_step()
    
    # Manually update the step - this overrides decorator parameters
    lai.update_step(
        step_id=step_id,
        state="Processing in progress",
        eval_score=0.5,
        eval_description="Halfway complete"
    )
    
    # Do some processing...
    result = transform(data)
    
    # Update again before completion
    lai.update_step(
        step_id=step_id,
        eval_score=1.0,
        eval_description="Successfully completed transformation"
    )
    
    return result

# Any updates made within the decorated function overwrite the parameters passed into the decorator.

#### Nested Usage
Decorators can be nested for complex workflows:

```python
@lai.step(state="Main workflow", action="Process batch")
def process_batch(items: list) -> list:
    results = []
    
    @lai.event(description="Process single item")
    def process_item(item):
        # LLM calls here create their own events automatically
        return transform(item)
    
    for item in items:
        results.append(process_item(item))
    
    return results
```

#### Async Support
Both decorators fully support async functions:

```python
@lai.step(state="Async operation", action="Fetch data")
async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

@lai.event(description="Async processing")
async def process_async(data: dict) -> dict:
    await asyncio.sleep(1)
    return transform(data)
```

### Data Masking
Protect sensitive information with custom masking functions:

```python
def mask_pii(text):
    # Your PII masking logic here
    return text.replace("SSN:", "XXX-XX-")

lai.init(
    session_name="Secure Session",
    masking_function=mask_pii
)
```

### Image Analysis
Upload screenshots for visual context:

```python
# With step creation
lai.create_step(
    state="Analyzing UI",
    action="Check layout",
    goal="Verify responsive design",
    screenshot_path="/path/to/screenshot.png"
)

# With events

lai.create_event(
    description="UI validation",
    screenshots=[base64_encoded_image1, base64_encoded_image2]
)
```

### Prompt Management
Fetch and cache prompts from the Lucidic platform:

```python
prompt = lai.get_prompt(
    prompt_name="customer_support",
    variables={"issue_type": "billing"},
    cache_ttl=3600,  # Cache for 1 hour
    label="v1.2"
)
```

### Mass Simulations
Run large-scale testing and evaluation:

```python
# Create a mass simulation
mass_sim_id = lai.create_mass_sim(
    mass_sim_name="Load Test",
    total_num_sessions=1000
)

# Initialize sessions with mass_sim_id
lai.init(
    session_name="Test Session",
    mass_sim_id=mass_sim_id
)
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

## Crash events on uncaught exceptions

When the SDK is initialized, Lucidic will capture uncaught exceptions and create a final crash event before the process exits. This is enabled by default and requires no additional configuration.

### Behavior

- On an uncaught exception (main thread):
  - A Lucidic event is created and linked to the active session.
  - The event description contains the full Python traceback. If a `masking_function` was provided to `lai.init()`, it is applied; long descriptions are truncated to ~16K characters.
  - The event result is set to: "process exited with code 1".
  - The session is ended as unsuccessful with reason `uncaughtException` (independent of `auto_end`).
  - The telemetry provider is best-effort flushed and shut down.
  - Python’s default exit behavior is preserved (exit code 1 and default exception printing).

- On signals (`SIGINT`, `SIGTERM`):
  - A final event is created with a description that includes the signal name and a best-effort stack snapshot.
  - The event result is set to: `"process exited with code <128+signum>"` (e.g., 130 for SIGINT, 143 for SIGTERM).
  - Existing auto-end and telemetry cleanup run, and default signal semantics are preserved.

### Configuration

- Enabled by default after `lai.init(...)`. To opt out:

```python
import lucidicai as lai

lai.init(
    session_name="my-session",
    capture_uncaught=False,  # disables crash event capture
)
```

This behavior is independent of `auto_end`; even when `auto_end` is `False`, the SDK will end the session as unsuccessful in this fatal path.

### Caveats and lifecycle notes

- Multiple handlers and ordering:
  - If other libraries register their own handlers, ordering can affect which path runs first. Lucidic guards against duplication, but if another handler exits the process earlier, the crash event may not complete.

- Main-thread semantics:
  - Only uncaught exceptions on the main thread are treated as process-ending. Exceptions in worker threads do not exit the process by default and are not recorded as crash events by this mechanism.

- Best-effort transport:
  - Network issues or abrupt termination (e.g., forced container kill, `os._exit`) can prevent event delivery despite best efforts.

- Exit semantics:
  - We do not call `sys.exit(1)` from the handler; Python already exits with code 1 for uncaught exceptions, and default printing is preserved by chaining to the original `sys.excepthook`.

- Not intercepted:
  - `SystemExit` raised explicitly (e.g., `sys.exit(...)`) and `os._exit(...)` calls are not treated as uncaught exceptions and will not produce a crash event.

## Best Practices

1. **Initialize Once**: Call `lai.init()` at the start of your application or workflow
2. **Use Steps**: Break complex workflows into logical steps for better tracking
3. **Handle Errors**: Wrap SDK calls in try-except blocks for production applications
4. **Session Cleanup**: With `auto_end` enabled (default), sessions automatically end on exit. For manual control, set `auto_end=False` and call `lai.end_session()`
5. **Mask Sensitive Data**: Use masking functions to protect PII and confidential information

## Examples

### Customer Support Bot
```python
import lucidicai as lai
from openai import OpenAI

# Initialize for customer support workflow
lai.init(
    session_name="Customer Support",
    providers=["openai"],
    task="Handle customer inquiry",
    tags=["support", "chat"]
)

# Step 1: Understand the issue
lai.create_step(
    state="Customer reported login issue",
    action="Diagnose problem",
    goal="Identify root cause"
)

client = OpenAI()
# ... your chatbot logic here ...

lai.end_step()

# Step 2: Provide solution
lai.create_step(
    state="Issue identified as password reset",
    action="Guide through reset process",
    goal="Resolve customer issue"
)

# ... more chatbot logic ...

lai.end_step()
lai.end_session(is_successful=True, session_eval=0.95)
```

### Data Analysis Pipeline
```python
import lucidicai as lai
import pandas as pd

lai.init(
    session_name="Quarterly Sales Analysis",
    providers=["openai"],
    task="Generate sales insights"
)

# Step 1: Data loading
lai.create_step(
    state="Loading Q4 sales data",
    action="Read and validate CSV files",
    goal="Prepare data for analysis"
)

# ... data loading logic ...

lai.end_step()

# Step 2: Analysis
lai.create_step(
    state="Data loaded successfully",
    action="Generate insights using GPT-4",
    goal="Create executive summary"
)

# ... LLM analysis logic ...

lai.end_step()
lai.end_session(is_successful=True)
```

## Support

- **Documentation**: [https://docs.lucidic.ai](https://docs.lucidic.ai)
- **Issues**: [GitHub Issues](https://github.com/Lucidic-AI/Lucidic-Python/issues)

## License

This SDK is distributed under the MIT License.