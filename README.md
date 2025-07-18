# Lucidic AI Python SDK

The official Python SDK for [Lucidic AI](https://lucidic.ai), providing comprehensive observability and analytics for LLM-powered applications.

## Features

- **Session & Step Tracking** - Track complex AI agent workflows with hierarchical session management
- **Multi-Provider Support** - Automatic instrumentation for OpenAI, Anthropic, LangChain, and more
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
    lucidic_api_key="...",                 # Optional: Override env var
    agent_id="...",                        # Optional: Override env var
    providers=["openai", "anthropic"],     # Optional: LLM providers to track
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