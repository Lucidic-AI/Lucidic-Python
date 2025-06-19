# OpenAI Agents Handler Documentation

## Overview

The `OpenAIAgentsHandler` provides integration between OpenAI Agents SDK and Lucidic AI, tracking agent executions as steps and enabling comprehensive observability for agent-based applications.

## Features

- Automatic tracking of agent executions as Lucidic steps
- Handoff detection between agents
- Tool function wrapper for compatibility
- Integration with OpenAI handler for LLM call tracking

## Usage

### Basic Setup

```python
import lucidicai as lai
from agents import Agent, Runner

# Initialize with OpenAI Agents provider
lai.init(
    session_name="My Agent Session",
    providers=["openai_agents"],  # This enables both handlers
    task="Agent task description"
)

# Create and run agents
agent = Agent(
    name="MyAgent",
    instructions="You are a helpful assistant"
)

result = Runner.run_sync(agent, "Hello!")

# End session
lai.end_session()
```

### Tool Functions

The SDK provides a wrapper to make functions compatible with OpenAI Agents SDK:

```python
from lucidicai.providers.openai_agents_handler import OpenAIAgentsHandler

# Define your tool function
def calculate(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Wrap the function
handler = OpenAIAgentsHandler()
wrapped_tool = handler.wrap_tool_function(calculate)

# Use with agent
agent = Agent(
    name="Calculator",
    instructions="You are a calculator",
    tools=[wrapped_tool]
)
```

### Multi-Agent Workflows

For workflows with agent handoffs:

```python
# Start session
session_id = lai.init(
    session_name="Multi-Agent Workflow",
    providers=["openai_agents"]
)

# First agent
dispatcher = Agent(name="Dispatcher", instructions="Route requests")
result1 = Runner.run_sync(dispatcher, "I need help")

# End current session
lai.end_session()

# Continue for handoff
lai.continue_session(session_id, providers=["openai_agents"])

# Second agent
specialist = Agent(name="Specialist", instructions="Handle specialized requests")
result2 = Runner.run_sync(specialist, "Continuing from dispatcher")

lai.end_session()
```

## How It Works

1. **Agent Execution Tracking**: The handler wraps `Runner.run_sync` to create a Lucidic step for each agent execution
2. **State Management**: Tracks agent name, instructions, and input as step metadata
3. **Handoff Detection**: Monitors `result.last_agent` to detect when control transfers between agents
4. **Event Creation**: Works with OpenAIHandler to track underlying LLM API calls as events

## API Reference

### OpenAIAgentsHandler

#### Methods

##### `wrap_tool_function(func: Callable) -> Callable`
Wraps a function to make it compatible with OpenAI Agents SDK tools.

**Parameters:**
- `func`: The function to wrap

**Returns:**
- A wrapped function with `name` and `description` attributes

##### `prepare_tools(tools: List[Any]) -> List[Any]`
Prepares a list of tools, wrapping raw functions as needed.

**Parameters:**
- `tools`: List of tool functions or objects

**Returns:**
- List of prepared tools

## Integration Details

When you specify `providers=["openai_agents"]`, Lucidic automatically:

1. Enables the `OpenAIAgentsHandler` to track agent executions
2. Enables the `OpenAIHandler` to track LLM API calls
3. Creates a hierarchical tracking structure:
   - Session (overall workflow)
   - Steps (agent executions)
   - Events (LLM API calls within each step)

## Known Limitations

1. **Tool Execution**: Some tool functions may have compatibility issues with OpenAI Agents SDK
2. **Async Support**: Currently only `run_sync` is supported
3. **Handoffs**: Detected but don't automatically transfer control

## Best Practices

1. Always wrap tool functions with `wrap_tool_function()` for compatibility
2. Use `lai.continue_session()` for explicit handoffs between agents
3. Check logs for handoff detection messages
4. Use agent instructions to work around tool limitations