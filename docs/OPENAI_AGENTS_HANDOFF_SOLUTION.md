# OpenAI Agents SDK Handoff Tracking Solution

## Problem Statement

The current OpenAI Agents handler only creates one step per `run_sync` call, even when multiple agents execute due to handoffs. We need to create individual steps for each agent that runs during a handoff chain.

## Key Findings

### 1. RunResult Object Structure

The `RunResult` object returned by `Runner.run_sync()` contains:

- **`last_agent`**: The final agent that executed
- **`final_output`**: The final output from the last agent
- **`new_items`**: A list of all execution items, including:
  - `MessageOutputItem`: Agent messages
  - `HandoffCallItem`: When an agent calls the handoff tool
  - `HandoffOutputItem`: Actual handoff execution with source/target agents
  - `ToolCallItem`: Tool invocations
  - `ToolCallOutputItem`: Tool results

### 2. Handoff Tracking via new_items

The key to tracking all handoffs is the `HandoffOutputItem` objects in `result.new_items`:

```python
# HandoffOutputItem structure:
item.source  # The agent handing off control
item.target  # The agent receiving control
item.raw     # Raw handoff data
```

### 3. Complete Execution Flow

For a handoff chain like Receptionist → TechSupport → Engineering:

1. **Receptionist** executes
   - Creates MessageOutputItem (thinking)
   - Creates HandoffCallItem (calls transfer_to_tech_support)
   - Creates HandoffOutputItem (source=Receptionist, target=TechSupport)

2. **TechSupport** executes (automatic)
   - Creates MessageOutputItem (thinking)
   - Creates HandoffCallItem (calls transfer_to_engineering)
   - Creates HandoffOutputItem (source=TechSupport, target=Engineering)

3. **Engineering** executes (automatic)
   - Creates MessageOutputItem (final response)

All of this happens in a single `run_sync` call!

## Solution Implementation

### Enhanced Handler (openai_agents_handler_v2.py)

The enhanced handler:

1. **Extracts handoff chain** from `result.new_items`:
   ```python
   def _extract_handoff_chain(self, result) -> List[Tuple[str, str]]:
       handoff_chain = []
       for item in result.new_items:
           if type(item).__name__ == "HandoffOutputItem":
               source_name = item.source.name
               target_name = item.target.name
               handoff_chain.append((source_name, target_name))
       return handoff_chain
   ```

2. **Creates individual steps** for each agent:
   ```python
   def _create_steps_for_handoff_chain(self, initial_agent, prompt, handoff_chain, result):
       # Step for initial agent
       step_id = lai.create_step(...)
       
       # Step for each handoff target
       for source, target in handoff_chain:
           step_id = lai.create_step(
               state=f"Running agent '{target}' (handoff from '{source}')",
               action="Handle handoff",
               goal=f"Continue processing from {source}"
           )
   ```

3. **Properly ends steps** with handoff information:
   - Steps that handoff show: "Handoff from X to Y"
   - Final step shows: "Completed: FinalAgent"

## Usage Example

```python
import lucidicai as lai
from lucidicai.providers.openai_agents_handler_v2 import OpenAIAgentsHandlerV2
from agents import Agent, Runner

# Initialize with V2 handler
lai.init(session_name="Multi-Agent Workflow", providers=[], task="Complex task")

# Set up enhanced handler
handler = OpenAIAgentsHandlerV2()
handler.override()

# Create agent chain
receptionist = Agent(name="Receptionist", instructions="...")
tech_support = Agent(name="TechSupport", instructions="...")
engineering = Agent(name="Engineering", instructions="...")

receptionist.handoff = [tech_support]
tech_support.handoff = [engineering]

# Single call creates multiple steps!
result = Runner.run_sync(receptionist, "I have a critical bug!")

# Output:
# [LUCIDIC V2] Created step for initial agent 'Receptionist': step_123
# [LUCIDIC V2] Created step for handoff target 'TechSupport': step_456
# [LUCIDIC V2] Created step for handoff target 'Engineering': step_789
# [LUCIDIC V2] Created 3 steps total
```

## Benefits

1. **Complete visibility**: Every agent execution is tracked as a separate step
2. **Handoff tracking**: Clear indication of which agent handed off to which
3. **Proper lifecycle**: Each step has proper start/end with appropriate metadata
4. **Single call**: All tracking happens within one `run_sync` execution

## Integration Notes

- The V2 handler can replace the current handler
- It maintains backward compatibility
- Works with the existing OpenAI handler for LLM event tracking
- No changes needed to user code - just use the enhanced handler

## Future Enhancements

1. **Async support**: Add wrapper for `run_async`
2. **Streaming support**: Track streaming handoffs
3. **Tool tracking**: Create events for each tool call within steps
4. **Parallel handoffs**: Support for agents that handoff to multiple targets