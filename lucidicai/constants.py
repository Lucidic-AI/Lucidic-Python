"""Constants used throughout the Lucidic SDK"""

# Step states
class StepState:
    """Constants for step states"""
    RUNNING = "Running: {agent_name}"
    FINISHED = "Finished: {agent_name}"
    HANDOFF = "Handoff: {agent_name}"
    TRANSFERRED = "Transferred to {agent_name}"
    ERROR = "Error in {agent_name}"

# Step actions  
class StepAction:
    """Constants for step actions"""
    EXECUTE = "Execute {agent_name}"
    TRANSFER = "Transfer from {from_agent}"
    HANDOFF = "Handoff from {from_agent}"
    DELIVERED = "{agent_name} finished processing"
    FAILED = "Agent execution failed"

# Step goals
class StepGoal:
    """Constants for step goals"""
    PROCESS_REQUEST = "Process request"
    CONTINUE_PROCESSING = "Continue processing"
    CONTINUE_WITH = "Continue with {agent_name}"
    PROCESSING_FINISHED = "Processing finished"
    ERROR = "Error: {error}"

# Event descriptions
class EventDescription:
    """Constants for event descriptions"""
    TOOL_CALL = "Tool call: {tool_name}"
    HANDOFF_EXECUTED = "Agent {agent_name} executed via handoff"
    WAITING_RESPONSE = "Waiting for response..."
    WAITING_STRUCTURED = "Waiting for structured response..."
    RESPONSE_RECEIVED = "Response received"

# Event results
class EventResult:
    """Constants for event results"""
    HANDOFF_COMPLETED = "Handoff from {from_agent} completed"
    TOOL_ARGS = "Args: {args}, Kwargs: {kwargs}"
    TOOL_RESULT = "Result: {result}"

# Log messages
class LogMessage:
    """Constants for log messages"""
    SESSION_INIT = "Session initialized successfully"
    SESSION_CONTINUE = "Session {session_id} continuing..."
    INSTRUMENTATION_ENABLED = "OpenAI Agents SDK instrumentation enabled"
    INSTRUMENTATION_DISABLED = "OpenAI Agents SDK instrumentation disabled"
    NO_ACTIVE_SESSION = "No active session for agent tracking"
    HANDLER_INTERCEPTED = "Intercepted {method} call"
    AGENT_RUNNING = "Running agent '{agent_name}' with prompt: {prompt}"
    AGENT_COMPLETED = "Agent completed successfully"
    STEP_CREATED = "Created step: {step_id}"
    STEP_ENDED = "Step ended: {step_id}"
    HANDOFF_DETECTED = "Handoff chain detected: {chain}"