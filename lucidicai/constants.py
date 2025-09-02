"""Constants used throughout the Lucidic SDK (steps removed)."""

# Event descriptions (generic)
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
    INSTRUMENTATION_ENABLED = "Instrumentation enabled"
    INSTRUMENTATION_DISABLED = "Instrumentation disabled"
    NO_ACTIVE_SESSION = "No active session for tracking"
    HANDLER_INTERCEPTED = "Intercepted {method} call"
    AGENT_RUNNING = "Running agent '{agent_name}'"
    AGENT_COMPLETED = "Agent completed successfully"