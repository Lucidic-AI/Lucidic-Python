"""OpenTelemetry to Lucidic AI concept converter for OpenAI Agents SDK"""
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger("Lucidic")

# OpenAI Agents SDK span types (from AgentOps implementation)
AGENT_SPAN = "agent"
FUNCTION_SPAN = "function" 
GENERATION_SPAN = "generation"
HANDOFF_SPAN = "handoff"
RESPONSE_SPAN = "response"

@dataclass
class SpanInfo:
    """Stores information about a span for conversion"""
    span_id: str
    parent_id: Optional[str]
    name: str
    span_type: str
    span_data: Any  # The actual span data object (AgentSpanData, FunctionSpanData, etc.)
    start_time: float
    end_time: Optional[float] = None
    status: Optional[str] = None
    error: Optional[str] = None
    step_id: Optional[str] = None  # Associated Lucidic step ID


class OpenTelemetryConverter:
    """Converts OpenTelemetry concepts to Lucidic AI concepts
    
    Mapping:
    - Traces → Sessions
    - Spans → Steps (including nested spans)
    - GenerationSpanData → Creates both Step AND Event (to capture raw LLM I/O)
    - FunctionSpanData → Creates Step with Event for function call details
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, str] = {}  # trace_id -> session_id
        self.active_steps: Dict[str, str] = {}  # span_id -> step_id
        self.span_hierarchy: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        self.span_info: Dict[str, SpanInfo] = {}  # span_id -> SpanInfo
        self.current_agent_context: Optional[Dict[str, Any]] = None  # Track current agent for context
        
    def on_trace_start(self, trace_data: Dict[str, Any], session) -> str:
        """Convert a new trace to a Lucidic session"""
        trace_id = trace_data.get("trace_id")
        
        # Trace becomes a Session
        logger.info(f"Creating session from trace {trace_id}")
        
        # Extract metadata for session
        session_name = trace_data.get("name", "OpenAI Agents Session")
        task = self._extract_task_from_trace(trace_data)
        
        # Session is already initialized by the provider
        self.active_sessions[trace_id] = session.session_id
        
        # Update session with trace metadata
        session.update_session(
            task=task,
            tags={"source": "openai_agents_sdk", "trace_id": trace_id}
        )
        
        return session.session_id
    
    def on_span_start(self, span_type: str, span_data: Any, parent_context: Any, session) -> str:
        """Convert a span to a Lucidic step
        
        Args:
            span_type: Type of span (agent, function, generation, etc.)
            span_data: The actual span data object (AgentSpanData, FunctionSpanData, etc.)
            parent_context: Parent span context
            session: Active Lucidic session
        """
        # Generate span ID from data
        span_id = self._generate_span_id(span_data)
        parent_id = self._get_parent_id(parent_context)
        
        # Store span info for later use
        span_info = SpanInfo(
            span_id=span_id,
            parent_id=parent_id,
            name=getattr(span_data, 'name', str(span_data)),
            span_type=span_type,
            span_data=span_data,
            start_time=getattr(span_data, 'start_time', 0)
        )
        self.span_info[span_id] = span_info
        
        # Track hierarchy
        if parent_id:
            if parent_id not in self.span_hierarchy:
                self.span_hierarchy[parent_id] = []
            self.span_hierarchy[parent_id].append(span_id)
        
        # Create a Step for this Span
        logger.info(f"Creating step from {span_type} span: {span_id}")
        
        # Extract step details based on span type
        goal, action, state = self._extract_step_details_from_span(span_type, span_data)
        
        step_id = session.create_step(
            goal=goal,
            action=action,
            state=state
        )
        
        self.active_steps[span_id] = step_id
        span_info.step_id = step_id
        
        # For GenerationSpanData, create an event to capture raw LLM I/O
        if span_type == GENERATION_SPAN:
            self._create_llm_event(span_data, session, step_id)
        
        # For FunctionSpanData, create an event for function details
        elif span_type == FUNCTION_SPAN:
            self._create_function_event(span_data, session, step_id)
        
        # Update agent context
        if span_type == AGENT_SPAN:
            self.current_agent_context = {
                "name": getattr(span_data, 'name', 'unknown'),
                "instructions": getattr(span_data, 'instructions', ''),
                "tools": getattr(span_data, 'tools', []),
                "handoffs": getattr(span_data, 'handoffs', [])
            }
        
        return step_id
    
    def on_span_end(self, span_data: Dict[str, Any], session) -> None:
        """Handle span completion"""
        span_id = span_data.get("span_id")
        step_id = self.active_steps.get(span_id)
        
        if not step_id:
            logger.warning(f"No step found for span {span_id}")
            return
        
        # Update span info
        if span_id in self.span_info:
            self.span_info[span_id].end_time = span_data.get("end_time")
            self.span_info[span_id].status = span_data.get("status")
            self.span_info[span_id].error = span_data.get("error")
        
        # Update step with completion info
        is_successful = span_data.get("status") != "error"
        error_info = span_data.get("error", {})
        
        session.update_step(
            step_id=step_id,
            is_finished=True,
            eval_score=100 if is_successful else 0,
            eval_description=error_info.get("message") if error_info else "Step completed successfully"
        )
        
        # Process any remaining span events
        for event in span_data.get("events", []):
            self._create_event_from_span_event(event, session, step_id=step_id)
    
    def on_trace_end(self, trace_data: Dict[str, Any], session) -> None:
        """Handle trace completion"""
        trace_id = trace_data.get("trace_id")
        
        # Update session completion
        is_successful = trace_data.get("status") != "error"
        
        session.update_session(
            is_finished=True,
            is_successful=is_successful,
            is_successful_reason=trace_data.get("error", {}).get("message") if not is_successful else "Session completed successfully"
        )
        
        # Cleanup
        if trace_id in self.active_sessions:
            del self.active_sessions[trace_id]
    
    def _extract_task_from_trace(self, trace_data: Dict[str, Any]) -> str:
        """Extract task description from trace data"""
        # Look for task in attributes or name
        attributes = trace_data.get("attributes", {})
        if "task" in attributes:
            return attributes["task"]
        if "prompt" in attributes:
            return attributes["prompt"]
        return trace_data.get("name", "OpenAI Agents Task")
    
    def _extract_step_details_from_span(self, span_type: str, span_data: Any) -> tuple[str, str, str]:
        """Extract goal, action, and state from span data based on type"""
        
        if span_type == AGENT_SPAN:
            name = getattr(span_data, 'name', 'Unknown Agent')
            instructions = getattr(span_data, 'instructions', 'No instructions')
            goal = f"Execute agent: {name}"
            action = f"Running agent with instructions: {instructions[:100]}..."
            state = f"Agent: {name}"
        
        elif span_type == FUNCTION_SPAN:
            name = getattr(span_data, 'name', 'Unknown Function')
            from_agent = getattr(span_data, 'from_agent', 'Unknown')
            goal = f"Execute function: {name}"
            action = f"Agent '{from_agent}' calling function '{name}'"
            state = f"Function execution: {name}"
        
        elif span_type == GENERATION_SPAN:
            model = getattr(span_data, 'model', 'unknown')
            input_data = getattr(span_data, 'input', {})
            # Extract first message for context
            first_msg = ""
            if isinstance(input_data, dict) and 'messages' in input_data:
                messages = input_data['messages']
                if messages and len(messages) > 0:
                    first_msg = messages[0].get('content', '')[:50] + "..."
            
            goal = f"Generate response using {model}"
            action = f"Making LLM call with prompt: {first_msg}"
            state = f"LLM Generation ({model})"
        
        elif span_type == HANDOFF_SPAN:
            from_agent = getattr(span_data, 'from_agent', 'unknown')
            to_agent = getattr(span_data, 'to_agent', 'unknown')
            goal = f"Hand off to agent: {to_agent}"
            action = f"Transferring control from '{from_agent}' to '{to_agent}'"
            state = f"Handoff: {from_agent} → {to_agent}"
        
        elif span_type == RESPONSE_SPAN:
            goal = "Process model response"
            action = "Handling and formatting LLM response"
            state = "Response processing"
        
        else:
            # Generic handling
            name = getattr(span_data, 'name', span_type)
            goal = f"Execute: {name}"
            action = f"Processing {span_type} span"
            state = f"Span: {span_type}"
        
        return goal, action, state
    
    def _create_llm_event(self, span_data: Any, session, step_id: str) -> None:
        """Create an event to capture raw LLM input/output from GenerationSpanData"""
        model = getattr(span_data, 'model', 'unknown')
        input_data = getattr(span_data, 'input', {})
        output_data = getattr(span_data, 'output', {})
        
        # Extract full message history
        messages = []
        if isinstance(input_data, dict) and 'messages' in input_data:
            messages = input_data['messages']
        
        # Format the messages for description
        description = f"LLM Call to {model}\n\n"
        description += "=== INPUT MESSAGES ===\n"
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            description += f"\n[{role.upper()}]:\n{content}\n"
        
        # Extract the response
        result = "=== MODEL RESPONSE ===\n"
        if isinstance(output_data, dict):
            if 'choices' in output_data and output_data['choices']:
                # Standard OpenAI response format
                choice = output_data['choices'][0]
                if 'message' in choice:
                    result += choice['message'].get('content', str(choice['message']))
                elif 'text' in choice:
                    result += choice['text']
            else:
                # Raw output
                result += json.dumps(output_data, indent=2)
        else:
            result += str(output_data)
        
        # Extract cost/usage information
        cost = None
        if isinstance(output_data, dict) and 'usage' in output_data:
            usage = output_data['usage']
            # You might want to calculate cost here based on model and usage
            
        # Extract any tool calls
        tool_calls = []
        if isinstance(output_data, dict) and 'choices' in output_data:
            for choice in output_data['choices']:
                if 'message' in choice and 'tool_calls' in choice['message']:
                    tool_calls.extend(choice['message']['tool_calls'])
        
        if tool_calls:
            result += "\n\n=== TOOL CALLS ===\n"
            result += json.dumps(tool_calls, indent=2)
        
        session.create_event(
            step_id=step_id,
            description=description,
            result=result,
            model=model,
            cost_added=cost,
            is_finished=True,
            is_successful=True
        )
    
    def _create_function_event(self, span_data: Any, session, step_id: str) -> None:
        """Create an event for function call details"""
        name = getattr(span_data, 'name', 'Unknown Function')
        input_data = getattr(span_data, 'input', {})
        output_data = getattr(span_data, 'output', None)
        from_agent = getattr(span_data, 'from_agent', 'Unknown')
        
        description = f"Function Call: {name}\n"
        description += f"Called by: {from_agent}\n\n"
        description += "=== FUNCTION INPUT ===\n"
        description += json.dumps(input_data, indent=2) if input_data else "No input"
        
        result = "=== FUNCTION OUTPUT ===\n"
        if output_data is not None:
            result += json.dumps(output_data, indent=2) if isinstance(output_data, (dict, list)) else str(output_data)
        else:
            result += "No output captured"
        
        session.create_event(
            step_id=step_id,
            description=description,
            result=result,
            is_finished=True,
            is_successful=True
        )
    
    def _generate_span_id(self, span_data: Any) -> str:
        """Generate a unique span ID from span data"""
        # Use the span's attributes or generate from content
        if hasattr(span_data, 'id'):
            return str(span_data.id)
        elif hasattr(span_data, 'name'):
            import hashlib
            return hashlib.md5(f"{span_data.name}_{id(span_data)}".encode()).hexdigest()[:16]
        else:
            return str(id(span_data))
    
    def _get_parent_id(self, parent_context: Any) -> Optional[str]:
        """Extract parent span ID from context"""
        if parent_context is None:
            return None
        if hasattr(parent_context, 'span_id'):
            return parent_context.span_id
        elif isinstance(parent_context, dict) and 'span_id' in parent_context:
            return parent_context['span_id']
        return None
    
    def _create_event_from_span_event(self, event_data: Dict[str, Any], session, step_id: Optional[str] = None) -> None:
        """Create a Lucidic Event from a span event (typically an API call)"""
        event_type = event_data.get("type", "")
        
        # These are the actual API calls
        if event_type in ["openai_api_call", "api_call", "llm_call"]:
            description = self._format_event_description(event_data)
            result = event_data.get("result", "")
            model = event_data.get("model")
            cost = event_data.get("cost")
            
            event_kwargs = {
                "description": description,
                "result": result,
                "is_finished": True,
                "is_successful": event_data.get("status") != "error"
            }
            
            if model:
                event_kwargs["model"] = model
            if cost:
                event_kwargs["cost_added"] = cost
            if step_id:
                event_kwargs["step_id"] = step_id
            
            session.create_event(**event_kwargs)
    
    def _format_event_description(self, event_data: Dict[str, Any]) -> str:
        """Format event description based on event type and data"""
        event_type = event_data.get("type", "")
        
        if event_type == "openai_api_call":
            messages = event_data.get("messages", [])
            if messages:
                return f"OpenAI API Call: {messages[-1].get('content', '')[:100]}..."
            return "OpenAI API Call"
        
        elif event_type == "api_call":
            return f"API Call: {event_data.get('endpoint', 'unknown')}"
        
        elif event_type == "llm_call":
            return f"LLM Call: {event_data.get('prompt', '')[:100]}..."
        
        return f"Event: {event_data.get('name', event_type)}"
    
    def cleanup(self):
        """Clean up converter state"""
        self.active_sessions.clear()
        self.active_steps.clear()
        self.span_hierarchy.clear()
        self.span_info.clear()
    
    # Public methods for external testing
    def _convert_trace_to_session(self, trace_data: Any) -> Dict[str, Any]:
        """Convert trace data to session data (for testing)"""
        session_data = {
            'session_name': f"Agent Workflow: {getattr(trace_data, 'name', 'unknown')}",
            'task': self._extract_task_from_trace({'attributes': getattr(trace_data, 'attributes', {})})
        }
        return session_data
    
    def _convert_span_to_step(self, span_data: Any) -> Dict[str, Any]:
        """Convert span data to step data (for testing)"""
        span_type = getattr(span_data, 'span_type', 'unknown')
        name = getattr(span_data, 'name', 'unknown')
        
        # Use the internal method to get details
        goal, action, state = self._extract_step_details_from_span(
            span_type if span_type != 'unknown' else AGENT_SPAN,
            span_data
        )
        
        return {
            'state': state,
            'action': action,
            'goal': goal
        }
    
    def _convert_span_event_to_event(self, event_data: Any) -> Dict[str, Any]:
        """Convert span event to event data (for testing)"""
        return {
            'description': getattr(event_data, 'name', 'unknown event'),
            'model': getattr(event_data, 'attributes', {}).get('model', 'unknown'),
            'cost_added': getattr(event_data, 'attributes', {}).get('cost', None)
        }