"""OpenAI Agents SDK handler for Lucidic AI"""
import logging
from typing import Optional, Any, Dict, Callable, List
from .base_providers import BaseProvider
from lucidicai.singleton import singleton
import lucidicai as lai
from lucidicai.constants import StepState, StepAction, StepGoal, EventDescription, EventResult, LogMessage

logger = logging.getLogger("Lucidic")

@singleton
class OpenAIAgentsHandler(BaseProvider):
    """Handler for OpenAI Agents SDK integration
    
    This handler intercepts OpenAI Agents SDK Runner execution to track
    agent steps. The actual LLM calls are tracked by the OpenAI handler.
    
    Note: For handoffs to work, use the 'handoffs' parameter when creating agents:
    Agent(name='...', instructions='...', handoffs=[target_agent])
    
    DO NOT use: agent.handoff = [target_agent] (this doesn't work)
    """
    
    def __init__(self):
        super().__init__()
        self._provider_name = "OpenAI Agents SDK"
        self._is_instrumented = False
        self._active_steps = {}  # Track active steps by agent name
        self._session_id = None  # Track current session for handoffs
        
    def override(self):
        """Replace the OpenAI Agents SDK Runner methods with tracking versions"""
        if self._is_instrumented:
            logger.warning("OpenAI Agents SDK already instrumented")
            return
            
        try:
            # Import OpenAI Agents SDK Runner
            from agents import Runner
            
            # Store original run methods
            self._original_run_sync = Runner.run_sync
            
            # Replace with wrapped versions
            Runner.run_sync = self._wrap_run_sync(Runner.run_sync)
            
            self._is_instrumented = True
            logger.info(LogMessage.INSTRUMENTATION_ENABLED)
            
        except ImportError as e:
            logger.error(f"Failed to import OpenAI Agents SDK: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to instrument OpenAI Agents SDK: {e}")
            raise
    
    # ========== Helper Methods ==========
    
    def _end_active_step(self, agent_name: str, step_id: str, state: str, action: str, goal: str):
        """Helper to end an active step and mark it as ended"""
        try:
            lai.end_step(
                step_id=step_id,
                state=state,
                action=action,
                goal=goal
            )
            if agent_name in self._active_steps:
                self._active_steps[agent_name]['ended'] = True
            logger.debug(f"Ended step {step_id} for {agent_name}")
        except Exception as e:
            logger.error(f"Failed to end step {step_id}: {e}")
    
    def _create_step_with_metadata(self, agent_name: str, user_prompt: str, metadata: dict, is_handoff: bool = False, previous_agent: str = None) -> str:
        """Create a step with appropriate metadata and defaults"""
        # Generate default values
        if is_handoff:
            default_state = StepState.HANDOFF.format(agent_name=agent_name)
            default_action = StepAction.TRANSFER.format(from_agent=previous_agent)
            default_goal = user_prompt[:200] if user_prompt else StepGoal.CONTINUE_PROCESSING
        else:
            default_state = StepState.RUNNING.format(agent_name=agent_name)
            default_action = StepAction.EXECUTE.format(agent_name=agent_name)
            default_goal = user_prompt[:200] if user_prompt else StepGoal.PROCESS_REQUEST
        
        # Use user-provided values or defaults
        step_kwargs = {
            'state': metadata.get('state', default_state),
            'action': metadata.get('action', default_action),
            'goal': metadata.get('goal', default_goal)
        }
        
        # Add any additional fields from metadata
        for key in ['eval_score', 'eval_description', 'screenshot', 'screenshot_path']:
            if key in metadata:
                step_kwargs[key] = metadata[key]
        
        return lai.create_step(**step_kwargs)
    
    def _detect_and_handle_previous_handoff(self, current_agent_name: str) -> tuple[bool, Optional[str]]:
        """Detect if this is a handoff and handle the previous step"""
        is_handoff = False
        previous_agent = None
        
        # If there are other active steps, this might be a handoff
        if self._active_steps:
            # Find the most recent active step
            for agent_name, step_info in self._active_steps.items():
                if agent_name != current_agent_name and not step_info.get('ended', False):
                    is_handoff = True
                    previous_agent = agent_name
                    # End the previous step
                    self._end_active_step(
                        agent_name=agent_name,
                        step_id=step_info['step_id'],
                        state=f"Handoff from {agent_name} to {current_agent_name}",
                        action="Initiated handoff",
                        goal=f"Transfer control to {current_agent_name}"
                    )
                    break
        
        return is_handoff, previous_agent
    
    # ========== Main Wrapper Methods ==========
    
    def _wrap_run_sync(self, original_func):
        """Wrap the sync run method to track execution"""
        def wrapper(agent, *args, **kwargs):
            # Extract custom metadata if provided
            metadata = kwargs.pop('lucidic_metadata', {})
            # Log the agent execution
            logger.info(LogMessage.AGENT_RUNNING.format(
                agent_name=agent.name, 
                prompt=args[0] if args else 'No prompt'
            ))
            
            # Get current session
            session = lai.get_session()
            if not session:
                logger.warning(LogMessage.NO_ACTIVE_SESSION)
                return original_func(agent, *args, **kwargs)
            
            # Store session ID for potential handoffs
            self._session_id = session.session_id
            
            # Check if this is a handoff continuation
            is_handoff, previous_agent = self._detect_and_handle_previous_handoff(agent.name)
            
            # Create a step for the agent execution
            # NOTE: In the Agents SDK, each step represents an agent's execution.
            # We create one synthetic event per step to track the agent's work.
            user_prompt = args[0] if args else ""
            step_id = self._create_step_with_metadata(agent.name, user_prompt, metadata, is_handoff, previous_agent)
            
            # Track this step
            self._active_steps[agent.name] = {
                'step_id': step_id,
                'ended': False,
                'metadata': metadata
            }
            
            if not step_id:
                logger.warning("Failed to create step for agent execution")
            
            try:
                # Execute the original function
                result = original_func(agent, *args, **kwargs)
                
                # Note: Handoff detection and step creation is now handled by the OpenAI handler
                # when it detects different agents making API calls.
                # We only need to log the handoff chain for debugging.
                
                # Check if handoffs occurred by analyzing the result
                handoff_chain = self._extract_handoff_chain(result, agent)
                
                if len(handoff_chain) > 1:
                    logger.info(f"Handoff chain detected: {' → '.join([a['name'] for a in handoff_chain])}")
                
                # Log the final result
                if hasattr(result, 'messages') and result.messages:
                    final_msg = result.messages[-1].content if hasattr(result.messages[-1], 'content') else str(result.messages[-1])
                    logger.info("Agent completed successfully")
                    logger.debug(f"Final output: {final_msg[:100]}..." if len(final_msg) > 100 else f"Final output: {final_msg}")
                else:
                    result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                    logger.info(f"Agent completed: {result_preview}")
                
                # End all active steps that haven't been ended yet
                # The OpenAI handler creates steps on-demand during API calls,
                # so we just need to end any steps that are still active
                for agent_name, step_info in list(self._active_steps.items()):
                    if not step_info.get('ended', False):
                        step_id = step_info['step_id']
                        try:
                            # Extract result information if available
                            final_output = ""
                            if hasattr(result, 'final_output'):
                                final_output = result.final_output[:200] + "..." if len(result.final_output) > 200 else result.final_output
                            elif hasattr(result, 'messages') and result.messages:
                                final_msg = result.messages[-1]
                                final_output = final_msg.content[:200] if hasattr(final_msg, 'content') else str(final_msg)[:200]
                            
                            # Get original metadata
                            original_metadata = step_info.get('metadata', {})
                            
                            # Generate appropriate completion values based on whether this is the final agent
                            is_final_agent = (hasattr(result, 'last_agent') and result.last_agent.name == agent_name) or (agent_name == agent.name and len(handoff_chain) <= 1)
                            
                            if is_final_agent:
                                # This is the final agent that produced the result
                                default_end_state = StepState.FINISHED.format(agent_name=agent_name)
                                default_end_action = StepAction.DELIVERED.format(agent_name=agent_name)
                                default_end_goal = final_output[:200] if final_output else StepGoal.PROCESSING_FINISHED
                            else:
                                # This agent handed off to another
                                default_end_state = "Transferred to next agent"
                                default_end_action = f"Handoff from {agent_name}"
                                default_end_goal = "Continue processing"
                            
                            # Use custom values if provided, otherwise use defaults
                            self._end_active_step(
                                agent_name=agent_name,
                                step_id=step_id,
                                state=original_metadata.get('end_state', default_end_state),
                                action=original_metadata.get('end_action', default_end_action),
                                goal=original_metadata.get('end_goal', default_end_goal)
                            )
                            logger.debug(f"Step ended for {agent_name}: {step_id}")
                        except Exception as e:
                            logger.error(f"Failed to end step {step_id} for {agent_name}: {e}")
                
                return result
                
            except Exception as e:
                logger.error(f"Agent execution failed: {str(e)}")
                # End step with error
                if step_id:
                    self._end_active_step(
                        agent_name=agent.name,
                        step_id=step_id,
                        state=f"Error in {agent.name}",
                        action="Agent execution failed",
                        goal=f"Error: {str(e)}"
                    )
                raise
        
        return wrapper
    
    def undo_override(self):
        """Restore the original methods"""
        if not self._is_instrumented:
            return
            
        try:
            # Restore Runner methods
            if hasattr(self, '_original_run_sync'):
                from agents import Runner
                Runner.run_sync = self._original_run_sync
            
            self._is_instrumented = False
            self._active_steps.clear()
            self._session_id = None
            logger.info(LogMessage.INSTRUMENTATION_DISABLED)
            
        except Exception as e:
            logger.error(f"Failed to uninstrument OpenAI Agents SDK: {e}")
    
    def handle_response(self, response, kwargs, session: Optional = None):
        """Not used for this provider - we use method wrapping instead"""
        pass
    
    # ========== Tool Wrapping Methods ==========
    
    @staticmethod
    def wrap_tool_function(func: Callable) -> Callable:
        """Wrap a tool function to ensure it has the required attributes
        
        This fixes the 'function' object has no attribute 'name' issue
        """
        # If it's already a tool object (has name attribute), return as-is
        if hasattr(func, 'name'):
            return func
            
        # Check if it's decorated with @function_tool from OpenAI SDK
        # These tools have special attributes we should preserve
        if hasattr(func, '__wrapped__'):
            # This might be a decorated function, check if it has tool attributes
            if hasattr(func, 'name') or hasattr(func, 'tool_name'):
                return func
        
        # Create a wrapper class that mimics OpenAI Agents SDK tool structure
        class ToolWrapper:
            def __init__(self, func):
                self._func = func
                self.name = func.__name__
                self.description = func.__doc__ or f"Tool function: {func.__name__}"
                self.tool_name = func.__name__  # Some versions might use tool_name
                # Preserve original function attributes
                self.__name__ = func.__name__
                self.__doc__ = func.__doc__
                self.__module__ = getattr(func, '__module__', None)
                self.__qualname__ = getattr(func, '__qualname__', func.__name__)
                
            def __call__(self, *args, **kwargs):
                """Execute the wrapped function and track as event"""
                try:
                    # Log tool execution
                    logger.info(f"[Tool] Executing {self.name} with args={args}, kwargs={kwargs}")
                    
                    # Create an event for the tool call
                    session = lai.get_session()
                    event_id = None
                    if session:
                        # Find the appropriate step for this tool call
                        # Check handler's active steps first
                        handler = OpenAIAgentsHandler()
                        step_id = None
                        
                        # Look for an active (non-ended) step
                        for agent_name, step_info in handler._active_steps.items():
                            if not step_info.get('ended', False):
                                step_id = step_info['step_id']
                                break
                        
                        # If no active step found, use the last created step
                        if not step_id and handler._active_steps:
                            # Get the most recently added step
                            last_agent = list(handler._active_steps.keys())[-1]
                            step_id = handler._active_steps[last_agent]['step_id']
                        
                        if step_id:
                            event_id = lai.create_event(
                                step_id=step_id,
                                description=f"Tool call: {self.name}",
                                result=f"Args: {args}, Kwargs: {kwargs}",
                                model="tool"
                            )
                    
                    # Execute the original function
                    result = self._func(*args, **kwargs)
                    
                    # Update event with result
                    if session and event_id:
                        lai.end_event(
                            event_id=event_id,
                            result=f"Result: {result}"
                        )
                    
                    return result
                except Exception as e:
                    logger.error(f"[Tool] Error in {self.name}: {e}")
                    raise
            
            def __repr__(self):
                return f"<Tool: {self.name}>"
            
            def __str__(self):
                return self.name
        
        return ToolWrapper(func)
    
    # ========== Utility Methods ==========
    
    def _extract_handoff_chain(self, result, initial_agent):
        """Extract the complete handoff chain from the result
        
        Returns a list of dicts with agent info
        """
        chain = [{'name': initial_agent.name}]
        
        # Check if result has new_items for handoff information
        if hasattr(result, 'new_items'):
            for item in result.new_items:
                # Check for HandoffOutputItem
                if hasattr(item, '__class__') and 'HandoffOutputItem' in item.__class__.__name__:
                    if hasattr(item, 'target'):
                        target_name = item.target.name if hasattr(item.target, 'name') else str(item.target)
                        # Add to chain if not already there
                        if not any(a['name'] == target_name for a in chain):
                            chain.append({'name': target_name})
        
        # Also check if last_agent is different (indicates handoff)
        if hasattr(result, 'last_agent') and result.last_agent.name != initial_agent.name:
            if not any(a['name'] == result.last_agent.name for a in chain):
                chain.append({'name': result.last_agent.name})
        
        return chain
    
    @staticmethod
    def prepare_tools(tools: List[Any]) -> List[Any]:
        """Prepare a list of tools for use with OpenAI Agents SDK
        
        Wraps functions to ensure compatibility
        """
        prepared = []
        handler = OpenAIAgentsHandler()
        for tool in tools:
            if callable(tool) and not hasattr(tool, 'name'):
                # Wrap raw functions
                prepared.append(handler.wrap_tool_function(tool))
            else:
                # Already compatible or not a function
                prepared.append(tool)
        return prepared