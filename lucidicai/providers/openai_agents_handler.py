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
            is_handoff = False
            previous_agent = None
            
            # If there are other active steps, this might be a handoff
            if self._active_steps:
                # Find the most recent active step
                for agent_name, step_info in self._active_steps.items():
                    if agent_name != agent.name and not step_info.get('ended', False):
                        is_handoff = True
                        previous_agent = agent_name
                        # End the previous step
                        prev_step_id = step_info['step_id']
                        try:
                            lai.end_step(
                                step_id=prev_step_id,
                                state=f"Handoff from {agent_name} to {agent.name}",
                                action="Initiated handoff",
                                goal=f"Transfer control to {agent.name}"
                            )
                            self._active_steps[agent_name]['ended'] = True
                        except Exception as e:
                            logger.error(f"Failed to end previous step: {e}")
            
            # Create a step for the agent execution
            user_prompt = args[0] if args else ""
            
            # Generate default values
            if is_handoff:
                default_state = StepState.HANDOFF.format(agent_name=agent.name)
                default_action = StepAction.TRANSFER.format(from_agent=previous_agent)
                default_goal = user_prompt[:200] if user_prompt else StepGoal.CONTINUE_PROCESSING
            else:
                default_state = StepState.RUNNING.format(agent_name=agent.name)
                default_action = StepAction.EXECUTE.format(agent_name=agent.name)
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
            
            step_id = lai.create_step(**step_kwargs)
            
            # Track this step
            self._active_steps[agent.name] = {
                'step_id': step_id,
                'ended': False,
                'metadata': metadata
            }
            
            if step_id:
                print(f"\n[LUCIDIC - OpenAI Agents] Created step: {step_id}")
                if metadata and any(key in metadata for key in ['state', 'action', 'goal']):
                    print(f"[LUCIDIC - OpenAI Agents] Using custom metadata")
                else:
                    print(f"[LUCIDIC - OpenAI Agents] Using automatic defaults")
            
            try:
                # Execute the original function
                print(f"\n[LUCIDIC - OpenAI Agents] Executing agent...")
                result = original_func(agent, *args, **kwargs)
                
                # Note: Actual tool usage is tracked by the OpenAI handler when tools are called
                # We don't need to create an event just for listing available tools
                
                # Check if handoffs occurred by analyzing the result
                handoff_chain = self._extract_handoff_chain(result, agent)
                
                # If there were handoffs, create steps for each agent that executed
                if len(handoff_chain) > 1:
                    print(f"\n[LUCIDIC - OpenAI Agents] Handoff chain detected: {' → '.join([a['name'] for a in handoff_chain])}")
                    
                    # Create steps for each agent in the chain (skip first, we already have a step)
                    for i in range(1, len(handoff_chain)):
                        handoff_info = handoff_chain[i]
                        prev_agent = handoff_chain[i-1]['name']
                        curr_agent = handoff_info['name']
                        
                        # Create a step for this handoff
                        handoff_step_id = lai.create_step(
                            state=f"Handoff: {curr_agent}",
                            action=f"Transfer from {prev_agent}",
                            goal="Continue processing request"
                        )
                        
                        if handoff_step_id:
                            print(f"[LUCIDIC - OpenAI Agents] Created handoff step for {curr_agent}: {handoff_step_id}")
                            
                            # Track this step
                            self._active_steps[curr_agent] = {
                                'step_id': handoff_step_id,
                                'ended': False
                            }
                            
                            # Note: The actual API calls by this agent will be tracked by the OpenAI handler
                            # We don't create a handoff event here as it's not an LLM call
                            
                            # End intermediate steps (not the last one)
                            if i < len(handoff_chain) - 1:
                                lai.end_step(
                                    step_id=handoff_step_id,
                                    state=f"Transferred to {handoff_chain[i+1]['name']}",
                                    action=f"Handoff from {curr_agent}",
                                    goal=f"Continue with {handoff_chain[i+1]['name']}"
                                )
                                self._active_steps[curr_agent]['ended'] = True
                
                # Log the final result
                if hasattr(result, 'messages') and result.messages:
                    final_msg = result.messages[-1].content if hasattr(result.messages[-1], 'content') else str(result.messages[-1])
                    print(f"\n[LUCIDIC - OpenAI Agents] Agent completed successfully")
                    print(f"[LUCIDIC - OpenAI Agents] Final output: {final_msg[:100]}..." if len(final_msg) > 100 else f"[LUCIDIC - OpenAI Agents] Final output: {final_msg}")
                else:
                    result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                    print(f"\n[LUCIDIC - OpenAI Agents] Agent completed: {result_preview}")
                
                # End the appropriate step
                if step_id:
                    # Find the final agent in the chain
                    final_agent_name = handoff_chain[-1]['name'] if handoff_chain else agent.name
                    
                    # End the initial step if it handed off
                    if len(handoff_chain) > 1 and not self._active_steps.get(agent.name, {}).get('ended', False):
                        lai.end_step(
                            step_id=step_id,
                            state=f"Transferred to next agent",
                            action=f"Handoff from {agent.name}",
                            goal="Continue processing"
                        )
                        self._active_steps[agent.name]['ended'] = True
                    
                    # End the final agent's step
                    final_step_id = self._active_steps.get(final_agent_name, {}).get('step_id', step_id)
                    if final_step_id and not self._active_steps.get(final_agent_name, {}).get('ended', False):
                        try:
                            # Use actual result metadata
                            final_output = ""
                            tokens_used = 0
                            
                            if hasattr(result, 'final_output'):
                                final_output = result.final_output[:200] + "..." if len(result.final_output) > 200 else result.final_output
                            
                            # Extract token usage if available
                            if hasattr(result, 'context_wrapper') and hasattr(result.context_wrapper, 'usage'):
                                usage = result.context_wrapper.usage
                                if hasattr(usage, 'total_tokens'):
                                    tokens_used = usage.total_tokens
                            
                            # Get original metadata
                            step_info = self._active_steps.get(final_agent_name, {})
                            original_metadata = step_info.get('metadata', {})
                            
                            # Generate default completion values
                            default_end_state = StepState.FINISHED.format(agent_name=final_agent_name)
                            default_end_action = StepAction.DELIVERED.format(agent_name=final_agent_name)
                            default_end_goal = final_output[:200] if final_output else StepGoal.PROCESSING_FINISHED
                            
                            # Use custom values if provided, otherwise use defaults
                            lai.end_step(
                                step_id=final_step_id,
                                state=original_metadata.get('end_state', default_end_state),
                                action=original_metadata.get('end_action', default_end_action),
                                goal=original_metadata.get('end_goal', default_end_goal)
                            )
                            if final_agent_name in self._active_steps:
                                self._active_steps[final_agent_name]['ended'] = True
                            print(f"[LUCIDIC - OpenAI Agents] Step ended: {final_step_id}")
                        except Exception as e:
                            logger.error(f"Failed to end step {final_step_id}: {e}")
                
                # No guidance needed - handoffs work when using correct syntax
                
                return result
                
            except Exception as e:
                print(f"\n[LUCIDIC - OpenAI Agents] Agent execution failed: {str(e)}")
                # End step with error
                if step_id:
                    try:
                        lai.end_step(
                            step_id=step_id,
                            state=f"Error in {agent.name}",
                            action="Agent execution failed",
                            goal=f"Error: {str(e)}"
                        )
                        self._active_steps[agent.name]['ended'] = True
                    except Exception as end_error:
                        logger.error(f"Failed to end step on error: {end_error}")
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
                    if session and session.active_step:
                        event_id = lai.create_event(
                            description=f"Tool call: {self.name}",
                            result=f"Args: {args}, Kwargs: {kwargs}",
                            model="tool"
                        )
                    
                    # Execute the original function
                    result = self._func(*args, **kwargs)
                    
                    # Update event with result
                    if session and session.active_step and event_id:
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