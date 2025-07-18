"""Decorators for the Lucidic SDK to simplify step and event tracking."""
import functools
import contextvars
import inspect
import json
import logging
from typing import Any, Callable, Optional, TypeVar, Union

from .client import Client
from .errors import LucidicNotInitializedError

logger = logging.getLogger("Lucidic")

F = TypeVar('F', bound=Callable[..., Any])

# Create context variables to store the current step and event
_current_step = contextvars.ContextVar("current_step", default=None)
_current_event = contextvars.ContextVar("current_event", default=None)

def get_decorator_step():
    return _current_step.get()

def get_decorator_event():
    return _current_event.get()


def step(
    state: Optional[str] = None,
    action: Optional[str] = None,
    goal: Optional[str] = None,
    screenshot_path: Optional[str] = None,
    eval_score: Optional[float] = None,
    eval_description: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator that wraps a function with step tracking.
    
    The decorated function will be wrapped with create_step() at the start
    and end_step() at the end, ensuring proper cleanup even on exceptions.
    
    Args:
        state: State description for the step
        action: Action description for the step  
        goal: Goal description for the step
        eval_score: Evaluation score for the step
        eval_description: Evaluation description for the step
        
    Example:
        @lai.step(
            state="Processing user input",
            action="Validate and parse request",
            goal="Extract intent from user message"
        )
        def process_user_input(message: str) -> dict:
            # Function logic here
            return parsed_intent
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Check if SDK is initialized
            try:
                client = Client()
                if not client.session:
                    # No active session, run function normally
                    logger.warning("No active session, running function normally")
                    return func(*args, **kwargs)
            except LucidicNotInitializedError:
                # SDK not initialized, run function normally
                logger.warning("Lucidic not initialized, running function normally")
                return func(*args, **kwargs)
            
            # Create the step
            step_params = {
                'state': state,
                'action': action,
                'goal': goal,
                'screenshot_path': screenshot_path,
                'eval_score': eval_score,
                'eval_description': eval_description
            }
            # Remove None values
            step_params = {k: v for k, v in step_params.items() if v is not None}
            
            # Import here to avoid circular imports
            from . import create_step, end_step
            step_id = create_step(**step_params)
            tok = _current_step.set(step_id)
            
            try:
                # Execute the wrapped function
                result = func(*args, **kwargs)
                # End step successfully
                end_step(step_id=step_id)
                _current_step.reset(tok)
                return result
            except Exception as e:
                # End step with error indication
                try:
                    end_step(
                        step_id=step_id,
                        eval_score=0.0,
                        eval_description=f"Step failed with error: {str(e)}"
                    )
                    _current_step.reset(tok)
                except Exception:
                    # If end_step fails, just log it
                    logger.error(f"Failed to end step {step_id} after error")
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check if SDK is initialized
            try:
                client = Client()
                if not client.session:
                    # No active session, run function normally
                    logger.warning("No active session, running function normally")
                    return await func(*args, **kwargs)
            except LucidicNotInitializedError:
                # SDK not initialized, run function normally
                logger.warning("Lucidic not initialized, running function normally")
                return await func(*args, **kwargs)
            
            # Create the step
            step_params = {
                'state': state,
                'action': action,
                'goal': goal,
                'screenshot_path': screenshot_path,
                'eval_score': eval_score,
                'eval_description': eval_description
            }
            # Remove None values
            step_params = {k: v for k, v in step_params.items() if v is not None}
            
            # Import here to avoid circular imports
            from . import create_step, end_step
            
            step_id = create_step(**step_params)
            tok = _current_step.set(step_id)
            try:
                # Execute the wrapped function
                result = await func(*args, **kwargs)
                # End step successfully
                end_step(step_id=step_id)
                _current_step.reset(tok)
                return result
            except Exception as e:
                # End step with error indication
                try:
                    end_step(
                        step_id=step_id,
                        eval_score=0.0,
                        eval_description=f"Step failed with error: {str(e)}"
                    )
                    _current_step.reset(tok)
                except Exception:
                    # If end_step fails, just log it
                    logger.error(f"Failed to end step {step_id} after error")
                raise
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


### -- TODO -- Updating even within function causes function result to not be recorded.
def event(
    description: Optional[str] = None,
    result: Optional[str] = None,
    model: Optional[str] = None,
    cost_added: Optional[float] = 0
) -> Callable[[F], F]:
    """
    Decorator that creates an event for a function call.
    
    The decorated function will create an event that captures:
    - Function inputs (as string representation) if description not provided
    - Function output (as string representation) if result not provided
    
    LLM calls within the function will create their own events as normal.
    
    Args:
        description: Custom description for the event. If not provided,
                    will use string representation of function inputs
        result: Custom result for the event. If not provided,
                will use string representation of function output
        model: Model name if this function represents a model call (default: None)
        cost_added: Cost to add for this event (default: 0)
        
    Example:
        @lai.event(description="Parse user query", model="custom-parser")
        def parse_query(query: str) -> dict:
            # Function logic here
            return {"intent": "search", "query": query}
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Check if SDK is initialized
            try:
                client = Client()
                if not client.session:
                    # No active session, run function normally
                    logger.warning("No active session, running function normally")
                    return func(*args, **kwargs)
            except (LucidicNotInitializedError, AttributeError):
                # SDK not initialized or no session, run function normally
                logger.warning("Lucidic not initialized, running function normally")
                return func(*args, **kwargs)
            
            # Import here to avoid circular imports
            from . import create_event, end_event
            
            # Build event description from inputs if not provided
            event_desc = description
            if not event_desc:
                # Get function signature
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Create string representation of inputs
                input_parts = []
                for param_name, param_value in bound_args.arguments.items():
                    try:
                        input_parts.append(f"{param_name}={repr(param_value)}")
                    except Exception:
                        input_parts.append(f"{param_name}=<{type(param_value).__name__}>")
                
                event_desc = f"{func.__name__}({', '.join(input_parts)})"
            
            # Create the event
            event_id = create_event(
                description=event_desc,
                model=model,
                cost_added=cost_added
            )
            tok = _current_event.set(event_id)
            try:
                # Execute the wrapped function
                function_result = func(*args, **kwargs)
                
                # Build event result from output if not provided
                event_result = result
                if not event_result:
                    try:
                        event_result = repr(function_result)
                    except Exception:
                        event_result = str(function_result)
                
                # Update and end the event
                end_event(
                    event_id=event_id,
                    result=event_result,
                )
                _current_event.reset(tok)
                return function_result
                
            except Exception as e:
                # Update event with error
                try:
                    end_event(
                        event_id=event_id,
                        result=f"Error: {str(e)}",
                    )
                    _current_event.reset(tok)
                except Exception:
                    logger.error(f"Failed to end event {event_id} after error")
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check if SDK is initialized
            try:
                client = Client()
                if not client.session:
                    # No active session, run function normally
                    logger.warning("No active session, running function normally")
                    return await func(*args, **kwargs)
            except (LucidicNotInitializedError, AttributeError):
                # SDK not initialized or no session, run function normally
                logger.warning("Lucidic not initialized, running function normally")
                return await func(*args, **kwargs)
            
            # Import here to avoid circular imports
            from . import create_event, end_event
            
            # Build event description from inputs if not provided
            event_desc = description
            if not event_desc:
                # Get function signature
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Create string representation of inputs
                input_parts = []
                for param_name, param_value in bound_args.arguments.items():
                    try:
                        input_parts.append(f"{param_name}={repr(param_value)}")
                    except Exception:
                        input_parts.append(f"{param_name}=<{type(param_value).__name__}>")
                
                event_desc = f"{func.__name__}({', '.join(input_parts)})"
            
            # Create the event
            event_id = create_event(
                description=event_desc,
                model=model,
                cost_added=cost_added
            )
            tok = _current_event.set(event_id)
            try:
                # Execute the wrapped function
                function_result = await func(*args, **kwargs)
                
                # Build event result from output if not provided
                event_result = result
                if not event_result:
                    try:
                        event_result = repr(function_result)
                    except Exception:
                        event_result = str(function_result)
                
                # Update and end the event
                end_event(
                    event_id=event_id,
                    result=event_result,
                )
                _current_event.reset(tok)
                return function_result
                
            except Exception as e:
                # Update event with error
                try:
                    end_event(
                        event_id=event_id,
                        result=f"Error: {str(e)}",
                    )
                    _current_event.reset(tok)
                except Exception:
                    logger.error(f"Failed to end event {event_id} after error")
                raise
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator