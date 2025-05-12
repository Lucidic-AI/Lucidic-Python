import atexit
import os
import signal
from typing import List, Literal, Optional

from .action import Action
from .client import Client
from .errors import APIKeyVerificationError, InvalidOperationError, LucidicNotInitializedError, PromptError
from .event import Event
from .providers.anthropic_handler import AnthropicHandler
from .providers.langchain import LucidicLangchainHandler
from .providers.openai_handler import OpenAIHandler
from .session import Session
from .state import State
from .step import Step

ProviderType = Literal["openai", "anthropic", "langchain"]

__all__ = [
    'Client',
    'Session',
    'Step',
    'Event',
    'Action',
    'State',
    'init',
    'configure',
    'create_step',
    'end_step',
    'update_step',
    'create_event',
    'update_event',
    'end_event',
    'end_session',
    'get_prompt',
    'ProviderType',
    'APIKeyVerificationError',
    'LucidicNotInitializedError',
    'PromptError',
    'InvalidOperationError',
    'LucidicLangchainHandler',
    'AnthropicHandler',
    'OpenAIHandler'
]


def init(
    session_name: str,
    lucidic_api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    task: Optional[str] = None,
    provider: Optional[ProviderType] = None,
    mass_sim_id: Optional[str] = None,
    rubrics: Optional[list] = None
) -> None:
    """
    Initialize the Lucidic client.
    
    Args:
        session_name: The name of the session.
        lucidic_api_key: API key for authentication. If not provided, will use the LUCIDIC_API_KEY environment variable.
        agent_id: Agent ID. If not provided, will use the LUCIDIC_AGENT_ID environment variable.
        task: Task description.
        provider: Provider type ("openai", "anthropic", "langchain").
        mass_sim_id: Optional mass simulation ID, if session is to be part of a mass simulation.
        rubrics: Optional rubrics for evaluation, list of strings.
    
    Raises:
        InvalidOperationError: If the client is already initialized.
        APIKeyVerificationError: If the API key is invalid.
    """
    try:
        client = Client()
        raise InvalidOperationError("Lucidic client already initialized. Please call lai.reset() before reinitializing.")
    except LucidicNotInitializedError:
        pass
    if lucidic_api_key is None:
        lucidic_api_key = os.getenv("LUCIDIC_API_KEY", None)
        if lucidic_api_key is None:
            raise APIKeyVerificationError("Make sure to either pass your API key into lai.init() or set the LUCIDIC_API_KEY environment variable.")
    if agent_id is None:
        agent_id = os.getenv("LUCIDIC_AGENT_ID", None)
        if agent_id is None:
            raise InvalidOperationError("Lucidic agent ID not specified. Make sure to either pass your agent ID into lai.init() or set the LUCIDIC_AGENT_ID environment variable.")

    client = Client(
        lucidic_api_key=lucidic_api_key,
        agent_id=agent_id,
        session_name=session_name,
        mass_sim_id=mass_sim_id,
        task=task,
        rubrics=rubrics
    )
    # Set up provider
    if provider == "openai":
        client.set_provider(OpenAIHandler(client))
    elif provider == "anthropic":
        client.set_provider(AnthropicHandler(client))
    elif provider == "langchain":
        print(f"[Lucidic] For LangChain, make sure to create a handler and attach it to your top-level Agent class.")
    client.init_session()


def end_session(is_finished: bool, session_eval: Optional[str] = None, session_eval_reason: Optional[str] = None, is_successful: Optional[bool] = None, is_successful_reason: Optional[str] = None) -> None:
    """
    End the current session.
    
    Args:
        is_finished: Whether the session was finished.
    """
    client = Client()
    if not client.session:
        print("[Lucidic] Warning: end_session called when session not initialized. Please call lai.init() first.")
        return
    client.session.end_session(is_finished=is_finished, session_eval=session_eval, session_eval_reason=session_eval_reason, is_successful=is_successful, is_successful_reason=is_successful_reason)
    client.clear_session()


def reset() -> None:
    """
    Reset the client.
    """
    end_session(True)
    Client().reset()


def create_step(
    state: Optional[str] = None, 
    action: Optional[str] = None, 
    goal: Optional[str] = None,
    eval_score: Optional[float] = None,
    eval_description: Optional[str] = None,
    screenshot: Optional[str] = None,
    screenshot_path: Optional[str] = None
) -> None:
    """
    Create a new step. Previous step must be finished to create a new step.
    
    Args:
        state: State description.
        action: Action description.
        goal: Goal description.
        eval_score: Evaluation score.
        eval_description: Evaluation description.
        screenshot: Screenshot encoded in base64. Provide either screenshot or screenshot_path.
        screenshot_path: Screenshot path. Provide either screenshot or screenshot_path.
    """
    client = Client()
    if not client.session:
        print("[Lucidic] Warning: create_step called when session not initialized. Please call lai.init() first.")
        return
    client.session.create_step(**locals())


def update_step(
    state: Optional[str] = None, 
    action: Optional[str] = None, 
    goal: Optional[str] = None,
    eval_score: Optional[float] = None,
    eval_description: Optional[str] = None,
    screenshot: Optional[str] = None,
    screenshot_path: Optional[str] = None
) -> None:
    """
    Update the current step.
    
    Args:
        state: State description.
        action: Action description.
        goal: Goal description.
        eval_score: Evaluation score.
        eval_description: Evaluation description.
        screenshot: Screenshot encoded in base64. Provide either screenshot or screenshot_path.
        screenshot_path: Screenshot path. Provide either screenshot or screenshot_path.
    """
    client = Client()
    if not client.session:
        print("[Lucidic] Warning: update_step called when session not initialized. Please call lai.init() first.")
        return
    if not client.session.active_step:
        raise InvalidOperationError("No active step to update")
    client.session.update_step(**locals())

def update_previous_step(
    index: int, # -1 is the latest step, -2 is the second latest, etc.
    state: Optional[str] = None, 
    action: Optional[str] = None, 
    goal: Optional[str] = None,
    eval_score: Optional[float] = None,
    eval_description: Optional[str] = None,
    screenshot: Optional[str] = None,
    screenshot_path: Optional[str] = None
) -> None:
    """
    Update a previous step.
    
    Args:
        index: Index of the step to update. -1 is the latest step, -2 is the second latest, etc.
        state: State description.
        action: Action description.
        goal: Goal description.
        eval_score: Evaluation score.
        eval_description: Evaluation description.
        screenshot: Screenshot encoded in base64. Provide either screenshot or screenshot_path.
        screenshot_path: Screenshot path. Provide either screenshot or screenshot_path.
    """
    client = Client()
    if not client.session:
        print("[Lucidic] Warning: update_previous_step called when session not initialized. Please call lai.init() first.")
        return
    if index >= 0:
        raise InvalidOperationError("Index must be negative, -1 is the latest step, -2 is the second latest, etc.")
    if index < -len(client.session.step_history):
        raise InvalidOperationError("Index out of bounds")
    if not client.session.step_history[index]:
        print("[Lucidic] Warning: update_previous_step called on an empty step. Please check your index.")
        return
    client.session.step_history[index].update_step(**locals())

def end_step(
    state: Optional[str] = None, 
    action: Optional[str] = None, 
    goal: Optional[str] = None,
    eval_score: Optional[float] = None,
    eval_description: Optional[str] = None,
    screenshot: Optional[str] = None,
    screenshot_path: Optional[str] = None
) -> None:
    """
    End the current step.
    
    Args:
        state: State description.
        action: Action description.
        goal: Goal description.
        eval_score: Evaluation score.
        eval_description: Evaluation description.
        screenshot: Screenshot encoded in base64. Provide either screenshot or screenshot_path.
        screenshot_path: Screenshot path. Provide either screenshot or screenshot_path.
    """
    client = Client()
    if not client.session:
        print("[Lucidic] Warning: end_step called when session not initialized. Please call lai.init() first.")
        return
    if not client.session.active_step:
        raise InvalidOperationError("No active step to end")
    client.session.update_step(is_finished=True, **locals())


def create_event(
    description: Optional[str] = None,
    result: Optional[str] = None,
    cost_added: Optional[float] = None, 
    model: Optional[str] = None,
    screenshots: Optional[List[str]] = None
) -> None:
    """
    Create a new event in the current step. Current step must not be finished.
    
    Args:
        description: Description of the event.
        result: Result of the event.
        cost_added: Cost added by the event.
        model: Model used for the event.
        screenshots: List of screenshots encoded in base64.
    """

    client = Client()
    if not client.session:
        print("[Lucidic] Warning: create_event called when session not initialized. Please call lai.init() first.")
        return
    if not client.session.active_step:
        raise InvalidOperationError("No active step to create event in")
    client.session.active_step.create_event(**locals())


def update_event(
    description: Optional[str] = None,
    result: Optional[str] = None,
    cost_added: Optional[float] = None, 
    model: Optional[str] = None,
    screenshots: Optional[List[str]] = None
) -> None:
    """
    Update the latest event in the current step.
    
    Args:
        description: Description of the event.
        result: Result of the event.
        cost_added: Cost added by the event.
        model: Model used for the event.
    """
    client = Client()
    if not client.session:
        print("[Lucidic] Warning: update_event called when session not initialized. Please call lai.init() first.")
        return
    if not client.session.active_step:
        raise InvalidOperationError("No active step to update event in")
    if not client.session.active_step.event_history:
        raise InvalidOperationError("No events exist in the current step")
    client.session.active_step.event_history[-1].update_event(**locals())


def update_previous_event(
    index: int, # -1 is the latest event, -2 is the second latest, etc.
    description: Optional[str] = None,  
    result: Optional[str] = None,
    cost_added: Optional[float] = None, 
    model: Optional[str] = None,
    screenshots: Optional[List[str]] = None
) -> None:
    """
    Update a previous event in the current step.
    
    Args:
        index: Index of the event to update. -1 is the latest event, -2 is the second latest, etc.
        description: Description of the event.
        result: Result of the event.
        cost_added: Cost added by the event.
        model: Model used for the event.
    """
    client = Client()
    if not client.session:
        print("[Lucidic] Warning: update_previous_event called when session not initialized. Please call lai.init() first.")
        return
    if not client.session.active_step:
        raise InvalidOperationError("No active step to update previous event in")
    if not client.session.active_step.event_history:
        raise InvalidOperationError("No events exist in the current step")
    if index >= 0:
        raise InvalidOperationError("Index must be negative, -1 is the latest event, -2 is the second latest, etc.")
    if index < -len(client.session.active_step.event_history):
        raise InvalidOperationError("Index out of bounds")
    client.session.active_step.event_history[index].update_event(**locals())


def end_event(
    description: Optional[str] = None,
    result: Optional[str] = None,
    cost_added: Optional[float] = None, 
    model: Optional[str] = None,
    screenshots: Optional[List[str]] = None
) -> None:
    """
    End the latest event in the current step.
    
    Args:
        description: Description of the event.
        result: Result of the event.
        cost_added: Cost added by the event.
        model: Model used for the event.
    """
    client = Client()
    if not client.session:
        print("[Lucidic] Warning: end_event called when session not initialized. Please call lai.init() first.")
        return
    if not client.session.active_step:
        raise InvalidOperationError("No active step to end event in")
    if not client.session.active_step.event_history:
        raise InvalidOperationError("No events exist in the current step")
    latest_event = client.session.active_step.event_history[-1]
    if latest_event.is_finished:
        raise InvalidOperationError("Latest event is already finished")
    latest_event.update_event(is_finished=True, **locals())


def get_prompt(
    prompt_name: str, 
    variables: Optional[dict] = None,
    cache_ttl: Optional[int] = 300,
    label: Optional[str] = 'production'
) -> str:
    """
    Get a prompt from the prompt database.
    
    Args:
        prompt_name: Name of the prompt.
        variables: {{Variables}} to replace in the prompt, supplied as a dictionary.
        cache_ttl: Time-to-live for the prompt in the cache in seconds (default: 300). Set to -1 to cache forever. Set to 0 to disable caching.
        label: Optional label for the prompt.
    
    Returns:
        str: The prompt.
    """
    client = Client()
    if not client.session:
        print("[Lucidic] Warning: get_prompt called when session not initialized, and will return an empty string. Please call lai.init() first.")
        return ""
    prompt = client.get_prompt(prompt_name, cache_ttl, label)
    if variables:
        for key, val in variables.items():
            index = prompt.find("{{" + key +"}}")
            if index == -1:
                raise PromptError("Supplied variable not found in prompt")
            prompt = prompt.replace("{{" + key +"}}", str(val))
    if "{{" in prompt and "}}" in prompt and prompt.find("{{") < prompt.find("}}"):
        print("[Lucidic] Warning: Unreplaced variable(s) left in prompt. Please check your prompt.")
    return prompt


@atexit.register
def cleanup():
    original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        print("[Lucidic] Cleanup: This should only take a few seconds...")
        try:
            client = Client()
            end_session(True)
        except LucidicNotInitializedError:
            print("[Lucidic] Client not yet initialized, shutting down")
    finally:
        signal.signal(signal.SIGINT, original_handler)
        pass
