import atexit
import signal
import os
from typing import Optional, Union, Literal

from PIL import Image

from .action import Action
from .client import Client
from .errors import APIKeyVerificationError, InvalidOperationError, LucidicNotInitializedError, PromptError
from .event import Event
from .langchain import LucidicLangchainHandler
from .providers.anthropic_handler import AnthropicHandler
from .providers.openai_handler import OpenAIHandler
from .session import Session
from .state import State
from .step import Step

ProviderType = Literal["openai", "anthropic", "langchain"]


def init(
    session_name: str,
    lucidic_api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    mass_sim_id: Optional[str] = None,
    task: Optional[str] = None,
    provider: Optional[ProviderType] = None,
) -> bool:
    if lucidic_api_key is None:
        lucidic_api_key = os.getenv("LUCIDIC_API_KEY", None)
        if lucidic_api_key is None:
            raise APIKeyVerificationError("Make sure to either pass your API key into lai.init() or set the LUCIDIC_API_KEY environment variable.")
    if agent_id is None:
        agent_id = os.getenv("LUCIDIC_AGENT_ID", None)
        if agent_id is None:
            raise InvalidOperationError("Lucidic agent ID not specified. Make sure to either pass your agent ID into lai.init() or set the LUCIDIC_AGENT_ID environment variable.")
    try:
        client = Client(
            lucidic_api_key=lucidic_api_key,
            agent_id=agent_id,
            session_name=session_name,
            mass_sim_id=mass_sim_id,
            task=task
        )
        # Set up provider
        if provider == "openai":
            client.set_provider(OpenAIHandler(client))
        elif provider == "anthropic":
            client.set_provider(AnthropicHandler(client))
        elif provider == "langchain":
            print(f"[Lucidic] For LangChain, make sure to create a handler and attach it to your top-level Agent class.")
        client.init_session()
        return True
    except Exception as e:
        raise e
        

def reset() -> None:
    Client().reset()


def create_step(
    state: Optional[str] = None, 
    action: Optional[str] = None, 
    goal: Optional[str] = None
) -> Step:
    client = Client()
    if not client.session:
        raise LucidicNotInitializedError()
    state = state or "state not provided"
    return client.session.create_step(state=state, action=action, goal=goal)


def end_step(
    is_successful: bool,
    eval_score: Optional[float] = None,
    eval_description: Optional[float] = None,
    state: Optional[str] = None,
    action: Optional[str] = None,
    screenshot = None,
    screenshot_path = None
) -> None:
    client = Client()
    if not client.session:
        raise LucidicNotInitializedError()
    client.session.end_step(
        is_successful=is_successful,
        eval_score=eval_score,
        eval_description=eval_description,
        state=state, action=action, 
        screenshot=screenshot, 
        screenshot_path=screenshot_path
    )


def update_step(
    is_successful: Optional[bool] = None, 
    eval_score: Optional[float] = None,
    eval_description: Optional[float] = None,
    state: Optional[str] = None, 
    action: Optional[str] = None,
    goal: Optional[str] = None, 
    is_finished: Optional[bool] = None, 
    cost_added: Optional[float] = None
) -> None:
    client = Client()
    if not client.session:
        raise LucidicNotInitializedError
    client.session.update_step(
        is_successful=is_successful,
        eval_score=eval_score,
        eval_description=eval_description,
        state=state,
        action=action,
        goal=goal,
        is_finished=is_finished,
        cost_added=cost_added
    )

def create_event(
    description: Optional[str] = None,
    result: Optional[str] = None
) -> Event:
    client = Client()
    if not client.session:
        raise LucidicNotInitializedError()
    return client.session.create_event(description=description, result=result)


def end_event(
    is_successful: bool, 
    cost_added: Optional[float] = None, 
    model: Optional[str] = None
) -> None:
    client = Client()
    if not client.session:
        raise LucidicNotInitializedError()
    client.session.end_event(is_successful=is_successful, cost_added=cost_added, model=model)


def end_session(
    is_successful: bool
) -> None:
    client = Client()
    if client.session:
        client.session.end_session(is_successful=is_successful)
        client.clear_session()


def get_prompt(
    prompt_name: str, 
    variables: Optional[dict] = None
) -> str:
    client = Client()
    if not client.session:
        raise LucidicNotInitializedError()
    prompt = client.get_prompt(prompt_name)
    if variables:
        for key, val in variables.items():
            index = prompt.find("{{" + key +"}}")
            if index == -1:
                raise PromptError("Supplied variable not found in prompt")
            prompt = prompt.replace("{{" + key +"}}", str(val))
    if "{{" in prompt and "}}" in prompt and prompt.find("{{") < prompt.find("}}"):
        raise PromptError("Unreplaced variable left in prompt")
    return prompt


@atexit.register
def cleanup():
    original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        print(f"Lucidic AI Cleanup: This should only take a few seconds...")
        try:
            client = Client()
            end_session(True)
        except Exception as e:
            print(f"[Lucidic] Client not yet initialized, shutting down")
    finally:
        signal.signal(signal.SIGINT, original_handler)


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
    'end_event',
    'end_session',
    'get_prompt',
    'ProviderType',
    'APIKeyVerificationError',
    'LucidicNotInitializedError',
    'PromptError',
    'InvalidOperationError',
    'LucidicLangchainHandler',  # Add this to export the handler
]