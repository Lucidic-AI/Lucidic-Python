from typing import Optional, Union, Literal

from .client import Client
from .session import Session
from .step import Step
from .event import Event
from .action import Action
from .state import State
from .errors import APIKeyVerificationError, SessionHTTPError
from .providers.openai_handler import OpenAIHandler
from .providers.anthropic_handler import AnthropicHandler
from .langchain import LucidicLangchainHandler

ProviderType = Literal["openai", "anthropic", "langchain"]

def init(
    lucidic_api_key: str,
    agent_id: str,
    session_name: str,
    mass_sim_id: Optional[str] = None,
    task: Optional[str] = None,
    provider: Optional[ProviderType] = None,
) -> Union[Session, None]:
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
            # Create and store Langchain handler on client
            client.langchain_handler = LucidicLangchainHandler(client)
            
        return client.init_session()
    except APIKeyVerificationError as e:
        print(f"Failed to initialize client: {e}")
        return None
        
def configure(
    lucidic_api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_name: Optional[str] = None,
    mass_sim_id: Optional[str] = None,
    task: Optional[str] = None,
) -> None:
    Client().configure(
        lucidic_api_key=lucidic_api_key,
        agent_id=agent_id,
        session_name=session_name,
        mass_sim_id=mass_sim_id,
        task=task
    )

def create_step(state: Optional[str] = None, action: Optional[str] = None, goal: Optional[str] = None) -> Step:
    client = Client()
    if not client.session:
        raise ValueError("No active session. Call init() first")
    state = state or "state not provided"
    return client.session.create_step(state=state, action=action, goal=goal)

def finish_step(is_successful: bool, state: Optional[str] = None, action: Optional[str] = None) -> None:
    client = Client()
    if not client.session:
        raise ValueError("No active session. Call init() first")
    client.session.finish_step(is_successful=is_successful, state=state, action=action)

def update_step(
    is_successful: Optional[bool] = None, state: Optional[str] = None, action: Optional[str] = None,
    goal: Optional[str] = None, is_finished: Optional[bool] = None, cost_added: Optional[float] = None) -> None:
    client = Client()
    if not client.session:
        raise ValueError("No active session. Call init() first")
    client.session.update_step(
        is_successful=is_successful,
        state=state,
        action=action,
        goal=goal,
        is_finished=is_finished,
        cost_added=cost_added
    )

def create_event(description: Optional[str] = None, result: Optional[str] = None) -> Event:
    client = Client()
    if not client.session:
        raise ValueError("No active session. Call init() first")
    return client.session.create_event(description=description, result=result)

def end_event(is_successful: bool, cost_added: Optional[float] = None, model: Optional[str] = None) -> None:
    client = Client()
    if not client.session:
        raise ValueError("No active session. Call init() first")
    client.session.end_event(is_successful=is_successful, cost_added=cost_added, model=model)

def end_session(is_successful: bool) -> None:
    client = Client()
    if client.session:
        client.session.finish_session(is_successful=is_successful)
        client.clear_session()

def get_prompt(prompt_name: str, variables: Optional[dict] = None) -> str:
    client = Client()
    if not client.session:
        # TODO: Write more descriptive Exception classes 
        raise Exception("No session")
    # TODO: make more efficient? maybe experiment on string length, naive way might be faster until string is very large
    prompt = client.get_prompt(prompt_name)
    for key, val in variables.items():
        index = prompt.find("{{" + key +"}}")
        if index == -1:
            raise Exception("Supplied variable not found in prompt")
        prompt = prompt.replace("{{" + key +"}}", val)
    if "{{" in prompt and "}}" in prompt and prompt.find("{{") < prompt.find("}}"):
        raise Exception("Unreplaced variable left in prompt")
    return prompt

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
    'finish_step',
    'update_step',
    'create_event',
    'end_event',
    'end_session',
    'get_prompt',
    'ProviderType',
    'APIKeyVerificationError',
    'SessionHTTPError',
    'LucidicLangchainHandler',  # Add this to export the handler
]