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

ProviderType = Literal["openai", "anthropic"]

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
        
        if provider == "openai":
            client.set_provider(OpenAIHandler(client))
        elif provider == "anthropic":
            client.set_provider(AnthropicHandler(client))
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
    # After init(), Client() returns existing singleton
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
    'create_event',
    'end_event',
    'end_session',
    'ProviderType',
    'APIKeyVerificationError',
    'SessionHTTPError',
    
]