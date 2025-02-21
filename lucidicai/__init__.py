from typing import Optional, Union, Literal

from .client import Client
from .session import Session
from .errors import APIKeyVerificationError
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

def end_session(is_successful: bool):
    """End the current session"""
    if Client().session:
        Client().session.finish_session(is_successful=is_successful)
        Client().clear_session()

__all__ = ['Client', 'Session', 'init', 'configure', 'end_session', 'ProviderType']