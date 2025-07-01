import atexit
import logging
import os
import signal
from typing import List, Literal, Optional

from .client import Client
from .errors import APIKeyVerificationError, InvalidOperationError, LucidicNotInitializedError, PromptError
from .event import Event
from .providers.anthropic_handler import AnthropicHandler
from .providers.langchain import LucidicLangchainHandler
from .providers.openai_handler import OpenAIHandler
from .providers.openai_agents_handler import OpenAIAgentsHandler
from .providers.pydantic_ai_handler import PydanticAIHandler
from .session import Session
from .step import Step

ProviderType = Literal["openai", "anthropic", "langchain", "pydantic_ai", "openai_agents"]

# Configure logging
logger = logging.getLogger("Lucidic")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[Lucidic] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _setup_providers(client: Client, providers: List[ProviderType]) -> None:
    """Set up providers for the client, avoiding duplication
    
    Args:
        client: The Lucidic client instance
        providers: List of provider types to set up
    """
    # Track which providers have been set up to avoid duplication
    setup_providers = set()
    
    for provider in providers:
        if provider in setup_providers:
            continue
            
        if provider == "openai":
            client.set_provider(OpenAIHandler())
            setup_providers.add("openai")
        elif provider == "anthropic":
            client.set_provider(AnthropicHandler())
            setup_providers.add("anthropic")
        elif provider == "langchain":
            logger.info("For LangChain, make sure to create a handler and attach it to your top-level Agent class.")
            setup_providers.add("langchain")
        elif provider == "pydantic_ai":
            client.set_provider(PydanticAIHandler())
            setup_providers.add("pydantic_ai")
        elif provider == "openai_agents":
            try:
                # For OpenAI Agents SDK, we want both handlers
                client.set_provider(OpenAIAgentsHandler())
                setup_providers.add("openai_agents")
                # Also enable OpenAI handler if not already set up
                if "openai" not in setup_providers:
                    client.set_provider(OpenAIHandler())
                    setup_providers.add("openai")
            except Exception as e:
                logger.error(f"Failed to set up OpenAI Agents provider: {e}")
                raise

__all__ = [
    'Client',
    'Session',
    'Step',
    'Event',
    'init',
    'continue_session',
    'create_step',
    'end_step',
    'update_step',
    'create_event',
    'update_event',
    'end_event',
    'end_session',
    'get_prompt',
    'get_session',
    'ProviderType',
    'APIKeyVerificationError',
    'LucidicNotInitializedError',
    'PromptError',
    'InvalidOperationError',
    'LucidicLangchainHandler',
    'AnthropicHandler',
    'OpenAIHandler',
    'OpenAIAgentsHandler',
    'PydanticAIHandler'
]


def init(
    session_name: str,
    lucidic_api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    task: Optional[str] = None,
    providers: Optional[List[ProviderType]] = [],
    mass_sim_id: Optional[str] = None,
    rubrics: Optional[list] = None,
    tags: Optional[list] = None,
) -> str:
    """
    Initialize the Lucidic client.
    
    Args:
        session_name: The name of the session.
        lucidic_api_key: API key for authentication. If not provided, will use the LUCIDIC_API_KEY environment variable.
        agent_id: Agent ID. If not provided, will use the LUCIDIC_AGENT_ID environment variable.
        task: Task description.
        providers: List of provider types ("openai", "anthropic", "langchain", "pydantic_ai").
        mass_sim_id: Optional mass simulation ID, if session is to be part of a mass simulation.
        rubrics: Optional rubrics for evaluation, list of strings.
        tags: Optional tags for the session, list of strings.
    
    Raises:
        InvalidOperationError: If the client is already initialized.
        APIKeyVerificationError: If the API key is invalid.
    """
    if lucidic_api_key is None:
        lucidic_api_key = os.getenv("LUCIDIC_API_KEY", None)
        if lucidic_api_key is None:
            raise APIKeyVerificationError("Make sure to either pass your API key into lai.init() or set the LUCIDIC_API_KEY environment variable.")
    if agent_id is None:
        agent_id = os.getenv("LUCIDIC_AGENT_ID", None)
        if agent_id is None:
            raise APIKeyVerificationError("Lucidic agent ID not specified. Make sure to either pass your agent ID into lai.init() or set the LUCIDIC_AGENT_ID environment variable.")

    # get current client which will be NullClient if never lai is never initialized
    client = Client()
    # ff not yet initialized or still the NullClient -> creaet a real client when init is called
    if not getattr(client, 'initialized', False):
        client = Client(lucidic_api_key=lucidic_api_key, agent_id=agent_id)
    
    # Set up providers
    _setup_providers(client, providers)
    session_id = client.init_session(
        session_name=session_name,
        mass_sim_id=mass_sim_id,
        task=task,
        rubrics=rubrics,
        tags=tags
    )
    logger.info("Session initialized successfully")
    return session_id


def continue_session(
    session_id: str,
    lucidic_api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    providers: Optional[List[ProviderType]] = []
):
    if lucidic_api_key is None:
        lucidic_api_key = os.getenv("LUCIDIC_API_KEY", None)
        if lucidic_api_key is None:
            raise APIKeyVerificationError("Make sure to either pass your API key into lai.init() or set the LUCIDIC_API_KEY environment variable.")
    if agent_id is None:
        agent_id = os.getenv("LUCIDIC_AGENT_ID", None)
        if agent_id is None:
            raise APIKeyVerificationError("Lucidic agent ID not specified. Make sure to either pass your agent ID into lai.init() or set the LUCIDIC_AGENT_ID environment variable.")
    try:
        client = Client()
        if client.session:
            raise InvalidOperationError("[Lucidic] Session already in progress. Please call lai.end_session() first.")
    except LucidicNotInitializedError:
        client = Client(
            lucidic_api_key=lucidic_api_key,
            agent_id=agent_id,
        )
    
    # Set up providers
    _setup_providers(client, providers)
    session_id = client.continue_session(session_id=session_id)
    logger.info(f"Session {session_id} continuing...")
    return session_id  # For consistency


def update_session(
    task: Optional[str] = None,
    session_eval: Optional[float] = None,
    session_eval_reason: Optional[str] = None,
    is_successful: Optional[bool] = None,
    is_successful_reason: Optional[str] = None
) -> None:
    """
    Update the current session.
    
    Args:
        task: Task description.
        session_eval: Session evaluation.
        session_eval_reason: Session evaluation reason.
        is_successful: Whether the session was successful.
        is_successful_reason: Session success reason.
    """
    client = Client()
    if not client.session:
        return
    client.session.update_session(**locals())


def end_session(
    session_eval: Optional[float] = None,
    session_eval_reason: Optional[str] = None,
    is_successful: Optional[bool] = None,
    is_successful_reason: Optional[str] = None
) -> None:
    """
    End the current session.
    
    Args:
        session_eval: Session evaluation.
        session_eval_reason: Session evaluation reason.
        is_successful: Whether the session was successful.
        is_successful_reason: Session success reason.
    """
    client = Client()
    if not client.session:
        return
    client.session.update_session(is_finished=True, **locals())
    client.clear()


def reset_sdk() -> None:
    """
    Reset the SDK.
    """
    client = Client()
    if not client.initialized:
        return
    client.clear()


def create_mass_sim(
    mass_sim_name: str,
    total_num_sessions: int,
    lucidic_api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    task: Optional[str] = None,
    tags: Optional[list] = None
) -> str:
    """
    Create a new mass simulation.
    
    Args:
        mass_sim_name: Name of the mass simulation.
        total_num_sessions: Total intended number of sessions. More sessions can be added later.
        lucidic_api_key: API key for authentication. If not provided, will use the LUCIDIC_API_KEY environment variable.
        agent_id: Agent ID. If not provided, will use the LUCIDIC_AGENT_ID environment variable.
        task: Task description.
        tags: Tags for the mass simulation.
    
    Returns:
        mass_sim_id: ID of the created mass simulation. Pass this to lai.init() to create a new session in the mass sim.
    """
    if lucidic_api_key is None:
        lucidic_api_key = os.getenv("LUCIDIC_API_KEY", None)
        if lucidic_api_key is None:
            raise APIKeyVerificationError("Make sure to either pass your API key into lai.init() or set the LUCIDIC_API_KEY environment variable.")
    if agent_id is None:
        agent_id = os.getenv("LUCIDIC_AGENT_ID", None)
        if agent_id is None:
            raise APIKeyVerificationError("Lucidic agent ID not specified. Make sure to either pass your agent ID into lai.init() or set the LUCIDIC_AGENT_ID environment variable.")
    try:
        client = Client()
    except LucidicNotInitializedError:
        client = Client( # TODO: fail hard if incorrect API key or agent ID provided and wrong, fail silently if not provided
            lucidic_api_key=lucidic_api_key,
            agent_id=agent_id,
        )
    mass_sim_id = client.init_mass_sim(mass_sim_name=mass_sim_name, total_num_sims=total_num_sessions, task=task, tags=tags)  # TODO: change total_num_sims to total_num_sessions everywhere
    logger.info(f"Created mass simulation with ID: {mass_sim_id}")
    return mass_sim_id


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
        return
    return client.session.create_step(**locals())


def update_step(
    step_id: Optional[str] = None,
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
        step_id: ID of the step to update.
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
        return
    if not client.session.active_step:
        raise InvalidOperationError("No active step to update")
    client.session.update_step(**locals())


def end_step(
    step_id: Optional[str] = None,
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
        step_id: ID of the step to end.
        state: State description.
        action: Action description.
        goal: Goal description.
        eval_score: Evaluation score.
        eval_description: Evaluation description.
        screenshot: Screenshot encoded in base64. Provide either screenshot or screenshot_path.
        screenshot_path: Screenshot path.
    """
    client = Client()
    if not client.session:
        return
    
    if not client.session.active_step and step_id is None:
        raise InvalidOperationError("No active step to end")
    
    # Filter out None values from locals
    params = locals()
    kwargs = {k: v for k, v in params.items() if v is not None and k not in ['client', 'params']}
    kwargs['is_finished'] = True
    
    client.session.update_step(**kwargs)


def create_event(
    step_id: Optional[str] = None,
    description: Optional[str] = None,
    result: Optional[str] = None,
    cost_added: Optional[float] = None, 
    model: Optional[str] = None,
    screenshots: Optional[List[str]] = None
) -> str:
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
        return
    return client.session.create_event(**locals())


def update_event(
    event_id: Optional[str] = None,
    description: Optional[str] = None,
    result: Optional[str] = None,
    cost_added: Optional[float] = None, 
    model: Optional[str] = None,
    screenshots: Optional[List[str]] = None
) -> None:
    """
    Update the event with the given ID in the current step.
    
    Args:
        event_id: ID of the event to update.
        description: Description of the event.
        result: Result of the event.
        cost_added: Cost added by the event.
        model: Model used for the event.
        screenshots: List of screenshots encoded in base64.
    """
    client = Client()
    if not client.session:
        return
    client.session.update_event(**locals())


def end_event(
    event_id: Optional[str] = None,
    description: Optional[str] = None,
    result: Optional[str] = None,
    cost_added: Optional[float] = None, 
    model: Optional[str] = None,
    screenshots: Optional[List[str]] = None
) -> None:
    """
    End the latest event in the current step.
    
    Args:
        event_id: ID of the event to end.
        description: Description of the event.
        result: Result of the event.
        cost_added: Cost added by the event.
        model: Model used for the event.
    """
    client = Client()
    if not client.session:
        return
    client.session.update_event(is_finished=True, **locals())


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
        return ""
    prompt = client.get_prompt(prompt_name, cache_ttl, label)
    if variables:
        for key, val in variables.items():
            index = prompt.find("{{" + key +"}}")
            if index == -1:
                raise PromptError("Supplied variable not found in prompt")
            prompt = prompt.replace("{{" + key +"}}", str(val))
    if "{{" in prompt and "}}" in prompt and prompt.find("{{") < prompt.find("}}"):
        logger.warning("Unreplaced variable(s) left in prompt. Please check your prompt.")
    return prompt


def get_session():
    """Get the current session object
    
    Returns:
        Session: The current session object, or None if no session exists
    """
    try:
        client = Client()
        return client.session
    except (LucidicNotInitializedError, AttributeError) as e:
        logger.debug(f"No active session: {str(e)}")
        return None


