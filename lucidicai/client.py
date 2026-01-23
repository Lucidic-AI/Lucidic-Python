"""LucidicAI Client - Instance-based SDK for AI observability.

This module provides the main LucidicAI class, which is the entry point
for all SDK operations. Each client instance maintains its own state,
HTTP connections, and telemetry context.

Example:
    from lucidicai import LucidicAI

    client = LucidicAI(api_key="...", agent_id="...", providers=["openai"])

    with client.create_session(session_name="My Session") as session:
        @client.event
        def my_function():
            # LLM calls are automatically tracked
            pass
        my_function()
"""

import logging
import os
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .api.client import HttpClient
from .api.resources.session import SessionResource
from .api.resources.event import EventResource
from .api.resources.dataset import DatasetResource
from .api.resources.experiment import ExperimentResource
from .api.resources.prompt import PromptResource
from .api.resources.feature_flag import FeatureFlagResource
from .api.resources.evals import EvalsResource
from .core.config import SDKConfig
from .core.errors import LucidicError
from .session_obj import Session
from .sdk.shutdown_manager import ShutdownManager, SessionState

logger = logging.getLogger("Lucidic")

F = TypeVar("F", bound=Callable[..., Any])


def get_shutdown_manager() -> ShutdownManager:
    """Get the singleton ShutdownManager instance."""
    return ShutdownManager()


class LucidicAI:
    """Instance-based Lucidic AI client for observability.

    Each LucidicAI instance maintains its own:
    - HTTP connections and configuration
    - API resources (sessions, events, datasets)
    - Active sessions
    - Telemetry registration

    Multiple clients can coexist with different configurations.

    Args:
        api_key: Lucidic API key. Falls back to LUCIDIC_API_KEY env var.
        agent_id: Agent identifier. Falls back to LUCIDIC_AGENT_ID env var.
        providers: List of LLM providers to instrument (e.g., ["openai", "anthropic"]).
        auto_end: Whether sessions auto-end on context exit or process shutdown.
        production: If True, suppress SDK errors. If None, checks LUCIDIC_PRODUCTION env var.
        region: Deployment region ("us", "india"). Falls back to LUCIDIC_REGION env var.
        base_url: Custom base URL for API requests. Takes precedence over region.
                  Falls back to LUCIDIC_BASE_URL env var.
        **kwargs: Additional configuration options passed to SDKConfig.

    Raises:
        ValueError: If required configuration is missing, invalid, or region is unrecognized.

    Example:
        # Basic usage
        client = LucidicAI(api_key="...", agent_id="...")

        # With providers for auto-instrumentation
        client = LucidicAI(
            api_key="...",
            agent_id="...",
            providers=["openai", "anthropic"]
        )

        # Production mode (suppress errors)
        client = LucidicAI(
            api_key="...",
            agent_id="...",
            production=True
        )

        # India region
        client = LucidicAI(
            api_key="...",
            agent_id="...",
            region="india"
        )

        # Custom base URL (e.g., self-hosted deployment)
        client = LucidicAI(
            api_key="...",
            agent_id="...",
            base_url="https://custom.example.com/api"
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        agent_id: Optional[str] = None,
        providers: Optional[List[str]] = None,
        auto_end: bool = True,
        production: Optional[bool] = None,
        region: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        # Generate unique client ID for telemetry routing
        self._client_id = str(uuid.uuid4())

        # Resolve production mode: arg > env > default (False)
        if production is None:
            production = os.getenv("LUCIDIC_PRODUCTION", "").lower() in ("true", "1")
        self._production = production

        # Build configuration
        self._config = SDKConfig.from_env(
            api_key=api_key,
            agent_id=agent_id,
            auto_end=auto_end,
            region=region,
            base_url=base_url,
            **kwargs,
        )

        # Validate configuration
        errors = self._config.validate()
        if errors:
            error_msg = f"Invalid configuration: {', '.join(errors)}"
            if self._production:
                logger.error(f"[LucidicAI] {error_msg}")
                # In production mode, allow initialization but mark as invalid
                self._valid = False
            else:
                raise ValueError(error_msg)
        else:
            self._valid = True

        # Initialize HTTP client
        self._http = HttpClient(self._config)

        # Initialize API resources
        self._resources: Dict[str, Any] = {
            "sessions": SessionResource(self._http, self, self._config, self._production),
            "events": EventResource(self._http, self._production),
            "datasets": DatasetResource(self._http, self._config.agent_id, self._production),
            "experiments": ExperimentResource(self._http, self._config.agent_id, self._production),
            "prompts": PromptResource(self._http, self._config, self._production),
            "feature_flags": FeatureFlagResource(self._http, self._config.agent_id, self._production),
            "evals": EvalsResource(self._http, self._production),
        }

        # Active sessions for this client
        self._sessions: Dict[str, Session] = {}
        self._session_lock = threading.Lock()

        # Store providers list
        self._providers = providers or []

        # Initialize telemetry if providers specified
        if self._providers:
            self._initialize_telemetry()

        # Register with shutdown manager
        shutdown_manager = get_shutdown_manager()
        shutdown_manager.register_client(self)

        logger.info(
            f"[LucidicAI] Initialized client {self._client_id[:8]}... "
            f"(production={self._production}, providers={self._providers})"
        )

    def _initialize_telemetry(self) -> None:
        """Initialize telemetry and register this client."""
        try:
            from .telemetry.telemetry_manager import get_telemetry_manager

            manager = get_telemetry_manager()
            manager.ensure_initialized(self._providers)
            manager.register_client(self)
            logger.debug(f"[LucidicAI] Registered with telemetry manager")
        except Exception as e:
            if self._production:
                logger.error(f"[LucidicAI] Failed to initialize telemetry: {e}")
            else:
                raise

    @property
    def client_id(self) -> str:
        """Get the unique client identifier."""
        return self._client_id

    @property
    def config(self) -> SDKConfig:
        """Get the client configuration."""
        return self._config

    @property
    def agent_id(self) -> Optional[str]:
        """Get the agent ID."""
        return self._config.agent_id

    @property
    def is_valid(self) -> bool:
        """Check if the client is properly configured."""
        return self._valid

    @property
    def experiments(self) -> ExperimentResource:
        """Access experiments resource.

        Example:
            experiment_id = client.experiments.create(
                experiment_name="My Experiment",
                description="Testing new model"
            )
        """
        return self._resources["experiments"]

    @property
    def prompts(self) -> PromptResource:
        """Access prompts resource.

        Example:
            prompt = client.prompts.get(
                prompt_name="greeting",
                variables={"name": "Alice"}
            )
        """
        return self._resources["prompts"]

    @property
    def feature_flags(self) -> FeatureFlagResource:
        """Access feature flags resource.

        Example:
            flag_value = client.feature_flags.get(
                flag_name="new_feature",
                default=False
            )
        """
        return self._resources["feature_flags"]

    @property
    def sessions(self) -> SessionResource:
        """Access sessions resource.

        Example:
            with client.sessions.create(session_name="My Session") as session:
                # Do work
                pass
        """
        return self._resources["sessions"]

    @property
    def events(self) -> EventResource:
        """Access events resource.

        Example:
            event_id = client.events.create(
                type="custom_event",
                data={"key": "value"}
            )
        """
        return self._resources["events"]

    @property
    def datasets(self) -> DatasetResource:
        """Access datasets resource.

        Example:
            dataset = client.datasets.get(dataset_id)
            client.datasets.create(name="My Dataset")
        """
        return self._resources["datasets"]

    @property
    def evals(self) -> EvalsResource:
        """Access evals resource for submitting evaluation results.

        Example:
            client.evals.emit(result=True, name="task_success")
            client.evals.emit(result=0.95, name="accuracy")
            client.evals.emit(result="excellent", name="quality")
        """
        return self._resources["evals"]

    # ==================== Decorators ====================

    def event(
        self, func: Optional[Callable] = None, **decorator_kwargs
    ) -> Callable[[F], F]:
        """Create an event decorator bound to this client.

        The decorator tracks function calls as events when the current
        context belongs to this client.

        Can be used with or without parentheses:
            @client.event
            def my_function(): ...

            @client.event()
            def my_function(): ...

            @client.event(tags=["important"])
            def my_function(): ...

        Args:
            func: The function to decorate (when used without parentheses).
            **decorator_kwargs: Additional event metadata.

        Returns:
            A decorator function or decorated function.

        Example:
            @client.event
            def process_data(data):
                # Function call is tracked as an event
                return result

            @client.event(tags=["important"])
            async def async_process(data):
                # Async functions are also supported
                return result
        """
        from .sdk.decorators import event as event_decorator

        decorator = event_decorator(client=self, **decorator_kwargs)

        # If func is provided, we're being used without parentheses
        if func is not None:
            return decorator(func)

        # Otherwise, return the decorator for later application
        return decorator

    # ==================== Lifecycle ====================

    def close(self) -> None:
        """Close the client and clean up resources.

        This ends all active sessions and unregisters from telemetry.
        """
        logger.info(f"[LucidicAI] Closing client {self._client_id[:8]}...")

        # Collect session IDs under lock (don't clear - let end() handle removal)
        with self._session_lock:
            session_ids = list(self._sessions.keys())

        # End sessions WITHOUT holding lock (HTTP calls can be slow)
        # end() will acquire lock and pop each session from _sessions
        for session_id in session_ids:
            try:
                self.sessions.end(session_id)
            except Exception as e:
                logger.debug(f"[LucidicAI] Error ending session on close: {e}")

        # Unregister from telemetry
        if self._providers:
            try:
                from .telemetry.telemetry_manager import get_telemetry_manager

                manager = get_telemetry_manager()
                manager.unregister_client(self._client_id)
            except Exception as e:
                logger.debug(f"[LucidicAI] Error unregistering from telemetry: {e}")

        # Unregister from shutdown manager
        try:
            shutdown_manager = get_shutdown_manager()
            shutdown_manager.unregister_client(self._client_id)
        except Exception as e:
            logger.debug(f"[LucidicAI] Error unregistering from shutdown manager: {e}")

        # Close HTTP client
        try:
            self._http.close()
        except Exception as e:
            logger.debug(f"[LucidicAI] Error closing HTTP client: {e}")

        logger.info(f"[LucidicAI] Client {self._client_id[:8]}... closed")

    async def aclose(self) -> None:
        """Close the client (async version)."""
        logger.info(f"[LucidicAI] Closing async client {self._client_id[:8]}...")

        # Collect session IDs under lock (don't clear - let aend() handle removal)
        with self._session_lock:
            session_ids = list(self._sessions.keys())

        # End sessions WITHOUT holding lock (HTTP calls can be slow)
        # aend() will acquire lock and pop each session from _sessions
        for session_id in session_ids:
            try:
                await self.sessions.aend(session_id)
            except Exception as e:
                logger.debug(f"[LucidicAI] Error ending session on close: {e}")

        if self._providers:
            try:
                from .telemetry.telemetry_manager import get_telemetry_manager

                manager = get_telemetry_manager()
                manager.unregister_client(self._client_id)
            except Exception as e:
                logger.debug(f"[LucidicAI] Error unregistering from telemetry: {e}")

        try:
            shutdown_manager = get_shutdown_manager()
            shutdown_manager.unregister_client(self._client_id)
        except Exception as e:
            logger.debug(f"[LucidicAI] Error unregistering from shutdown manager: {e}")

        try:
            await self._http.aclose()
        except Exception as e:
            logger.debug(f"[LucidicAI] Error closing HTTP client: {e}")

        logger.info(f"[LucidicAI] Async client {self._client_id[:8]}... closed")

    def __enter__(self) -> "LucidicAI":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.close()

    async def __aenter__(self) -> "LucidicAI":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.aclose()

    def __repr__(self) -> str:
        return (
            f"<LucidicAI(id={self._client_id[:8]}..., "
            f"agent_id={self._config.agent_id}, "
            f"sessions={len(self._sessions)})>"
        )
