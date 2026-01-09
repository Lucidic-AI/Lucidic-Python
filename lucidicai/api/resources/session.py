"""Session resource API operations."""
import logging
import threading
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..client import HttpClient

if TYPE_CHECKING:
    from ...client import LucidicAI
    from ...session_obj import Session
    from ...core.config import SDKConfig

logger = logging.getLogger("Lucidic")


def _truncate_id(id_str: Optional[str]) -> str:
    """Truncate ID for logging."""
    if not id_str:
        return "None"
    return f"{id_str[:8]}..." if len(id_str) > 8 else id_str


class SessionResource:
    """Handle session-related API operations."""

    def __init__(
        self,
        http: HttpClient,
        client: "LucidicAI",
        config: "SDKConfig",
        production: bool = False,
    ):
        """Initialize session resource.

        Args:
            http: HTTP client instance
            client: Parent LucidicAI client
            config: SDK configuration
            production: Whether to suppress errors in production mode
        """
        self.http = http
        self._client = client
        self._config = config
        self._production = production

    # ==================== High-Level Session Methods ====================

    def create(
        self,
        session_name: Optional[str] = None,
        session_id: Optional[str] = None,
        task: Optional[str] = None,
        tags: Optional[List[str]] = None,
        experiment_id: Optional[str] = None,
        datasetitem_id: Optional[str] = None,
        evaluators: Optional[List[str]] = None,
        auto_end: Optional[bool] = None,
        production_monitoring: bool = False,
    ) -> "Session":
        """Create a new session.

        Sessions track a unit of work and can be used as context managers.

        Args:
            session_name: Human-readable name for the session.
            session_id: Optional custom session ID. Auto-generated if not provided.
            task: Task description for the session.
            tags: List of tags for filtering/grouping.
            experiment_id: Link session to an experiment.
            datasetitem_id: Link session to a dataset item.
            evaluators: List of evaluator names to run.
            auto_end: Override client's auto_end setting for this session.
            production_monitoring: Enable lightweight production monitoring.

        Returns:
            A Session object that can be used as a context manager.

        Example:
            with client.sessions.create(session_name="My Session") as session:
                # Do work
                pass
        """
        # Late imports to avoid circular dependencies
        from ...session_obj import Session
        from ...core.errors import LucidicError
        from ...sdk.shutdown_manager import ShutdownManager, SessionState

        if not self._client.is_valid:
            if self._production:
                # Return a dummy session in production mode
                return Session(
                    client=self._client,
                    session_id=session_id or str(uuid.uuid4()),
                    session_name=session_name,
                    auto_end=False,
                )
            raise LucidicError("Client is not properly configured")

        # Use client's auto_end by default
        if auto_end is None:
            auto_end = self._config.auto_end

        # Generate session ID if not provided
        real_session_id = session_id or str(uuid.uuid4())

        # Build session parameters
        session_params: Dict[str, Any] = {
            "session_id": real_session_id,
            "session_name": session_name or "Unnamed Session",
            "agent_id": self._config.agent_id,
        }
        if task:
            session_params["task"] = task
        if tags:
            session_params["tags"] = tags
        if experiment_id:
            session_params["experiment_id"] = experiment_id
        if datasetitem_id:
            session_params["datasetitem_id"] = datasetitem_id
        if evaluators:
            session_params["evaluators"] = evaluators
        if production_monitoring:
            session_params["production_monitoring"] = True

        try:
            # Create via API
            response = self.create_session(session_params)
            real_session_id = response.get("session_id", real_session_id)
        except Exception as e:
            if self._production:
                logger.error(f"[SessionResource] Failed to create session: {e}")
            else:
                raise

        # Create Session object
        session = Session(
            client=self._client,
            session_id=real_session_id,
            session_name=session_name,
            auto_end=auto_end,
        )

        # Bind session context immediately
        session._bind_context()

        # Track session
        with self._client._session_lock:
            self._client._sessions[real_session_id] = session

        # Register with shutdown manager for auto-end
        if auto_end:
            shutdown_manager = ShutdownManager()
            state = SessionState(
                session_id=real_session_id,
                http_client=self._client._resources,
                auto_end=auto_end,
            )
            shutdown_manager.register_session(real_session_id, state)

        logger.debug(f"[SessionResource] Created session {real_session_id[:8]}...")
        return session

    async def acreate(
        self,
        session_name: Optional[str] = None,
        session_id: Optional[str] = None,
        task: Optional[str] = None,
        tags: Optional[List[str]] = None,
        experiment_id: Optional[str] = None,
        datasetitem_id: Optional[str] = None,
        evaluators: Optional[List[str]] = None,
        auto_end: Optional[bool] = None,
        production_monitoring: bool = False,
    ) -> "Session":
        """Create a new session (async version).

        See create() for full documentation.
        """
        # Late imports to avoid circular dependencies
        from ...session_obj import Session
        from ...core.errors import LucidicError
        from ...sdk.shutdown_manager import ShutdownManager, SessionState

        if not self._client.is_valid:
            if self._production:
                return Session(
                    client=self._client,
                    session_id=session_id or str(uuid.uuid4()),
                    session_name=session_name,
                    auto_end=False,
                )
            raise LucidicError("Client is not properly configured")

        if auto_end is None:
            auto_end = self._config.auto_end

        real_session_id = session_id or str(uuid.uuid4())

        session_params: Dict[str, Any] = {
            "session_id": real_session_id,
            "session_name": session_name or "Unnamed Session",
            "agent_id": self._config.agent_id,
        }
        if task:
            session_params["task"] = task
        if tags:
            session_params["tags"] = tags
        if experiment_id:
            session_params["experiment_id"] = experiment_id
        if datasetitem_id:
            session_params["datasetitem_id"] = datasetitem_id
        if evaluators:
            session_params["evaluators"] = evaluators
        if production_monitoring:
            session_params["production_monitoring"] = True

        try:
            response = await self.acreate_session(session_params)
            real_session_id = response.get("session_id", real_session_id)
        except Exception as e:
            if self._production:
                logger.error(f"[SessionResource] Failed to create session: {e}")
            else:
                raise

        session = Session(
            client=self._client,
            session_id=real_session_id,
            session_name=session_name,
            auto_end=auto_end,
        )

        session._bind_context()

        with self._client._session_lock:
            self._client._sessions[real_session_id] = session

        if auto_end:
            shutdown_manager = ShutdownManager()
            state = SessionState(
                session_id=real_session_id,
                http_client=self._client._resources,
                auto_end=auto_end,
            )
            shutdown_manager.register_session(real_session_id, state)

        logger.debug(f"[SessionResource] Created async session {real_session_id[:8]}...")
        return session

    def end(
        self,
        session_id: Optional[str] = None,
        is_successful: Optional[bool] = None,
        is_successful_reason: Optional[str] = None,
        session_eval: Optional[float] = None,
        session_eval_reason: Optional[str] = None,
    ) -> None:
        """End a session.

        Args:
            session_id: Session ID to end. If None, attempts to use current context session.
            is_successful: Whether the session was successful.
            is_successful_reason: Reason for success/failure status.
            session_eval: Evaluation score (0.0 to 1.0).
            session_eval_reason: Reason for the evaluation score.
        """
        from ...sdk.shutdown_manager import ShutdownManager

        if not self._client.is_valid:
            return

        # If no session_id, try to get from context
        if not session_id:
            from ...sdk.context import current_session_id
            session_id = current_session_id.get(None)

        if not session_id:
            logger.debug("[SessionResource] No session to end")
            return

        try:
            self.end_session(
                session_id=session_id,
                is_successful=is_successful,
                is_successful_reason=is_successful_reason,
                session_eval=session_eval,
                session_eval_reason=session_eval_reason,
            )
        except Exception as e:
            if self._production:
                logger.error(f"[SessionResource] Failed to end session: {e}")
            else:
                raise

        # Remove from tracking
        with self._client._session_lock:
            self._client._sessions.pop(session_id, None)

        # Unregister from shutdown manager
        shutdown_manager = ShutdownManager()
        shutdown_manager.unregister_session(session_id)

        logger.debug(f"[SessionResource] Ended session {session_id[:8]}...")

    async def aend(
        self,
        session_id: Optional[str] = None,
        is_successful: Optional[bool] = None,
        is_successful_reason: Optional[str] = None,
        session_eval: Optional[float] = None,
        session_eval_reason: Optional[str] = None,
    ) -> None:
        """End a session (async version).

        See end() for full documentation.
        """
        from ...sdk.shutdown_manager import ShutdownManager

        if not self._client.is_valid:
            return

        if not session_id:
            from ...sdk.context import current_session_id
            session_id = current_session_id.get(None)

        if not session_id:
            logger.debug("[SessionResource] No session to end")
            return

        try:
            await self.aend_session(
                session_id=session_id,
                is_successful=is_successful,
                is_successful_reason=is_successful_reason,
                session_eval=session_eval,
                session_eval_reason=session_eval_reason,
            )
        except Exception as e:
            if self._production:
                logger.error(f"[SessionResource] Failed to end session: {e}")
            else:
                raise

        with self._client._session_lock:
            self._client._sessions.pop(session_id, None)

        shutdown_manager = ShutdownManager()
        shutdown_manager.unregister_session(session_id)

        logger.debug(f"[SessionResource] Ended async session {session_id[:8]}...")

    # ==================== Low-Level HTTP Methods ====================

    def create_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new session via API.

        Args:
            params: Session parameters including:
                - session_name: Name of the session
                - agent_id: Agent ID
                - task: Optional task description
                - tags: Optional tags
                - etc.

        Returns:
            Created session data with session_id
        """
        session_id = params.get("session_id")
        session_name = params.get("session_name")
        logger.debug(
            f"[Session] create_session() called - "
            f"session_id={_truncate_id(session_id)}, name={session_name!r}, "
            f"params={list(params.keys())}"
        )

        response = self.http.post("initsession", params)

        resp_session_id = response.get("session_id") if response else None
        logger.debug(
            f"[Session] create_session() response - "
            f"session_id={_truncate_id(resp_session_id)}, response_keys={list(response.keys()) if response else 'None'}"
        )
        return response

    def get(self, session_id: str) -> Dict[str, Any]:
        """Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session data
        """
        return self.http.get(f"sessions/{session_id}")

    def update(self, session_id: str, **updates) -> Dict[str, Any]:
        """Update an existing session.

        Args:
            session_id: Session ID
            **updates: Fields to update (task, is_finished, etc.)

        Returns:
            Updated session data
        """
        logger.debug(
            f"[Session] update() called - "
            f"session_id={_truncate_id(session_id)}, updates={updates}"
        )

        # Add session_id to the updates payload
        updates["session_id"] = session_id
        response = self.http.put("updatesession", updates)

        logger.debug(
            f"[Session] update() response - "
            f"session_id={_truncate_id(session_id)}, response_keys={list(response.keys()) if response else 'None'}"
        )
        return response

    def end_session(
        self,
        session_id: str,
        is_successful: Optional[bool] = None,
        is_successful_reason: Optional[str] = None,
        session_eval: Optional[float] = None,
        session_eval_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """End a session via API.

        Args:
            session_id: Session ID
            is_successful: Whether session was successful
            is_successful_reason: Reason for success or failure
            session_eval: Session evaluation score
            session_eval_reason: Reason for evaluation

        Returns:
            Final session data
        """
        logger.debug(
            f"[Session] end_session() called - "
            f"session_id={_truncate_id(session_id)}, is_successful={is_successful}, "
            f"session_eval={session_eval}"
        )

        updates: Dict[str, Any] = {
            "is_finished": True
        }

        if is_successful is not None:
            updates["is_successful"] = is_successful

        if session_eval is not None:
            updates["session_eval"] = session_eval

        if session_eval_reason is not None:
            updates["session_eval_reason"] = session_eval_reason

        if is_successful_reason is not None:
            updates["is_successful_reason"] = is_successful_reason

        return self.update(session_id, **updates)

    def list(
        self,
        agent_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List sessions with optional filters.

        Args:
            agent_id: Filter by agent ID
            experiment_id: Filter by experiment ID
            limit: Maximum number of sessions
            offset: Pagination offset

        Returns:
            List of sessions and pagination info
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset
        }

        if agent_id:
            params["agent_id"] = agent_id

        if experiment_id:
            params["experiment_id"] = experiment_id

        return self.http.get("sessions", params)

    # ==================== Asynchronous HTTP Methods ====================

    async def acreate_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new session via API (asynchronous).

        Args:
            params: Session parameters

        Returns:
            Created session data with session_id
        """
        session_id = params.get("session_id")
        session_name = params.get("session_name")
        logger.debug(
            f"[Session] acreate_session() called - "
            f"session_id={_truncate_id(session_id)}, name={session_name!r}, "
            f"params={list(params.keys())}"
        )

        response = await self.http.apost("initsession", params)

        resp_session_id = response.get("session_id") if response else None
        logger.debug(
            f"[Session] acreate_session() response - "
            f"session_id={_truncate_id(resp_session_id)}, response_keys={list(response.keys()) if response else 'None'}"
        )
        return response

    async def aget(self, session_id: str) -> Dict[str, Any]:
        """Get a session by ID (asynchronous).

        Args:
            session_id: Session ID

        Returns:
            Session data
        """
        return await self.http.aget(f"sessions/{session_id}")

    async def aupdate(self, session_id: str, **updates) -> Dict[str, Any]:
        """Update an existing session (asynchronous).

        Args:
            session_id: Session ID
            **updates: Fields to update (task, is_finished, etc.)

        Returns:
            Updated session data
        """
        logger.debug(
            f"[Session] aupdate() called - "
            f"session_id={_truncate_id(session_id)}, updates={updates}"
        )

        updates["session_id"] = session_id
        response = await self.http.aput("updatesession", updates)

        logger.debug(
            f"[Session] aupdate() response - "
            f"session_id={_truncate_id(session_id)}, response_keys={list(response.keys()) if response else 'None'}"
        )
        return response

    async def aend_session(
        self,
        session_id: str,
        is_successful: Optional[bool] = None,
        is_successful_reason: Optional[str] = None,
        session_eval: Optional[float] = None,
        session_eval_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """End a session via API (asynchronous).

        Args:
            session_id: Session ID
            is_successful: Whether session was successful
            is_successful_reason: Reason for success or failure
            session_eval: Session evaluation score
            session_eval_reason: Reason for evaluation

        Returns:
            Final session data
        """
        logger.debug(
            f"[Session] aend_session() called - "
            f"session_id={_truncate_id(session_id)}, is_successful={is_successful}, "
            f"session_eval={session_eval}"
        )

        updates: Dict[str, Any] = {
            "is_finished": True
        }

        if is_successful is not None:
            updates["is_successful"] = is_successful

        if session_eval is not None:
            updates["session_eval"] = session_eval

        if session_eval_reason is not None:
            updates["session_eval_reason"] = session_eval_reason

        if is_successful_reason is not None:
            updates["is_successful_reason"] = is_successful_reason

        return await self.aupdate(session_id, **updates)

    async def alist(
        self,
        agent_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List sessions with optional filters (asynchronous).

        Args:
            agent_id: Filter by agent ID
            experiment_id: Filter by experiment ID
            limit: Maximum number of sessions
            offset: Pagination offset

        Returns:
            List of sessions and pagination info
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset
        }

        if agent_id:
            params["agent_id"] = agent_id

        if experiment_id:
            params["experiment_id"] = experiment_id

        return await self.http.aget("sessions", params)
