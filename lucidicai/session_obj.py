"""Session object - Represents an active session bound to a LucidicAI client.

A Session is returned by client.sessions.create() and provides:
- Context manager support for automatic session binding and cleanup
- Methods to update and end the session
- Access to the parent client for operations within the session
"""

from typing import Optional, List, Any, TYPE_CHECKING
import logging

from .sdk.context import current_session_id, current_client

if TYPE_CHECKING:
    from .client import LucidicAI

logger = logging.getLogger("Lucidic")


class Session:
    """Represents an active session bound to a LucidicAI client.

    Sessions track a unit of work (conversation, workflow, etc.) and provide:
    - Automatic context binding when used as a context manager
    - Methods to update session metadata and end the session
    - Reference to the parent client for additional operations

    Example:
        client = LucidicAI(api_key="...", providers=["openai"])

        # Using as context manager (recommended)
        with client.sessions.create(session_name="My Session") as session:
            # Session is active, LLM calls are tracked
            response = openai_client.chat.completions.create(...)

        # Manual usage
        session = client.sessions.create(session_name="Manual Session", auto_end=False)
        try:
            # Do work
            pass
        finally:
            session.end()
    """

    def __init__(
        self,
        client: "LucidicAI",
        session_id: str,
        session_name: Optional[str] = None,
        auto_end: bool = True,
    ):
        """Initialize a Session.

        Args:
            client: The LucidicAI client that owns this session
            session_id: The unique session identifier
            session_name: Optional human-readable name for the session
            auto_end: Whether to automatically end the session on context exit
        """
        self._client = client
        self._session_id = session_id
        self._session_name = session_name
        self._auto_end = auto_end
        self._ended = False
        self._context_token = None
        self._client_token = None

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id

    @property
    def session_name(self) -> Optional[str]:
        """Get the session name."""
        return self._session_name

    @property
    def client(self) -> "LucidicAI":
        """Get the parent client."""
        return self._client

    @property
    def is_finished(self) -> bool:
        """Check if the session has been ended."""
        return self._ended

    def update(
        self,
        task: Optional[str] = None,
        session_eval: Optional[float] = None,
        session_eval_reason: Optional[str] = None,
        is_successful: Optional[bool] = None,
        is_successful_reason: Optional[str] = None,
    ) -> None:
        """Update the session metadata.

        Args:
            task: Update the task description
            session_eval: Evaluation score (0.0 to 1.0)
            session_eval_reason: Reason for the evaluation score
            is_successful: Whether the session was successful
            is_successful_reason: Reason for success/failure status
        """
        if self._ended:
            logger.warning(f"[Session] Attempted to update ended session {self._session_id[:8]}...")
            return

        updates = {}
        if task is not None:
            updates["task"] = task
        if session_eval is not None:
            updates["session_eval"] = session_eval
        if session_eval_reason is not None:
            updates["session_eval_reason"] = session_eval_reason
        if is_successful is not None:
            updates["is_successful"] = is_successful
        if is_successful_reason is not None:
            updates["is_successful_reason"] = is_successful_reason

        if updates:
            logger.debug(
                f"[Session] update() called - session_id={self._session_id[:8]}..., updates={updates}"
            )
            self._client.sessions.update(self._session_id, updates)
            logger.debug(f"[Session] update() completed for session {self._session_id[:8]}...")
        else:
            logger.debug(f"[Session] update() called with no updates for session {self._session_id[:8]}...")

    async def aupdate(
        self,
        task: Optional[str] = None,
        session_eval: Optional[float] = None,
        session_eval_reason: Optional[str] = None,
        is_successful: Optional[bool] = None,
        is_successful_reason: Optional[str] = None,
    ) -> None:
        """Update the session metadata (async version).

        Args:
            task: Update the task description
            session_eval: Evaluation score (0.0 to 1.0)
            session_eval_reason: Reason for the evaluation score
            is_successful: Whether the session was successful
            is_successful_reason: Reason for success/failure status
        """
        if self._ended:
            logger.warning(f"[Session] Attempted to update ended session {self._session_id[:8]}...")
            return

        updates = {}
        if task is not None:
            updates["task"] = task
        if session_eval is not None:
            updates["session_eval"] = session_eval
        if session_eval_reason is not None:
            updates["session_eval_reason"] = session_eval_reason
        if is_successful is not None:
            updates["is_successful"] = is_successful
        if is_successful_reason is not None:
            updates["is_successful_reason"] = is_successful_reason

        if updates:
            logger.debug(
                f"[Session] aupdate() called - session_id={self._session_id[:8]}..., updates={updates}"
            )
            await self._client.sessions.aupdate(self._session_id, updates)
            logger.debug(f"[Session] aupdate() completed for session {self._session_id[:8]}...")
        else:
            logger.debug(f"[Session] aupdate() called with no updates for session {self._session_id[:8]}...")

    def end(
        self,
        is_successful: Optional[bool] = None,
        is_successful_reason: Optional[str] = None,
        session_eval: Optional[float] = None,
        session_eval_reason: Optional[str] = None,
    ) -> None:
        """End the session.

        Args:
            is_successful: Whether the session was successful
            is_successful_reason: Reason for success/failure status
            session_eval: Evaluation score (0.0 to 1.0)
            session_eval_reason: Reason for the evaluation score
        """
        if self._ended:
            return

        # Unbind context before ending
        self._unbind_context()

        self._client.sessions.end(
            session_id=self._session_id,
            is_successful=is_successful,
            is_successful_reason=is_successful_reason,
            session_eval=session_eval,
            session_eval_reason=session_eval_reason,
        )
        self._ended = True

    async def aend(
        self,
        is_successful: Optional[bool] = None,
        is_successful_reason: Optional[str] = None,
        session_eval: Optional[float] = None,
        session_eval_reason: Optional[str] = None,
    ) -> None:
        """End the session (async version).

        Args:
            is_successful: Whether the session was successful
            is_successful_reason: Reason for success/failure status
            session_eval: Evaluation score (0.0 to 1.0)
            session_eval_reason: Reason for the evaluation score
        """
        if self._ended:
            return

        # Unbind context before ending
        self._unbind_context()

        await self._client.sessions.aend(
            session_id=self._session_id,
            is_successful=is_successful,
            is_successful_reason=is_successful_reason,
            session_eval=session_eval,
            session_eval_reason=session_eval_reason,
        )
        self._ended = True

    def _bind_context(self) -> None:
        """Bind this session to the current context."""
        self._context_token = current_session_id.set(self._session_id)
        self._client_token = current_client.set(self._client)

    def _unbind_context(self) -> None:
        """Unbind this session from the current context."""
        if self._context_token:
            current_session_id.reset(self._context_token)
            self._context_token = None
        if self._client_token:
            current_client.reset(self._client_token)
            self._client_token = None

    def __enter__(self) -> "Session":
        """Enter the session context - binds session to current context."""
        # Only bind if not already bound (create_session() already binds context)
        if self._context_token is None:
            self._bind_context()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the session context - unbinds and optionally ends the session."""
        # Flush telemetry before ending
        try:
            from .telemetry.telemetry_manager import get_telemetry_manager

            manager = get_telemetry_manager()
            if manager.is_telemetry_initialized:
                logger.debug(
                    f"[Session] Flushing telemetry for session {self._session_id}"
                )
                flush_success = manager.force_flush(timeout_millis=5000)
                if not flush_success:
                    logger.warning("[Session] Telemetry flush may be incomplete")
        except Exception as e:
            logger.debug(f"[Session] Error flushing telemetry: {e}")

        # Unbind context
        self._unbind_context()

        # End session if auto_end is enabled
        if self._auto_end and not self._ended:
            try:
                self.end()
            except Exception as e:
                # Don't mask the original exception
                logger.debug(f"[Session] Error ending session: {e}")

    async def __aenter__(self) -> "Session":
        """Enter the async session context - binds session to current context."""
        # Only bind if not already bound (acreate_session() already binds context)
        if self._context_token is None:
            self._bind_context()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async session context - unbinds and optionally ends the session."""
        import asyncio

        # Flush telemetry before ending
        try:
            from .telemetry.telemetry_manager import get_telemetry_manager

            manager = get_telemetry_manager()
            if manager.is_telemetry_initialized:
                logger.debug(
                    f"[Session] Flushing telemetry for async session {self._session_id}"
                )
                flush_success = manager.force_flush(timeout_millis=5000)
                if not flush_success:
                    logger.warning("[Session] Telemetry flush may be incomplete")
        except Exception as e:
            logger.debug(f"[Session] Error flushing telemetry: {e}")

        # Unbind context
        self._unbind_context()

        # End session if auto_end is enabled
        if self._auto_end and not self._ended:
            try:
                await self.aend()
            except Exception as e:
                # Don't mask the original exception
                logger.debug(f"[Session] Error ending async session: {e}")

    def __repr__(self) -> str:
        status = "ended" if self._ended else "active"
        name_part = f", name={self._session_name!r}" if self._session_name else ""
        return f"<Session(id={self._session_id[:8]}...{name_part}, status={status})>"
