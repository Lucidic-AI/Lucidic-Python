"""Session resource API operations."""
import logging
import threading
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, TYPE_CHECKING
from urllib.parse import quote

from ..client import HttpClient
from ..models import session as session_models
from ..models.base import CursorPage
from ..pagination import apaginate, paginate

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
        checkpoint_id: Optional[str] = None,
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
            checkpoint_id: Load an immutable Training Modules checkpoint.
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
        if checkpoint_id:
            session_params["checkpoint_id"] = checkpoint_id
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

        # LUC-608: when the user passes datasetitem_id they intend tool-backed
        # dispatch. Init per-session fixture state + bind MockContext so
        # subsequent @mockable calls in this session route through the backend.
        # ToolsResource._init_session NEVER raises — session creation must not
        # be blocked by tool-init failures, and the session works as normal
        # observability without it.
        if datasetitem_id and self._client._has_tools_resource():
            self._client.tools._init_session(real_session_id)

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
        checkpoint_id: Optional[str] = None,
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
        if checkpoint_id:
            session_params["checkpoint_id"] = checkpoint_id
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

        # LUC-608: see create() for full rationale. Async sibling.
        if datasetitem_id and self._client._has_tools_resource():
            await self._client.tools._ainit_session(real_session_id)

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

    # ==================== v2 reads (LUC-907) ====================
    #
    # These reads are data-bearing, so — unlike the telemetry lifecycle methods
    # above — they do NOT swallow errors in production: a failure raises the
    # typed LucidicError the transport decoded. Each has an async sibling.

    def get(self, session_id: str) -> "session_models.SessionTrace":
        """Read a session's detail + trace (LUC-817).

        Accepts the session UUID or the client-supplied ``custom_session_id``.
        Returns a ``SessionTrace`` (session meta + flat ``occurred_at``-ordered
        events); use ``.tree()`` to rebuild the parent/child tree.
        """
        return session_models.SessionTrace.from_dict(
            self.http.get(f"sdk/v2/sessions/{quote(session_id, safe='')}")
        )

    async def aget(self, session_id: str) -> "session_models.SessionTrace":
        """Async sibling of ``get``."""
        return session_models.SessionTrace.from_dict(
            await self.http.aget(f"sdk/v2/sessions/{quote(session_id, safe='')}")
        )

    def evaluator_results(self, session_id: str) -> "session_models.SessionEvaluatorResults":
        """A session's eval scores: session-level ``evals`` + event-level
        ``event_evals`` (accepts a UUID or ``custom_session_id``)."""
        return session_models.SessionEvaluatorResults.from_dict(
            self.http.get(f"sdk/v2/sessions/{quote(session_id, safe='')}/evaluator-results")
        )

    async def aevaluator_results(self, session_id: str) -> "session_models.SessionEvaluatorResults":
        """Async sibling of ``evaluator_results``."""
        return session_models.SessionEvaluatorResults.from_dict(
            await self.http.aget(f"sdk/v2/sessions/{quote(session_id, safe='')}/evaluator-results")
        )

    def event(self, session_id: str, event_id: str, *, raw: bool = False) -> "session_models.Event":
        """Read one event of a session. ``raw=True`` returns the complete
        payload — inline in ``.payload`` (<2 MB) or a short-lived presigned
        ``.blob_url`` for an offloaded payload (fetch the bytes with
        ``lucidicai.api.downloads.fetch_presigned``)."""
        params = {"raw": "true"} if raw else None
        return session_models.Event.from_dict(
            self.http.get(f"sdk/v2/sessions/{quote(session_id, safe='')}/events/{event_id}", params)
        )

    async def aevent(self, session_id: str, event_id: str, *, raw: bool = False) -> "session_models.Event":
        """Async sibling of ``event``."""
        params = {"raw": "true"} if raw else None
        return session_models.Event.from_dict(
            await self.http.aget(f"sdk/v2/sessions/{quote(session_id, safe='')}/events/{event_id}", params)
        )

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
        agent_id: str,
        *,
        experiment_id: Optional[str] = None,
        production: Optional[bool] = None,
        cost: Optional[str] = None,
        duration: Optional[str] = None,
        num_events: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        eval_bool: Optional[Any] = None,
        eval_number: Optional[Any] = None,
        eval_string: Optional[Any] = None,
        ordering: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> Iterator["session_models.Session"]:
        """Lazily iterate an agent's sessions, following pages (LUC-816).

        Filters (all optional): ``experiment_id``; ``production`` (bool);
        ``cost`` / ``duration`` / ``num_events`` as ``"min:max"`` range strings
        (either side may be blank); ``tags`` (list, AND-matched); ``status``;
        and the repeatable eval filters ``eval_bool`` (``"name:true"`` /
        ``"name:false"``), ``eval_number`` (``"name:min:max"``), ``eval_string``
        (``"name:value"``) — each takes a single string or a list of them.
        ``ordering`` accepts ``start_time`` / ``id`` (± prefix); backend default
        is newest-first.

        Lazy generator: request errors surface on first iteration. Use
        ``count()`` for a cheap total, ``list_page()`` for eager single-page.
        """
        base = self._filter_params(agent_id, {
            "experiment_id": experiment_id, "production": production, "cost": cost,
            "duration": duration, "num_events": num_events, "tags": tags,
            "status": status, "eval_bool": eval_bool, "eval_number": eval_number,
            "eval_string": eval_string, "ordering": ordering, "page_size": page_size,
        })
        return paginate(
            lambda cursor: self._page_get(base, cursor), model=session_models.Session
        )

    def alist(self, agent_id: str, **filters: Any) -> AsyncIterator["session_models.Session"]:
        """Async sibling of ``list`` (same filter kwargs)."""
        base = self._filter_params(agent_id, filters)
        return apaginate(
            lambda cursor: self._apage_get(base, cursor), model=session_models.Session
        )

    def list_page(
        self, agent_id: str, *, cursor: Optional[str] = None, **filters: Any
    ) -> CursorPage:
        """Fetch a single page of sessions (manual pagination control)."""
        return CursorPage.from_body(
            self._page_get(self._filter_params(agent_id, filters), cursor),
            model=session_models.Session,
        )

    async def alist_page(
        self, agent_id: str, *, cursor: Optional[str] = None, **filters: Any
    ) -> CursorPage:
        """Async sibling of ``list_page``."""
        body = await self._apage_get(self._filter_params(agent_id, filters), cursor)
        return CursorPage.from_body(body, model=session_models.Session)

    def count(self, agent_id: str, **filters: Any) -> int:
        """Cheap total for the filtered set via HEAD (``X-Total-Count``), no
        paging. Same filter kwargs as ``list``."""
        headers = self.http.head("sdk/v2/sessions", self._filter_params(agent_id, filters))
        return int(headers.get("X-Total-Count") or 0)

    async def acount(self, agent_id: str, **filters: Any) -> int:
        """Async sibling of ``count``."""
        headers = await self.http.ahead("sdk/v2/sessions", self._filter_params(agent_id, filters))
        return int(headers.get("X-Total-Count") or 0)

    def tags(self, agent_id: str, **filters: Any) -> List[str]:
        """The tag set for the filtered query via HEAD (``X-Tags``), no paging."""
        headers = self.http.head("sdk/v2/sessions", self._filter_params(agent_id, filters))
        return [t for t in headers.get("X-Tags", "").split(",") if t]

    async def atags(self, agent_id: str, **filters: Any) -> List[str]:
        """Async sibling of ``tags``."""
        headers = await self.http.ahead("sdk/v2/sessions", self._filter_params(agent_id, filters))
        return [t for t in headers.get("X-Tags", "").split(",") if t]

    # ---- list internals ----

    _FILTER_KEYS = frozenset({
        "experiment_id", "production", "cost", "duration", "num_events", "tags",
        "status", "eval_bool", "eval_number", "eval_string", "ordering", "page_size",
    })

    @classmethod
    def _filter_params(cls, agent_id: str, f: Dict[str, Any]) -> Dict[str, Any]:
        """Build the /sdk/v2/sessions query params from the filter kwargs,
        omitting unset ones. Shared by list / list_page / count / tags.

        An unknown filter name raises TypeError: the ``**filters`` methods
        (count/tags/list_page) would otherwise silently ignore a typo like
        ``experiment=`` and return the *unfiltered* result — a wrong number
        with no error."""
        unknown = set(f) - cls._FILTER_KEYS
        if unknown:
            raise TypeError(f"unknown session filter(s): {sorted(unknown)}")
        params: Dict[str, Any] = {"agent_id": agent_id}
        for key in ("experiment_id", "status", "cost", "duration", "num_events",
                    "ordering", "page_size", "eval_bool", "eval_number", "eval_string"):
            value = f.get(key)
            if value is not None:
                params[key] = value
        production = f.get("production")
        if production is not None:
            params["production"] = "true" if production else "false"
        tags = f.get("tags")
        if tags:
            params["tags"] = ",".join(tags) if isinstance(tags, (list, tuple)) else tags
        return params

    def _page_get(self, base: Dict[str, Any], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base)
        if cursor:
            params["cursor"] = cursor
        return self.http.get("sdk/v2/sessions", params)

    async def _apage_get(self, base: Dict[str, Any], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base)
        if cursor:
            params["cursor"] = cursor
        return await self.http.aget("sdk/v2/sessions", params)

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
