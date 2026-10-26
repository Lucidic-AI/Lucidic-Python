"""Evals resource API operations."""
import asyncio
import logging
import math
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Union

from ..client import HttpClient

logger = logging.getLogger("Lucidic")

# default close() / sessions.end() wait for in-flight emit() posts
EVAL_FLUSH_TIMEOUT_SECONDS = 5.0
# how often aflush() looks at the pending posts
_AFLUSH_POLL_SECONDS = 0.02


def validate_flush_timeout(value: Any) -> float:
    """Return ``value`` as seconds, or raise ValueError if it is not a finite number >= 0."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(
            f"eval flush timeout must be a finite number of seconds >= 0, got {value!r}"
        )
    return float(value)


def _truncate_id(id_str: Optional[str]) -> str:
    """Truncate ID for logging."""
    if not id_str:
        return "None"
    return f"{id_str[:8]}..." if len(id_str) > 8 else id_str


def _infer_result_type(result: Any) -> str:
    """Infer result type from Python value.

    Note: bool must be checked first because bool is a subclass of int in Python.

    Args:
        result: The evaluation result value.

    Returns:
        The result type string: "boolean", "number", or "string".

    Raises:
        ValueError: If result is not a supported type.
    """
    if isinstance(result, bool):
        return "boolean"
    elif isinstance(result, (int, float)):
        return "number"
    elif isinstance(result, str):
        return "string"
    else:
        raise ValueError(
            f"Unsupported result type: {type(result).__name__}. "
            "Must be bool, int, float, or str."
        )


def _validate_result_type(result: Any, result_type: str) -> bool:
    """Validate that result matches the specified result_type.

    Args:
        result: The evaluation result value.
        result_type: The expected type ("boolean", "number", "string").

    Returns:
        True if the result matches the type, False otherwise.
    """
    if result_type == "boolean":
        return isinstance(result, bool)
    elif result_type == "number":
        # Check for bool first since bool is subclass of int
        return isinstance(result, (int, float)) and not isinstance(result, bool)
    elif result_type == "string":
        return isinstance(result, str)
    return False


def _is_blank_name(name: Any) -> bool:
    """No name, as the API counts it: None, or a string that is empty or only whitespace."""
    return name is None or (isinstance(name, str) and not name.strip())


def _validate_pass_threshold(pass_threshold: Any, result_type: str) -> None:
    """The API's rule for ``pass_threshold`` on POST /sdk/evals: a pass mark only
    with a number result, and only as a finite number.

    Raises:
        ValueError: If the pass mark is not allowed or not a finite number.
    """
    if result_type != "number":
        raise ValueError(
            f"pass_threshold only applies to a number result, not '{result_type}'"
        )
    finite = False
    if not isinstance(pass_threshold, bool) and isinstance(pass_threshold, (int, float)):
        try:
            finite = math.isfinite(pass_threshold)
        except OverflowError:  # an int too large for a float, which the API refuses too
            pass
    if not finite:
        raise ValueError(
            f"pass_threshold must be a finite number, got {pass_threshold!r}"
        )


class EvalsResource:
    """Handle evaluation-related API operations."""

    def __init__(self, http: HttpClient, production: bool = False):
        """Initialize evals resource.

        Args:
            http: HTTP client instance
            production: Whether to suppress errors in production mode
        """
        self.http = http
        self._production = production
        # each in-flight post's thread -> the session it posts for
        self._pending: Dict[threading.Thread, str] = {}
        self._pending_lock = threading.Lock()

    def emit(
        self,
        result: Union[bool, int, float, str],
        name: Optional[str] = None,
        description: Optional[str] = None,
        result_type: Optional[str] = None,
        session_id: Optional[str] = None,
        pass_threshold: Optional[float] = None,
    ) -> None:
        """Fire-and-forget evaluation submission that returns instantly.

        This function returns immediately while the actual evaluation
        submission happens in a background thread. Perfect for non-blocking
        evaluation logging.

        Args:
            result: The evaluation result. Can be bool, int, float, or str.
            name: Name for the evaluation. Required for a boolean or number
                result: None, an empty string or whitespace only counts as no
                name. A string result may omit it, and the backend then files it
                under a default name.
            description: Optional description of the evaluation.
            result_type: Optional explicit result type ("boolean", "number", "string").
                If not provided, it will be inferred from the result value.
            session_id: Optional session ID. If not provided, uses the current
                session from context.
            pass_threshold: Optional pass mark for a number result: the result
                passes at or above it. Only allowed with a number result and must
                be a finite number. It is stored on the agent's evaluator for this
                name, and the latest value sent wins.

        Raises:
            ValueError: If a boolean or number result has no name, the result
                does not match ``result_type``, or ``pass_threshold`` is not
                allowed, whether or not a session is active. In production mode
                it is logged and nothing is sent.

        Example:
            # Basic usage - type inferred
            client.evals.emit(result=True, name="task_passed")
            client.evals.emit(result=0.95, name="accuracy")
            client.evals.emit(result="excellent")

            # With name and description
            client.evals.emit(
                result=True,
                name="task_completed",
                description="User task was successful"
            )

            # Explicit session_id
            client.evals.emit(result=0.87, name="accuracy_score", session_id="abc-123")

            # A number with a pass mark: passes at 0.8 or above
            client.evals.emit(0.85, name="criterion_pass_rate", pass_threshold=0.8)
        """
        from ...sdk.context import current_session_id

        # Infer or validate result_type, before the session lookup, so a bad call
        # fails the same way inside and outside a session
        try:
            if result_type is None:
                inferred_type = _infer_result_type(result)
            else:
                # Validate that result matches the explicit type
                if not _validate_result_type(result, result_type):
                    error_msg = (
                        f"Result type mismatch: result is {type(result).__name__} "
                        f"but result_type is '{result_type}'"
                    )
                    if self._production:
                        logger.error(f"[EvalsResource] {error_msg}")
                        return
                    else:
                        raise ValueError(error_msg)
                inferred_type = result_type
            if _is_blank_name(name) and inferred_type in ("boolean", "number"):
                raise ValueError("name is required for boolean and number results")
            if pass_threshold is not None:
                _validate_pass_threshold(pass_threshold, inferred_type)
        except ValueError as e:
            if self._production:
                logger.error(f"[EvalsResource] {e}")
                return
            else:
                raise

        # Capture session from context if not provided
        captured_session_id = session_id
        if not captured_session_id:
            captured_session_id = current_session_id.get(None)

        if not captured_session_id:
            logger.debug("[EvalsResource] No active session for emit()")
            return

        # Capture all data for background thread
        captured_result = result
        captured_name = name
        captured_description = description
        captured_type = inferred_type
        captured_pass_threshold = pass_threshold

        def _background_emit():
            try:
                params: Dict[str, Any] = {
                    "session_id": captured_session_id,
                    "result": captured_result,
                    "result_type": captured_type,
                }
                if captured_name is not None:
                    params["name"] = captured_name
                if captured_description is not None:
                    params["description"] = captured_description
                if captured_pass_threshold is not None:
                    params["pass_threshold"] = captured_pass_threshold

                self._create_eval(params)
            except Exception as e:
                logger.debug(f"[EvalsResource] Background emit() failed: {e}")
            finally:
                with self._pending_lock:
                    self._pending.pop(threading.current_thread(), None)

        # daemon so a hung post never blocks exit; flush() waits
        thread = threading.Thread(target=_background_emit, daemon=True)
        # register and start under one lock, so a flush never sees an unstarted
        # thread (the thread only takes the lock when it finishes)
        with self._pending_lock:
            self._pending[thread] = captured_session_id
            try:
                thread.start()
            except Exception:
                # an unstarted thread can never be joined; don't leave it pending
                self._pending.pop(thread, None)
                raise

    def flush(
        self,
        timeout: float = EVAL_FLUSH_TIMEOUT_SECONDS,
        session_id: Union[str, Iterable[str], None] = None,
    ) -> bool:
        """Wait for in-flight emit() posts to finish.

        Args:
            timeout: Upper bound in seconds for the whole wait, shared across
                all pending posts.
            session_id: Wait only for the posts for this session, or for these
                sessions (an iterable of ids). If None, wait for every
                session's posts.

        Returns:
            True if every post it waits for (those started before this call,
            for the given sessions when ``session_id`` is set) has finished,
            sent or failed. False if the timeout ran out first.
        """
        deadline = time.monotonic() + timeout
        pending = self._pending_for(session_id)
        for thread in pending:
            thread.join(max(0.0, deadline - time.monotonic()))
        return self._report_unfinished("flush", timeout, session_id, pending)

    async def aflush(
        self,
        timeout: float = EVAL_FLUSH_TIMEOUT_SECONDS,
        session_id: Union[str, Iterable[str], None] = None,
    ) -> bool:
        """Async sibling of ``flush``.

        It waits on the event loop, with no executor thread, so the wait ends
        on time however busy the loop's executor is.
        """
        deadline = time.monotonic() + timeout
        pending = self._pending_for(session_id)
        while self._still_pending(pending):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(_AFLUSH_POLL_SECONDS, remaining))
        return self._report_unfinished("aflush", timeout, session_id, pending)

    def _pending_for(self, session_id: Union[str, Iterable[str], None]) -> List[threading.Thread]:
        """The in-flight posts for these sessions (all of them for None)."""
        if session_id is None:
            wanted = None
        elif isinstance(session_id, str):
            wanted = {session_id}
        else:
            wanted = set(session_id)
        with self._pending_lock:
            return [
                thread for thread, sid in self._pending.items()
                if wanted is None or sid in wanted
            ]

    def _still_pending(self, threads: List[threading.Thread]) -> List[threading.Thread]:
        with self._pending_lock:
            return [t for t in threads if t in self._pending]

    def _report_unfinished(
        self,
        method: str,
        timeout: float,
        session_id: Union[str, Iterable[str], None],
        pending: List[threading.Thread],
    ) -> bool:
        """Warn about posts still in flight after a wait; True if there are none."""
        unfinished = self._still_pending(pending)
        if unfinished:
            if session_id is None:
                for_session = ""
            elif isinstance(session_id, str):
                for_session = f" for session {_truncate_id(session_id)}"
            else:
                for_session = " for the given sessions"
            logger.warning(
                f"[EvalsResource] {method}() timed out after {timeout}s with "
                f"{len(unfinished)} eval(s) still in flight{for_session}"
            )
        return not unfinished

    def _create_eval(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send evaluation to backend API.

        Args:
            params: Evaluation parameters including:
                - session_id: Session ID
                - result: Evaluation result value
                - result_type: Type of result ("boolean", "number", "string")
                - name: Optional evaluation name
                - description: Optional description
                - pass_threshold: Optional pass mark (number results only)

        Returns:
            API response (typically empty for 201 Created)
        """
        session_id = params.get("session_id")
        name = params.get("name")
        result_type = params.get("result_type")
        logger.debug(
            f"[Evals] _create_eval() called - "
            f"session_id={_truncate_id(session_id)}, name={name!r}, "
            f"result_type={result_type!r}"
        )

        response = self.http.post("sdk/evals", params)

        logger.debug(
            f"[Evals] _create_eval() response - "
            f"session_id={_truncate_id(session_id)}, "
            f"response_keys={list(response.keys()) if response else 'None'}"
        )
        return response
