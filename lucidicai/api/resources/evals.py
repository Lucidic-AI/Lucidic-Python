"""Evals resource API operations."""
import logging
import math
import threading
import time
from typing import Any, Dict, Optional, Set, Union

from ..client import HttpClient

logger = logging.getLogger("Lucidic")

# default close() wait for in-flight emit() posts
EVAL_FLUSH_TIMEOUT_SECONDS = 5.0


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
        self._pending: Set[threading.Thread] = set()
        self._pending_lock = threading.Lock()

    def emit(
        self,
        result: Union[bool, int, float, str],
        name: Optional[str] = None,
        description: Optional[str] = None,
        result_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Fire-and-forget evaluation submission that returns instantly.

        This function returns immediately while the actual evaluation
        submission happens in a background thread. Perfect for non-blocking
        evaluation logging.

        Args:
            result: The evaluation result. Can be bool, int, float, or str.
            name: Optional name for the evaluation. If not provided, the backend
                will generate a default name based on the result type.
            description: Optional description of the evaluation.
            result_type: Optional explicit result type ("boolean", "number", "string").
                If not provided, it will be inferred from the result value.
            session_id: Optional session ID. If not provided, uses the current
                session from context.

        Example:
            # Basic usage - type inferred
            client.evals.emit(result=True)
            client.evals.emit(result=0.95)
            client.evals.emit(result="excellent")

            # With name and description
            client.evals.emit(
                result=True,
                name="task_completed",
                description="User task was successful"
            )

            # Explicit session_id
            client.evals.emit(result=0.87, name="accuracy_score", session_id="abc-123")
        """
        from ...sdk.context import current_session_id

        # Capture session from context if not provided
        captured_session_id = session_id
        if not captured_session_id:
            captured_session_id = current_session_id.get(None)

        if not captured_session_id:
            logger.debug("[EvalsResource] No active session for emit()")
            return

        # Infer or validate result_type
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
        except ValueError as e:
            if self._production:
                logger.error(f"[EvalsResource] {e}")
                return
            else:
                raise

        # Capture all data for background thread
        captured_result = result
        captured_name = name
        captured_description = description
        captured_type = inferred_type

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

                self._create_eval(params)
            except Exception as e:
                logger.debug(f"[EvalsResource] Background emit() failed: {e}")
            finally:
                with self._pending_lock:
                    self._pending.discard(threading.current_thread())

        # daemon so a hung post never blocks exit; flush() waits
        thread = threading.Thread(target=_background_emit, daemon=True)
        with self._pending_lock:
            self._pending.add(thread)
        try:
            thread.start()
        except Exception:
            # an unstarted thread can never be joined; don't leave it pending
            with self._pending_lock:
                self._pending.discard(thread)
            raise

    def flush(self, timeout: float = EVAL_FLUSH_TIMEOUT_SECONDS) -> bool:
        """Wait for in-flight emit() posts to finish.

        Args:
            timeout: Upper bound in seconds for the whole wait, shared across
                all pending posts.

        Returns:
            True if every post started before this call has finished (sent or
            failed), False if the timeout ran out first.
        """
        deadline = time.monotonic() + timeout
        with self._pending_lock:
            pending = list(self._pending)
        for thread in pending:
            thread.join(max(0.0, deadline - time.monotonic()))
        with self._pending_lock:
            unfinished = [t for t in pending if t in self._pending]
        if unfinished:
            logger.warning(
                f"[EvalsResource] flush() timed out after {timeout}s with "
                f"{len(unfinished)} eval(s) still in flight"
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
