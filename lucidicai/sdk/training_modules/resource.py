"""High-level Training Modules SDK namespace.

``client.training_modules`` is the inference-time SDK surface for
checkpoint-backed module tools. It hides backend inference-run bookkeeping
on success and returns structured soft failures for agent-facing failures.
"""
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from ...api.polling import await_for, wait_for
from ...api.resources.training_modules import TrainingModulesAPIResource
from ...core.errors import LucidicError, WaitTimeout
from ..context import current_session_id

if TYPE_CHECKING:
    from ...client import LucidicAI


logger = logging.getLogger("Lucidic")

ACTIVE_STATUSES = {"PENDING", "SUBMITTED", "RUNNING"}


class _PollAbort(Exception):
    """Carries a soft-failure result out of the poll closure (LUC-920 refactor).

    The training-module mock-call contract never raises — a poll-time problem
    (malformed response, transport error) degrades to a structured soft failure.
    The generic ``wait_for`` loop can't produce those, so the poll closure raises
    this to hand the already-built failure back for the caller to return."""

    def __init__(self, failure: Dict[str, Any]):
        self.failure = failure
        super().__init__(failure.get("error", {}).get("code", "poll_abort"))


class TrainingModulesResource:
    """User-facing ``client.training_modules`` namespace.

    Main entry points:

    - ``tools()`` / ``atools()`` list checkpoint module tools for a session.
    - ``openai_tools()`` and ``anthropic_tools()`` convert those schemas to
      provider-native tool specs.
    - ``call()`` / ``acall()`` submit a module inference and poll until a
      terminal result or SDK timeout.
    """

    def __init__(
        self,
        client: "LucidicAI",
        *,
        default_timeout_seconds: float = 300.0,
        default_poll_interval_seconds: float = 1.0,
    ):
        self._client = client
        self._api = TrainingModulesAPIResource(client._http)
        self.default_timeout_seconds = default_timeout_seconds
        self.default_poll_interval_seconds = default_poll_interval_seconds

    # ==================== Tool Discovery ====================

    def tools(
        self,
        session_id: Optional[str] = None,
        *,
        prompt_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List module tools, each annotated with its ``edit_site``.

        Resolves against an explicit ``checkpoint_id`` (introspection) or, by
        default, the active session's loaded checkpoint (so this works naturally
        inside ``with client.sessions.create(...)``). Pass ``prompt_name`` -- the
        prompt at the current LLM call -- to get just the tools that live at that
        edit site (LUC-801); server-side scoped, so nothing else leaks in.
        """
        body = self._api.list_tools(
            **self._tools_kwargs(session_id, prompt_name=prompt_name, checkpoint_id=checkpoint_id)
        )
        return list(body.get("tools") or [])

    async def atools(
        self,
        session_id: Optional[str] = None,
        *,
        prompt_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Async sibling of ``tools``."""
        body = await self._api.alist_tools(
            **self._tools_kwargs(session_id, prompt_name=prompt_name, checkpoint_id=checkpoint_id)
        )
        return list(body.get("tools") or [])

    def _tools_kwargs(
        self,
        session_id: Optional[str],
        *,
        prompt_name: Optional[str],
        checkpoint_id: Optional[str],
    ) -> Dict[str, Any]:
        # checkpoint_id introspects a checkpoint directly (no session needed);
        # otherwise resolve the active session so the call works inside a
        # `with client.sessions.create(...)` block.
        if checkpoint_id is not None:
            return {"checkpoint_id": checkpoint_id, "prompt_name": prompt_name}
        return {"session_id": self._resolve_session_id(session_id), "prompt_name": prompt_name}

    def openai_tools(
        self,
        session_id: Optional[str] = None,
        *,
        prompt_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Return checkpoint module tools as OpenAI function specs.

        Accepts the same ``prompt_name``/``checkpoint_id`` scoping as ``tools``.
        """
        module_tools = tools if tools is not None else self.tools(
            session_id, prompt_name=prompt_name, checkpoint_id=checkpoint_id
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["tool_name"],
                    "description": tool.get("tool_description", "") or "",
                    "parameters": tool.get("input_schema") or {},
                },
            }
            for tool in module_tools
            if tool.get("tool_name")
        ]

    async def aopenai_tools(
        self,
        session_id: Optional[str] = None,
        *,
        prompt_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Async sibling of ``openai_tools``."""
        module_tools = tools if tools is not None else await self.atools(
            session_id, prompt_name=prompt_name, checkpoint_id=checkpoint_id
        )
        return self.openai_tools(tools=module_tools)

    def anthropic_tools(
        self,
        session_id: Optional[str] = None,
        *,
        prompt_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Return checkpoint module tools as Anthropic tool specs.

        Accepts the same ``prompt_name``/``checkpoint_id`` scoping as ``tools``.
        """
        module_tools = tools if tools is not None else self.tools(
            session_id, prompt_name=prompt_name, checkpoint_id=checkpoint_id
        )
        return [
            {
                "name": tool["tool_name"],
                "description": tool.get("tool_description", "") or "",
                "input_schema": tool.get("input_schema") or {},
            }
            for tool in module_tools
            if tool.get("tool_name")
        ]

    async def aanthropic_tools(
        self,
        session_id: Optional[str] = None,
        *,
        prompt_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Async sibling of ``anthropic_tools``."""
        module_tools = tools if tools is not None else await self.atools(
            session_id, prompt_name=prompt_name, checkpoint_id=checkpoint_id
        )
        return self.anthropic_tools(tools=module_tools)

    # ==================== Inference ====================

    def call(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        session_id: Optional[str] = None,
        client_event_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        poll_interval_seconds: Optional[float] = None,
    ) -> Any:
        """Run a module tool and return the agent-facing result.

        Success returns only ``result["return_value"]`` so the agent sees a
        normal tool result. Failure and SDK timeout return a structured soft
        failure dict with enough metadata to debug the run.
        """
        resolved_session_id = self._resolve_session_id_for_call(session_id, tool_name)
        if isinstance(resolved_session_id, dict):
            return resolved_session_id

        normalized_args, arg_error = self._normalize_arguments(arguments, tool_name)
        if arg_error is not None:
            return arg_error

        event_id = client_event_id or str(uuid.uuid4())
        try:
            run = self._api.submit_inference(
                session_id=resolved_session_id,
                tool_name=tool_name,
                arguments=normalized_args,
                client_event_id=event_id,
            )
        except Exception as exc:
            return self._soft_failure(
                code="training_module_submission_failed",
                message=str(exc),
                session_id=resolved_session_id,
                tool_name=tool_name,
            )

        return self._poll_to_tool_result(
            run,
            session_id=resolved_session_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    async def acall(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        session_id: Optional[str] = None,
        client_event_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        poll_interval_seconds: Optional[float] = None,
    ) -> Any:
        """Async sibling of ``call``."""
        resolved_session_id = self._resolve_session_id_for_call(session_id, tool_name)
        if isinstance(resolved_session_id, dict):
            return resolved_session_id

        normalized_args, arg_error = self._normalize_arguments(arguments, tool_name)
        if arg_error is not None:
            return arg_error

        event_id = client_event_id or str(uuid.uuid4())
        try:
            run = await self._api.asubmit_inference(
                session_id=resolved_session_id,
                tool_name=tool_name,
                arguments=normalized_args,
                client_event_id=event_id,
            )
        except Exception as exc:
            return self._soft_failure(
                code="training_module_submission_failed",
                message=str(exc),
                session_id=resolved_session_id,
                tool_name=tool_name,
            )

        return await self._apoll_to_tool_result(
            run,
            session_id=resolved_session_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def dispatch_openai_tool_call(
        self,
        call: Any,
        *,
        session_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        poll_interval_seconds: Optional[float] = None,
    ) -> Any:
        """Dispatch one OpenAI tool call through Training Modules."""
        tool_name, arguments, client_event_id = _extract_openai_call(call)
        return self.call(
            tool_name,
            arguments,
            session_id=session_id,
            client_event_id=client_event_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    async def adispatch_openai_tool_call(
        self,
        call: Any,
        *,
        session_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        poll_interval_seconds: Optional[float] = None,
    ) -> Any:
        """Async sibling of ``dispatch_openai_tool_call``."""
        tool_name, arguments, client_event_id = _extract_openai_call(call)
        return await self.acall(
            tool_name,
            arguments,
            session_id=session_id,
            client_event_id=client_event_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def dispatch_anthropic_tool_call(
        self,
        block: Any,
        *,
        session_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        poll_interval_seconds: Optional[float] = None,
    ) -> Any:
        """Dispatch one Anthropic tool_use block through Training Modules."""
        tool_name, arguments, client_event_id = _extract_anthropic_block(block)
        return self.call(
            tool_name,
            arguments,
            session_id=session_id,
            client_event_id=client_event_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    async def adispatch_anthropic_tool_call(
        self,
        block: Any,
        *,
        session_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        poll_interval_seconds: Optional[float] = None,
    ) -> Any:
        """Async sibling of ``dispatch_anthropic_tool_call``."""
        tool_name, arguments, client_event_id = _extract_anthropic_block(block)
        return await self.acall(
            tool_name,
            arguments,
            session_id=session_id,
            client_event_id=client_event_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    # ==================== Internals ====================

    def _poll_to_tool_result(
        self,
        run: Dict[str, Any],
        *,
        session_id: str,
        timeout_seconds: Optional[float],
        poll_interval_seconds: Optional[float],
    ) -> Any:
        box = {"run": run}

        def poll() -> Dict[str, Any]:
            current = box["run"]
            inference_run_id = current.get("inference_run_id")
            if not inference_run_id:
                raise _PollAbort(self._soft_failure(
                    code="training_module_malformed_response",
                    message="Training Module inference response did not include inference_run_id.",
                    run=current, session_id=session_id, tool_name=current.get("tool_name"),
                ))
            try:
                box["run"] = self._api.get_inference(
                    inference_run_id=str(inference_run_id), session_id=session_id)
            except Exception as exc:
                raise _PollAbort(self._soft_failure(
                    code="training_module_poll_failed", message=str(exc),
                    run=current, session_id=session_id, tool_name=current.get("tool_name"),
                ))
            return box["run"]

        try:
            final = wait_for(
                poll,
                is_terminal=lambda r: r.get("status") not in ACTIVE_STATUSES,
                timeout=self._timeout(timeout_seconds),
                interval=self._poll_interval(poll_interval_seconds),
                initial=run,
            )
        except WaitTimeout as timed_out:
            return self._timeout_failure(timed_out.last_state, timeout_seconds)
        except _PollAbort as aborted:
            return aborted.failure
        return self._terminal_tool_result(final)

    async def _apoll_to_tool_result(
        self,
        run: Dict[str, Any],
        *,
        session_id: str,
        timeout_seconds: Optional[float],
        poll_interval_seconds: Optional[float],
    ) -> Any:
        box = {"run": run}

        async def poll() -> Dict[str, Any]:
            current = box["run"]
            inference_run_id = current.get("inference_run_id")
            if not inference_run_id:
                raise _PollAbort(self._soft_failure(
                    code="training_module_malformed_response",
                    message="Training Module inference response did not include inference_run_id.",
                    run=current, session_id=session_id, tool_name=current.get("tool_name"),
                ))
            try:
                box["run"] = await self._api.aget_inference(
                    inference_run_id=str(inference_run_id), session_id=session_id)
            except Exception as exc:
                raise _PollAbort(self._soft_failure(
                    code="training_module_poll_failed", message=str(exc),
                    run=current, session_id=session_id, tool_name=current.get("tool_name"),
                ))
            return box["run"]

        try:
            final = await await_for(
                poll,
                is_terminal=lambda r: r.get("status") not in ACTIVE_STATUSES,
                timeout=self._timeout(timeout_seconds),
                interval=self._poll_interval(poll_interval_seconds),
                initial=run,
            )
        except WaitTimeout as timed_out:
            return self._timeout_failure(timed_out.last_state, timeout_seconds)
        except _PollAbort as aborted:
            return aborted.failure
        return self._terminal_tool_result(final)

    def _terminal_tool_result(self, run: Dict[str, Any]) -> Any:
        status = run.get("status")
        if status == "SUCCEEDED":
            result = run.get("result") or {}
            if isinstance(result, dict) and "return_value" in result:
                return result.get("return_value")
            return result

        if status == "FAILED":
            error = run.get("error") or {}
            message = _error_message(error) or "Training Module inference failed."
            code = _error_code(error) or "training_module_inference_failed"
        elif status == "TIMED_OUT":
            message = "Training Module inference timed out on the backend."
            code = "training_module_inference_timed_out"
        elif status == "CANCELLED":
            message = "Training Module inference was cancelled."
            code = "training_module_inference_cancelled"
        else:
            message = f"Training Module inference ended with unexpected status {status!r}."
            code = "training_module_inference_unknown_status"

        return self._soft_failure(
            code=code,
            message=message,
            run=run,
            session_id=run.get("session_id"),
            tool_name=run.get("tool_name"),
        )

    def _timeout_failure(
        self,
        run: Dict[str, Any],
        timeout_seconds: Optional[float],
    ) -> Dict[str, Any]:
        timeout = self._timeout(timeout_seconds)
        return self._soft_failure(
            code="training_module_sdk_timeout",
            message=f"Training Module inference did not finish within {timeout:g} seconds.",
            run=run,
            session_id=run.get("session_id"),
            tool_name=run.get("tool_name"),
            extra={"timeout_seconds": timeout},
        )

    def _soft_failure(
        self,
        *,
        code: str,
        message: str,
        run: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "session_id": session_id,
            "tool_name": tool_name,
        }
        if run:
            metadata.update({
                "inference_run_id": run.get("inference_run_id"),
                "checkpoint_id": run.get("checkpoint_id"),
                "module_key": run.get("module_key"),
                "status": run.get("status"),
                "temporal_workflow_id": run.get("temporal_workflow_id"),
                "temporal_run_id": run.get("temporal_run_id"),
                "created_at": run.get("created_at"),
                "updated_at": run.get("updated_at"),
                "started_at": run.get("started_at"),
                "completed_at": run.get("completed_at"),
            })
        if extra:
            metadata.update(extra)
        return {
            "error": {
                "code": code,
                "message": message,
            },
            "metadata": {k: v for k, v in metadata.items() if v not in (None, "")},
        }

    def _resolve_session_id(self, session_id: Optional[str]) -> str:
        resolved = session_id or current_session_id.get(None)
        if not resolved:
            raise LucidicError(
                "Training Modules require a session_id or an active Lucidic session"
            )
        return str(resolved)

    def _resolve_session_id_for_call(
        self,
        session_id: Optional[str],
        tool_name: str,
    ) -> Union[str, Dict[str, Any]]:
        try:
            return self._resolve_session_id(session_id)
        except LucidicError as exc:
            return self._soft_failure(
                code="training_module_missing_session",
                message=str(exc),
                tool_name=tool_name,
            )

    def _normalize_arguments(
        self,
        arguments: Optional[Dict[str, Any]],
        tool_name: str,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        if arguments is None:
            return {}, None
        if not isinstance(arguments, dict):
            return {}, self._soft_failure(
                code="training_module_invalid_arguments",
                message="Training Module arguments must be a dict.",
                tool_name=tool_name,
            )
        return dict(arguments), None

    def _timeout(self, timeout_seconds: Optional[float]) -> float:
        timeout = self.default_timeout_seconds if timeout_seconds is None else timeout_seconds
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = self.default_timeout_seconds
        return max(0.0, timeout)

    def _poll_interval(self, poll_interval_seconds: Optional[float]) -> float:
        interval = (
            self.default_poll_interval_seconds
            if poll_interval_seconds is None
            else poll_interval_seconds
        )
        try:
            interval = float(interval)
        except (TypeError, ValueError):
            interval = self.default_poll_interval_seconds
        return max(0.05, interval)


def _extract_openai_call(call: Any) -> Tuple[str, Dict[str, Any], str]:
    if hasattr(call, "function"):
        fn = call.function
        name = fn.name if hasattr(fn, "name") else fn["name"]
        raw_args = fn.arguments if hasattr(fn, "arguments") else fn["arguments"]
        call_id = getattr(call, "id", None)
    elif isinstance(call, dict):
        fn = call["function"]
        name = fn["name"]
        raw_args = fn.get("arguments")
        call_id = call.get("id")
    else:
        raise TypeError(
            "dispatch_openai_tool_call expected an OpenAI tool call object or dict"
        )
    return str(name), _parse_json_arguments(raw_args, str(name)), str(call_id or uuid.uuid4())


def _extract_anthropic_block(block: Any) -> Tuple[str, Dict[str, Any], str]:
    if isinstance(block, dict):
        name = block.get("name")
        raw_input = block.get("input")
        block_id = block.get("id")
    else:
        name = getattr(block, "name", None)
        raw_input = getattr(block, "input", None)
        block_id = getattr(block, "id", None)

    if not name or not isinstance(name, str):
        raise TypeError("dispatch_anthropic_tool_call expected a block with a name")
    if raw_input is None:
        arguments = {}
    elif isinstance(raw_input, dict):
        arguments = dict(raw_input)
    elif isinstance(raw_input, str):
        arguments = _parse_json_arguments(raw_input, name)
    else:
        logger.warning(
            "[TrainingModulesResource] Anthropic tool_use %r input is %s; passing empty args",
            name, type(raw_input).__name__,
        )
        arguments = {}
    return name, arguments, str(block_id or uuid.uuid4())


def _parse_json_arguments(raw_args: Any, tool_name: str) -> Dict[str, Any]:
    if not raw_args:
        return {}
    try:
        parsed = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "[TrainingModulesResource] tool %r had unparseable JSON arguments; passing empty args",
            tool_name,
        )
        return {}
    if isinstance(parsed, dict):
        return parsed
    logger.warning(
        "[TrainingModulesResource] tool %r arguments parsed to %s; passing empty args",
        tool_name, type(parsed).__name__,
    )
    return {}


def _error_code(error: Any) -> Optional[str]:
    if isinstance(error, dict):
        value = error.get("code")
        return str(value) if value else None
    return None


def _error_message(error: Any) -> Optional[str]:
    if isinstance(error, dict):
        for key in ("message", "detail", "error"):
            value = error.get(key)
            if value:
                return str(value)
    if error:
        return str(error)
    return None
