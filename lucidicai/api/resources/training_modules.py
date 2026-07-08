"""Training Modules SDK HTTP resource.

Thin wrappers over the backend inference-time Training Modules endpoints:

- GET /sdk/training-modules/tools
- POST /sdk/training-modules/inferences
- GET /sdk/training-modules/inferences/{inference_run_id}
"""
import logging
from typing import Any, Dict, Optional

import httpx

from ..client import HttpClient
from ...core.errors import LucidicError

logger = logging.getLogger("Lucidic")


class TrainingModulesAPIResource:
    """HTTP-only Training Modules resource.

    High-level polling and tool-spec shaping live in
    ``lucidicai.sdk.training_modules.resource.TrainingModulesResource``.
    This class only knows the backend endpoint shapes.
    """

    def __init__(self, http: HttpClient):
        self.http = http

    def list_tools(
        self,
        *,
        session_id: Optional[str] = None,
        prompt_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return module tools for a session's checkpoint or an explicit checkpoint.

        Each tool carries its ``edit_site`` (the prompt it's exposed at). Pass
        ``prompt_name`` to scope to a single edit site server-side (LUC-801), and
        ``checkpoint_id`` to introspect a checkpoint without a live session.
        """
        try:
            return self.http.get(
                "sdk/training-modules/tools",
                _tools_params(session_id=session_id, prompt_name=prompt_name, checkpoint_id=checkpoint_id),
            )
        except httpx.HTTPStatusError as exc:
            raise LucidicError(_format_training_module_error(exc)) from exc

    async def alist_tools(
        self,
        *,
        session_id: Optional[str] = None,
        prompt_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async sibling of ``list_tools``."""
        try:
            return await self.http.aget(
                "sdk/training-modules/tools",
                _tools_params(session_id=session_id, prompt_name=prompt_name, checkpoint_id=checkpoint_id),
            )
        except httpx.HTTPStatusError as exc:
            raise LucidicError(_format_training_module_error(exc)) from exc

    def submit_inference(
        self,
        *,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        client_event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or replay a Training Module inference run."""
        body: Dict[str, Any] = {
            "session_id": session_id,
            "tool_name": tool_name,
            "arguments": arguments,
        }
        if client_event_id is not None:
            body["client_event_id"] = client_event_id
        try:
            return self.http.post("sdk/training-modules/inferences", body)
        except httpx.HTTPStatusError as exc:
            raise LucidicError(_format_training_module_error(exc)) from exc

    async def asubmit_inference(
        self,
        *,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        client_event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async sibling of ``submit_inference``."""
        body: Dict[str, Any] = {
            "session_id": session_id,
            "tool_name": tool_name,
            "arguments": arguments,
        }
        if client_event_id is not None:
            body["client_event_id"] = client_event_id
        try:
            return await self.http.apost("sdk/training-modules/inferences", body)
        except httpx.HTTPStatusError as exc:
            raise LucidicError(_format_training_module_error(exc)) from exc

    def get_inference(
        self,
        *,
        inference_run_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch an existing Training Module inference run."""
        params = {"session_id": session_id} if session_id else None
        try:
            return self.http.get(
                f"sdk/training-modules/inferences/{inference_run_id}",
                params,
            )
        except httpx.HTTPStatusError as exc:
            raise LucidicError(_format_training_module_error(exc)) from exc

    async def aget_inference(
        self,
        *,
        inference_run_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async sibling of ``get_inference``."""
        params = {"session_id": session_id} if session_id else None
        try:
            return await self.http.aget(
                f"sdk/training-modules/inferences/{inference_run_id}",
                params,
            )
        except httpx.HTTPStatusError as exc:
            raise LucidicError(_format_training_module_error(exc)) from exc


def _tools_params(
    *,
    session_id: Optional[str],
    prompt_name: Optional[str],
    checkpoint_id: Optional[str],
) -> Dict[str, Any]:
    """Build the /sdk/training-modules/tools query params, omitting unset ones.

    The backend resolves the checkpoint from ``checkpoint_id`` (preferred) or
    ``session_id``, and ``prompt_name`` scopes to one edit site.
    """
    params: Dict[str, Any] = {}
    if checkpoint_id is not None:
        params["checkpoint_id"] = checkpoint_id
    if session_id is not None:
        params["session_id"] = session_id
    if prompt_name is not None:
        params["prompt_name"] = prompt_name
    return params


def _format_training_module_error(exc: httpx.HTTPStatusError) -> str:
    """Best-effort human-readable Training Modules error."""
    try:
        body = exc.response.json()
    except ValueError:
        return f"HTTP {exc.response.status_code}: {exc.response.text or 'no body'}"

    if isinstance(body, dict):
        if "error" in body:
            return f"HTTP {exc.response.status_code}: {body['error']}"
        if "errors" in body:
            return f"HTTP {exc.response.status_code} validation: {body['errors']}"
    return f"HTTP {exc.response.status_code}: {body!r}"
