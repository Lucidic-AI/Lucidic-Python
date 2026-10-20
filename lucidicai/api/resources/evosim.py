"""EvoSim resource API operations (LUC-923 run management).

``client.evosims`` manages EvoSim runs: kick one off (``train``), list / read runs,
cancel a run, inspect an iteration's Training Module Instances, and block until a
run is terminal (``wait_for``). These hit the gen-3 ``/sdk/...`` paths (there is no
``/sdk/v2/evosim*`` surface).

``train`` / ``atrain`` swallow in production (a kickoff convenience); the
run-management reads/writes are data-bearing and do NOT swallow. Each has an
``a``-prefixed async sibling.
"""
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from ..client import HttpClient
from ..models.base import CursorPage
from ..models.evosim import EvoSim, TrainingModuleInstance
from ..pagination import apaginate, paginate
from ..polling import await_for, wait_for
from ...core.errors import LucidicError, require_agent_id

logger = logging.getLogger("Lucidic")

_EVOSIMS = "sdk/evosims"


class EvoSimsResource:
    """Handle EvoSim run-management API operations."""

    def __init__(
        self,
        http: HttpClient,
        agent_id: Optional[str] = None,
        production: bool = False,
    ):
        """Initialize EvoSim resource.

        Args:
            http: HTTP client instance
            agent_id: Default agent ID for training runs / listing
            production: Whether to suppress errors in production mode (train only)
        """
        self.http = http
        self._agent_id = agent_id
        self._production = production

    # ==================== kickoff (gen-3, swallows) ====================

    def train(self, config_json_file: Union[str, Path]) -> Dict[str, Any]:
        """Start an EvoSim training run from a JSON config file.

        The file must contain a JSON object; its top-level keys become the
        request payload for the training run. ``agent_id`` is filled in from
        the client config when the file doesn't provide one.

        Args:
            config_json_file: Path to the JSON training config

        Returns:
            Backend response describing the run (successful or failed)
        """
        # LUC-926: a training run needs an agent; raise before the swallow.
        require_agent_id(self._agent_id, "evosim.train")
        try:
            payload = _load_training_config(config_json_file)
            payload.setdefault("agent_id", self._agent_id)
            return self.http.post("sdk/evosim/training-run", payload)
        except Exception as e:
            if self._production:
                logger.error(f"[EvoSimsResource] Failed to start training run: {e}")
                return {"status": "failed", "error": str(e)}
            raise

    async def atrain(self, config_json_file: Union[str, Path]) -> Dict[str, Any]:
        """Async sibling of ``train``."""
        require_agent_id(self._agent_id, "evosim.train")
        try:
            payload = _load_training_config(config_json_file)
            payload.setdefault("agent_id", self._agent_id)
            return await self.http.apost("sdk/evosim/training-run", payload)
        except Exception as e:
            if self._production:
                logger.error(f"[EvoSimsResource] Failed to start training run: {e}")
                return {"status": "failed", "error": str(e)}
            raise

    # ==================== run management (LUC-923, no swallow) ====================

    def list(self, agent_id: Optional[str] = None, *, page_size: Optional[int] = None) -> Iterator[EvoSim]:
        """Lazily iterate an agent's EvoSim runs, newest first (GET /sdk/evosims;
        needs ``evosim:read``). ``agent_id`` defaults to the configured agent. List
        items are the light shape (no rolled-up ``status`` — call ``get`` for that)."""
        base = self._list_params(agent_id, page_size)
        return paginate(lambda c: self._page_get(base, c), model=EvoSim)

    def alist(self, agent_id: Optional[str] = None, *, page_size: Optional[int] = None) -> AsyncIterator[EvoSim]:
        """Async sibling of ``list``."""
        base = self._list_params(agent_id, page_size)
        return apaginate(lambda c: self._apage_get(base, c), model=EvoSim)

    def list_page(
        self, agent_id: Optional[str] = None, *, cursor: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of an agent's EvoSim runs."""
        return CursorPage.from_body(self._page_get(self._list_params(agent_id, page_size), cursor), model=EvoSim)

    async def alist_page(
        self, agent_id: Optional[str] = None, *, cursor: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``list_page``."""
        body = await self._apage_get(self._list_params(agent_id, page_size), cursor)
        return CursorPage.from_body(body, model=EvoSim)

    def get(self, evosim_id: str) -> EvoSim:
        """Read one EvoSim run with its rolled-up ``status`` / ``failure_reason``,
        derived ``checkpoint_id``, and ``iterations`` (GET /sdk/evosims/{id}; needs
        ``evosim:read``). Unknown / cross-org / out-of-binding id → ``NotFoundError``."""
        return self._detail(self.http.get(f"{_EVOSIMS}/{evosim_id}"))

    async def aget(self, evosim_id: str) -> EvoSim:
        """Async sibling of ``get``."""
        return self._detail(await self.http.aget(f"{_EVOSIMS}/{evosim_id}"))

    def cancel(self, evosim_id: str) -> Optional[str]:
        """Cancel an EvoSim run (POST /sdk/evosims/{id}/cancel; needs
        ``evosim:delete``). A kill switch — terminates the Temporal workflow and
        projects ``CANCELED`` onto the run. Idempotent: cancelling an already-terminal
        run leaves it untouched. Returns the effective status (e.g. ``"CANCELED"``, or
        the run's terminal status if it finished first). Raises
        ``ServiceUnavailableError`` (503) if Temporal is unavailable — retry."""
        resp = self.http.post(f"{_EVOSIMS}/{evosim_id}/cancel")
        return resp.get("status")

    async def acancel(self, evosim_id: str) -> Optional[str]:
        """Async sibling of ``cancel``."""
        resp = await self.http.apost(f"{_EVOSIMS}/{evosim_id}/cancel")
        return resp.get("status")

    def iteration_instances(self, iteration_id: str) -> List[TrainingModuleInstance]:
        """List the Training Module Instances (TMIs) of an EvoSim iteration (GET
        /sdk/evosim-iterations/{id}/instances; needs ``evosim:read``). Iteration ids
        come from ``get(evosim_id).iterations``. Not paginated."""
        resp = self.http.get(f"sdk/evosim-iterations/{iteration_id}/instances")
        return TrainingModuleInstance.from_list(resp.get("training_module_instances", []))

    async def aiteration_instances(self, iteration_id: str) -> List[TrainingModuleInstance]:
        """Async sibling of ``iteration_instances``."""
        resp = await self.http.aget(f"sdk/evosim-iterations/{iteration_id}/instances")
        return TrainingModuleInstance.from_list(resp.get("training_module_instances", []))

    def wait_for(self, evosim_id: str, *, timeout: float = 3600.0, interval: float = 10.0) -> EvoSim:
        """Block until an EvoSim run reaches a terminal status (``SUCCEEDED`` /
        ``PARTIAL_SUCCESS`` / ``FAILED`` / ``CANCELED``) or ``timeout`` elapses,
        polling ``get`` every ``interval`` seconds. Returns the terminal run — inspect
        ``.status`` / ``.failure_reason`` / ``.checkpoint_id``. Raises ``WaitTimeout``
        if the deadline passes first.

        EvoSim runs are long (up to ``max_iterations`` ×
        ``hard_stop_seconds_per_iteration`` — the defaults allow multiple hours), so
        the default ``timeout`` (1 h) may be too short for a big run; pass a larger
        value, or poll ``get`` yourself."""
        return wait_for(
            lambda: self.get(evosim_id),
            is_terminal=lambda run: run.is_terminal, timeout=timeout, interval=interval)

    async def await_for(self, evosim_id: str, *, timeout: float = 3600.0, interval: float = 10.0) -> EvoSim:
        """Async sibling of ``wait_for``."""
        return await await_for(
            lambda: self.aget(evosim_id),
            is_terminal=lambda run: run.is_terminal, timeout=timeout, interval=interval)

    # ==================== internals ====================

    def _list_params(self, agent_id: Optional[str], page_size: Optional[int]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "agent_id": require_agent_id(agent_id or self._agent_id, "evosims.list"),
        }
        if page_size is not None:
            params["page_size"] = page_size
        return params

    def _page_get(self, base: Dict[str, Any], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base)
        if cursor:
            params["cursor"] = cursor
        return self.http.get(_EVOSIMS, params)

    async def _apage_get(self, base: Dict[str, Any], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base)
        if cursor:
            params["cursor"] = cursor
        return await self.http.aget(_EVOSIMS, params)

    @staticmethod
    def _detail(resp: Dict[str, Any]) -> EvoSim:
        """Flatten the detail envelope ``{evosim, iterations, status, failure_reason,
        checkpoint_id}`` into a single typed ``EvoSim`` (the run id lives nested under
        ``evosim``, the rolled-up status at the top level)."""
        data = dict(resp.get("evosim") or {})
        data["status"] = resp.get("status")
        data["failure_reason"] = resp.get("failure_reason")
        data["checkpoint_id"] = resp.get("checkpoint_id")
        data["iterations"] = resp.get("iterations", [])
        return EvoSim.from_dict(data)


def _load_training_config(config_json_file: Union[str, Path]) -> Dict[str, Any]:
    """Read a training config file and return its keys as a payload dict."""
    path = Path(config_json_file)
    try:
        raw = path.read_text()
    except OSError as e:
        raise LucidicError(f"Cannot read EvoSim config file {path}: {e}") from e

    try:
        config = json.loads(raw)
    except ValueError as e:
        raise LucidicError(f"EvoSim config file {path} is not valid JSON: {e}") from e

    if not isinstance(config, dict):
        raise LucidicError(
            f"EvoSim config file {path} must contain a JSON object, "
            f"got {type(config).__name__}"
        )
    return dict(config)
