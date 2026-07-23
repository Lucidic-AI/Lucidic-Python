"""Experiment resource API operations."""
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from ..client import HttpClient
from ..models.base import CursorPage
from ..models.evaluator import Evaluator
from ..models.experiment import Experiment
from ..pagination import apaginate, paginate
from ...core.errors import require_agent_id

logger = logging.getLogger("Lucidic")


class ExperimentResource:
    """Handle experiment-related API operations."""

    def __init__(
        self,
        http: HttpClient,
        agent_id: Optional[str] = None,
        production: bool = False,
    ):
        """Initialize experiment resource.

        Args:
            http: HTTP client instance
            agent_id: Default agent ID for experiments
            production: Whether to suppress errors in production mode
        """
        self.http = http
        self._agent_id = agent_id
        self._production = production
        # client.experiments.evaluators — attach/detach/list evaluators on an
        # experiment. Stateless (experiment id is passed per call), so one instance
        # is reused across calls.
        self._evaluators = ExperimentEvaluatorsResource(http)

    @property
    def evaluators(self) -> "ExperimentEvaluatorsResource":
        """Wire evaluators onto experiments — e.g.
        ``client.experiments.evaluators.attach(experiment_id, ["accuracy"])`` /
        ``.list(experiment_id)`` / ``.detach(experiment_id, evaluator_id)``."""
        return self._evaluators

    def create(
        self,
        experiment_name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        LLM_boolean_evaluators: Optional[List[str]] = None,
        LLM_numeric_evaluators: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Create a new experiment.

        Args:
            experiment_name: Name of the experiment.
            description: Optional description.
            tags: Optional tags for filtering.
            LLM_boolean_evaluators: Boolean evaluator names.
            LLM_numeric_evaluators: Numeric evaluator names.

        Returns:
            The experiment ID if created successfully, None otherwise.
        """
        evaluator_names = []
        if LLM_boolean_evaluators:
            evaluator_names.extend(LLM_boolean_evaluators)
        if LLM_numeric_evaluators:
            evaluator_names.extend(LLM_numeric_evaluators)

        # LUC-926: an experiment must belong to an agent — raise before the
        # swallow rather than POST a null agent_id.
        require_agent_id(self._agent_id, "experiments.create")
        try:
            response = self.http.post(
                "createexperiment",
                {
                    "agent_id": self._agent_id,
                    "experiment_name": experiment_name,
                    "description": description or "",
                    "tags": tags or [],
                    "evaluator_names": evaluator_names,
                },
            )
            return response.get("experiment_id")
        except Exception as e:
            if self._production:
                logger.error(f"[ExperimentResource] Failed to create experiment: {e}")
                return None
            raise

    async def acreate(
        self,
        experiment_name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        LLM_boolean_evaluators: Optional[List[str]] = None,
        LLM_numeric_evaluators: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Create a new experiment (asynchronous).

        See create() for full documentation.
        """
        evaluator_names = []
        if LLM_boolean_evaluators:
            evaluator_names.extend(LLM_boolean_evaluators)
        if LLM_numeric_evaluators:
            evaluator_names.extend(LLM_numeric_evaluators)

        # LUC-926: same guard as create() — raise before the swallow.
        require_agent_id(self._agent_id, "experiments.create")
        try:
            response = await self.http.apost(
                "createexperiment",
                {
                    "agent_id": self._agent_id,
                    "experiment_name": experiment_name,
                    "description": description or "",
                    "tags": tags or [],
                    "evaluator_names": evaluator_names,
                },
            )
            return response.get("experiment_id")
        except Exception as e:
            if self._production:
                logger.error(f"[ExperimentResource] Failed to create experiment: {e}")
                return None
            raise

    # ==================== v2 reads (LUC-908) ====================
    #
    # Data-bearing reads: they do NOT swallow in production (they surface the
    # typed transport error), unlike create/acreate above. ``agent_id`` defaults
    # to the resource's configured agent. Each read has an async sibling.

    def list(
        self, agent_id: Optional[str] = None, *,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> Iterator[Experiment]:
        """Lazily iterate an agent's experiments, newest first (LUC-816).

        ``ordering`` accepts ``created_at`` / ``id`` (± prefix). Lazy generator —
        errors surface on first iteration; use ``count()`` for a cheap total.
        """
        base = self._list_params(agent_id, ordering, page_size)
        return paginate(lambda cursor: self._page_get(base, cursor), model=Experiment)

    def alist(
        self, agent_id: Optional[str] = None, *,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> AsyncIterator[Experiment]:
        """Async sibling of ``list``."""
        base = self._list_params(agent_id, ordering, page_size)
        return apaginate(lambda cursor: self._apage_get(base, cursor), model=Experiment)

    def list_page(
        self, agent_id: Optional[str] = None, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of experiments (manual pagination control)."""
        body = self._page_get(self._list_params(agent_id, ordering, page_size), cursor)
        return CursorPage.from_body(body, model=Experiment)

    async def alist_page(
        self, agent_id: Optional[str] = None, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``list_page``."""
        body = await self._apage_get(self._list_params(agent_id, ordering, page_size), cursor)
        return CursorPage.from_body(body, model=Experiment)

    def count(self, agent_id: Optional[str] = None) -> int:
        """Cheap total of the agent's experiments via HEAD (``X-Total-Count``)."""
        headers = self.http.head("sdk/v2/experiments", self._list_params(agent_id, None, None))
        return int(headers.get("X-Total-Count") or 0)

    async def acount(self, agent_id: Optional[str] = None) -> int:
        """Async sibling of ``count``."""
        headers = await self.http.ahead("sdk/v2/experiments", self._list_params(agent_id, None, None))
        return int(headers.get("X-Total-Count") or 0)

    def get(self, experiment_id: str) -> Experiment:
        """Read one experiment's detail (metrics + failure groups). Raises
        ``NotFoundError`` if it doesn't exist or the (bound) key can't see it."""
        return Experiment.from_dict(self.http.get(f"sdk/v2/experiments/{experiment_id}"))

    async def aget(self, experiment_id: str) -> Experiment:
        """Async sibling of ``get``."""
        return Experiment.from_dict(await self.http.aget(f"sdk/v2/experiments/{experiment_id}"))

    # ==================== v2 writes (LUC-915) ====================
    #
    # Data-bearing (destructive) write → no production swallow, unlike the gen-3
    # create/acreate above. Each has an async sibling.

    def delete(self, experiment_id: str, delete_sessions: bool = False) -> None:
        """Delete an experiment (DELETE /sdk/v2/experiments/{id}; needs
        ``experiment:delete``). The experiment row is hard-deleted and its
        surviving sessions are detached (their ``experiment`` is set to null).
        ``delete_sessions=True`` instead SOFT-deletes those sessions — hidden but
        recoverable via the retention sweep — and is sent as a JSON body flag.
        An unknown / cross-org / out-of-binding id → ``NotFoundError``."""
        self.http.delete(
            f"sdk/v2/experiments/{experiment_id}", data={"delete_sessions": delete_sessions}
        )

    async def adelete(self, experiment_id: str, delete_sessions: bool = False) -> None:
        """Async sibling of ``delete``."""
        await self.http.adelete(
            f"sdk/v2/experiments/{experiment_id}", data={"delete_sessions": delete_sessions}
        )

    # ---- internals ----

    def _list_params(
        self, agent_id: Optional[str], ordering: Optional[str], page_size: Optional[int]
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "agent_id": require_agent_id(agent_id or self._agent_id, "experiments.list"),
        }
        if ordering is not None:
            params["ordering"] = ordering
        if page_size is not None:
            params["page_size"] = page_size
        return params

    def _page_get(self, base: Dict[str, Any], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base)
        if cursor:
            params["cursor"] = cursor
        return self.http.get("sdk/v2/experiments", params)

    async def _apage_get(self, base: Dict[str, Any], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base)
        if cursor:
            params["cursor"] = cursor
        return await self.http.aget("sdk/v2/experiments", params)


class ExperimentEvaluatorsResource:
    """``client.experiments.evaluators`` — wire evaluators onto an experiment (LUC-915).

    Attach / detach / list the evaluator definitions bound to an experiment. Attach
    is **config-only**: it links evaluators so FUTURE sessions in the experiment
    inherit them at session-init — it does NOT re-score existing sessions or
    recompute metrics. The experiment id is passed per call. Data-bearing writes →
    no production swallow. Each method has an async sibling.
    """

    def __init__(self, http: HttpClient):
        self.http = http

    def list(self, experiment_id: str, *, page_size: Optional[int] = None) -> Iterator[Evaluator]:
        """Lazily iterate the evaluators attached to an experiment (the endpoint is
        cursor-paginated, so this transparently walks every page)."""
        base = self._page_params(page_size)
        return paginate(lambda c: self._page_get(self._path(experiment_id), base, c), model=Evaluator)

    def alist(self, experiment_id: str, *, page_size: Optional[int] = None) -> AsyncIterator[Evaluator]:
        """Async sibling of ``list``."""
        base = self._page_params(page_size)
        return apaginate(lambda c: self._apage_get(self._path(experiment_id), base, c), model=Evaluator)

    def list_page(
        self, experiment_id: str, *, cursor: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of an experiment's attached evaluators."""
        body = self._page_get(self._path(experiment_id), self._page_params(page_size), cursor)
        return CursorPage.from_body(body, model=Evaluator)

    async def alist_page(
        self, experiment_id: str, *, cursor: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``list_page``."""
        body = await self._apage_get(self._path(experiment_id), self._page_params(page_size), cursor)
        return CursorPage.from_body(body, model=Evaluator)

    def attach(self, experiment_id: str, names: List[str]) -> List[Evaluator]:
        """Attach evaluators to an experiment by NAME (POST; needs
        ``experiment:write``). Config-only — future sessions inherit; existing
        sessions are not re-scored. Idempotent (re-attaching is a no-op). If ANY
        name is unknown for the experiment's agent the whole call fails
        (``ValidationError``) and nothing is attached. Returns the experiment's
        full current attached evaluator set."""
        resp = self.http.post(self._path(experiment_id), {"evaluator_names": names})
        return Evaluator.from_list(resp.get("evaluators", []))

    async def aattach(self, experiment_id: str, names: List[str]) -> List[Evaluator]:
        """Async sibling of ``attach``."""
        resp = await self.http.apost(self._path(experiment_id), {"evaluator_names": names})
        return Evaluator.from_list(resp.get("evaluators", []))

    def detach(self, experiment_id: str, evaluator_id: str) -> None:
        """Detach one evaluator (by its UUID) from an experiment (DELETE; needs
        ``experiment:write``). Removes only the link — the evaluator definition and
        any already-recorded metrics are untouched. An evaluator that isn't
        attached (or an unknown id) → ``NotFoundError``."""
        self.http.delete(f"{self._path(experiment_id)}/{evaluator_id}")

    async def adetach(self, experiment_id: str, evaluator_id: str) -> None:
        """Async sibling of ``detach``."""
        await self.http.adelete(f"{self._path(experiment_id)}/{evaluator_id}")

    # ---- internals ----

    @staticmethod
    def _path(experiment_id: str) -> str:
        return f"sdk/v2/experiments/{experiment_id}/evaluators"

    @staticmethod
    def _page_params(page_size: Optional[int]) -> Dict[str, Any]:
        return {"page_size": page_size} if page_size is not None else {}

    def _page_get(self, path: str, base: Dict[str, Any], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base)
        if cursor:
            params["cursor"] = cursor
        return self.http.get(path, params or None)

    async def _apage_get(self, path: str, base: Dict[str, Any], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base)
        if cursor:
            params["cursor"] = cursor
        return await self.http.aget(path, params or None)
