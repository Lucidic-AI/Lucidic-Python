"""Experiment resource API operations."""
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from ..client import HttpClient
from ..models.base import CursorPage
from ..models.evaluator import Evaluator
from ..models.experiment import Experiment, FailureGroup
from ..models.taxonomy import TaxonomyRun, TaxonomyStatus
from ..pagination import apaginate, paginate
from ..polling import await_for, wait_for
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
        # Sub-namespaces (stateless — experiment id passed per call, one instance each).
        # client.experiments.evaluators — attach/detach/list evaluators (LUC-915).
        self._evaluators = ExperimentEvaluatorsResource(http)
        # client.experiments.taxonomy — trace-taxonomy generation (LUC-922).
        self._taxonomy = ExperimentTaxonomyResource(http)

    @property
    def evaluators(self) -> "ExperimentEvaluatorsResource":
        """Wire evaluators onto experiments — e.g.
        ``client.experiments.evaluators.attach(experiment_id, ["accuracy"])`` /
        ``.list(experiment_id)`` / ``.detach(experiment_id, evaluator_id)``."""
        return self._evaluators

    @property
    def taxonomy(self) -> "ExperimentTaxonomyResource":
        """Trace-taxonomy generation on an experiment — e.g.
        ``client.experiments.taxonomy.generate(experiment_id)`` /
        ``.status(experiment_id)`` / ``.wait_for(experiment_id)`` /
        ``.get(experiment_id)``."""
        return self._taxonomy

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

    # ==================== failure modes (LUC-922) ====================
    #
    # Async failure-mode analysis: generate_failure_modes triggers a Temporal run
    # (fire-and-forget — there is NO status endpoint), failure_groups reads the
    # result. Data-bearing → no production swallow. Each has an async sibling.
    # (Taxonomy generation is the client.experiments.taxonomy sub-namespace.)

    def generate_failure_modes(self, experiment_id: str) -> None:
        """Trigger failure-mode analysis for an experiment (POST
        /sdk/v2/experiments/{id}/failure-modes; needs ``failure-group:generate``).
        Clears the prior groups and recomputes them asynchronously; poll
        ``failure_groups`` for the result. Note there is **no status endpoint** — an
        experiment with no failures completes with zero groups, so treat a persistently
        empty list as "no failure modes found", not "still running". Needs at least one
        session (→ ``ValidationError``). May raise ``ServiceUnavailableError`` (503) if
        the calculation can't start — retry."""
        self.http.post(f"sdk/v2/experiments/{experiment_id}/failure-modes")

    async def agenerate_failure_modes(self, experiment_id: str) -> None:
        """Async sibling of ``generate_failure_modes``."""
        await self.http.apost(f"sdk/v2/experiments/{experiment_id}/failure-modes")

    def failure_groups(self, experiment_id: str) -> List[FailureGroup]:
        """Read an experiment's failure-mode groups (GET
        /sdk/v2/experiments/{id}/failure-groups; needs ``failure-group:read``) — the
        result of ``generate_failure_modes``. Not paginated (returns the full set)."""
        resp = self.http.get(f"sdk/v2/experiments/{experiment_id}/failure-groups")
        return FailureGroup.from_list(resp.get("event_failure_groups", []))

    async def afailure_groups(self, experiment_id: str) -> List[FailureGroup]:
        """Async sibling of ``failure_groups``."""
        resp = await self.http.aget(f"sdk/v2/experiments/{experiment_id}/failure-groups")
        return FailureGroup.from_list(resp.get("event_failure_groups", []))

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


class ExperimentTaxonomyResource:
    """``client.experiments.taxonomy`` — trace-taxonomy generation (LUC-922).

    Trigger a taxonomy run over an experiment's finished sessions, read the latest
    completed taxonomy, and poll generation to completion. A taxonomy is the required
    input to dataset generation (``client.datasets.generate``). The experiment id is
    passed per call. Data-bearing → no production swallow. Each method has an async
    sibling.
    """

    def __init__(self, http: HttpClient):
        self.http = http

    def _base(self, experiment_id: str) -> str:
        return f"sdk/v2/experiments/{experiment_id}/taxonomy"

    def generate(
        self, experiment_id: str, *,
        seed_dimensions: Optional[List[Dict[str, Any]]] = None,
        use_previous: Optional[bool] = None,
    ) -> TaxonomyRun:
        """Trigger a taxonomy run (POST /sdk/v2/experiments/{id}/taxonomy; needs
        ``taxonomy:generate``). ``seed_dimensions`` (≤20 ``{name, description?}``) and
        ``use_previous`` (warm-start from the last completed run) are optional. Needs
        ≥20 finished sessions (→ ``ValidationError``) and no run already in progress
        (→ ``ConflictError``). Returns the created run (``run_id`` + ``"queued"``
        status). May raise ``ServiceUnavailableError`` (503) — retry."""
        return TaxonomyRun.from_dict(
            self.http.post(self._base(experiment_id), self._generate_body(seed_dimensions, use_previous)))

    async def agenerate(
        self, experiment_id: str, *,
        seed_dimensions: Optional[List[Dict[str, Any]]] = None,
        use_previous: Optional[bool] = None,
    ) -> TaxonomyRun:
        """Async sibling of ``generate``."""
        return TaxonomyRun.from_dict(
            await self.http.apost(self._base(experiment_id), self._generate_body(seed_dimensions, use_previous)))

    def get(self, experiment_id: str) -> TaxonomyRun:
        """Read the latest **completed** taxonomy (GET /sdk/v2/experiments/{id}/
        taxonomy; needs ``taxonomy:read``) — its dimensions are in ``.taxonomy``.
        Raises ``NotFoundError`` until a run has completed (in-flight state isn't
        surfaced here — use ``status`` / ``wait_for``)."""
        return TaxonomyRun.from_dict(self.http.get(self._base(experiment_id)))

    async def aget(self, experiment_id: str) -> TaxonomyRun:
        """Async sibling of ``get``."""
        return TaxonomyRun.from_dict(await self.http.aget(self._base(experiment_id)))

    def status(self, experiment_id: str) -> TaxonomyStatus:
        """Read taxonomy generation status (GET /sdk/v2/experiments/{id}/taxonomy/
        status; needs ``taxonomy:read``) — the ``ongoing`` run (if any) and the
        ``latest_completed`` taxonomy."""
        return TaxonomyStatus.from_dict(self.http.get(f"{self._base(experiment_id)}/status"))

    async def astatus(self, experiment_id: str) -> TaxonomyStatus:
        """Async sibling of ``status``."""
        return TaxonomyStatus.from_dict(await self.http.aget(f"{self._base(experiment_id)}/status"))

    def wait_for(
        self, experiment_id: str, *, timeout: float = 1800.0, interval: float = 3.0
    ) -> TaxonomyStatus:
        """Block until no taxonomy run is in progress (or ``timeout`` elapses),
        polling ``status`` every ``interval`` seconds. Call this after ``generate``.
        Returns the terminal ``TaxonomyStatus``.

        To tell whether **your** run succeeded, compare the returned
        ``.latest_completed.run_id`` (or ``.version``) against the run ``generate``
        returned: on success they match. A **failed** run clears ``.ongoing`` without
        updating ``.latest_completed`` — so ``.latest_completed`` is then either a
        *prior* completed taxonomy (a run_id/version that is NOT yours — do not treat
        it as your result, and don't feed its stale dimensions into
        ``datasets.generate``) or ``None`` (no taxonomy ever completed). The failure
        reason itself isn't exposed by this endpoint. Raises ``WaitTimeout`` if the
        deadline passes first."""
        return wait_for(
            lambda: self.status(experiment_id),
            is_terminal=lambda s: s.is_terminal, timeout=timeout, interval=interval)

    async def await_for(
        self, experiment_id: str, *, timeout: float = 1800.0, interval: float = 3.0
    ) -> TaxonomyStatus:
        """Async sibling of ``wait_for``."""
        return await await_for(
            lambda: self.astatus(experiment_id),
            is_terminal=lambda s: s.is_terminal, timeout=timeout, interval=interval)

    @staticmethod
    def _generate_body(
        seed_dimensions: Optional[List[Dict[str, Any]]], use_previous: Optional[bool]
    ) -> Dict[str, Any]:
        # omit-None: both fields are optional (backend defaults seed_dimensions=[],
        # use_previous=False).
        body: Dict[str, Any] = {}
        if seed_dimensions is not None:
            body["seed_dimensions"] = seed_dimensions
        if use_previous is not None:
            body["use_previous"] = use_previous
        return body
