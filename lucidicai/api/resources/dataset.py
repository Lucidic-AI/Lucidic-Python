"""Dataset resource API operations."""
import logging
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, List, Optional, Union

from ..client import HttpClient
from ..models.base import CursorPage
from ..models.dataset import (
    CaseEvaluatorUpdate,
    DatasetCase,
    DatasetGenerationRun,
    DatasetItem,
    DatasetSchema,
    EvalRun,
    Fixture,
)
from ..models.evaluator import Evaluator
from ..pagination import apaginate, paginate
from ..polling import await_for, wait_for
from ...core.errors import InvalidOperationError

logger = logging.getLogger("Lucidic")

_SCHEMAS = "sdk/v2/datasets/schemas"
_FIXTURES = "sdk/v2/fixtures"
_GENERATE = "sdk/v2/datasets/generate"


def _id_list(ids: Union[str, Iterable[str]]) -> List[str]:
    """One id or many, as a list: a bare ``str`` is one id, not its characters.
    ``None`` is refused: only ``start_run`` reads it (as every case), before calling this."""
    if ids is None:
        raise InvalidOperationError("item_ids: pass one id or a list")
    return [ids] if isinstance(ids, str) else list(ids)


def _evaluator_choice(
    operation: str, evaluator_id: Optional[str], name: Optional[str],
    criteria: Optional[List[Dict[str, Any]]], result_type: Optional[str],
    description: Optional[str], pass_threshold: Optional[float],
) -> Dict[str, Any]:
    """The body key naming the evaluator: ``{"evaluator_id": ...}`` for an existing
    one, or ``{"evaluator": spec}`` to create or reuse one by name. Exactly one mode,
    and a spec needs ``name`` + ``criteria`` + ``result_type`` — checked here, so a
    mixed or partial call fails before any request."""
    spec_args = {"name": name, "criteria": criteria, "result_type": result_type,
                 "description": description, "pass_threshold": pass_threshold}
    given = [key for key, value in spec_args.items() if value is not None]
    if evaluator_id is not None:
        if given:
            raise InvalidOperationError(
                f"{operation}: pass evaluator_id or an evaluator spec, not both "
                f"(got evaluator_id and {', '.join(given)})."
            )
        return {"evaluator_id": evaluator_id}
    if not given:
        raise InvalidOperationError(
            f"{operation}: pass evaluator_id, or name + criteria + result_type to create "
            "or reuse an evaluator."
        )
    missing = [key for key in ("name", "criteria", "result_type") if spec_args[key] is None]
    if missing:
        raise InvalidOperationError(
            f"{operation}: an evaluator spec needs name, criteria and result_type "
            f"(missing {', '.join(missing)})."
        )
    # omit-None: description and pass_threshold are optional in the spec.
    return {"evaluator": {key: value for key, value in spec_args.items() if value is not None}}


class DatasetResource:
    """Handle dataset-related API operations."""

    def __init__(
        self,
        http: HttpClient,
        agent_id: Optional[str] = None,
        production: bool = False,
    ):
        """Initialize dataset resource.

        Args:
            http: HTTP client instance
            agent_id: Default agent ID for datasets
            production: Whether to suppress errors in production mode
        """
        self.http = http
        self._agent_id = agent_id
        self._production = production
        # v2 sub-namespaces: client.datasets.schemas / .fixtures.
        # Stateless (ids passed per call), so one instance each is reused.
        self._schemas = DatasetSchemasResource(http)
        self._fixtures = DatasetFixturesResource(http)

    # ==================== v2 surface ====================
    #
    # Data-bearing v2 reads/writes: they do NOT swallow in production (typed
    # transport errors propagate), unlike the gen-3 methods below. ``items`` is a
    # cursor-paginated read of a dataset's items; ``schemas`` / ``fixtures`` are
    # sub-namespaces. Each has an async sibling.

    @property
    def schemas(self) -> "DatasetSchemasResource":
        """Dataset schema CRUD — ``client.datasets.schemas.list()`` /
        ``create`` / ``get`` / ``update`` / ``delete`` (org-scoped typed
        definitions of a dataset item's ``input`` shape)."""
        return self._schemas

    @property
    def fixtures(self) -> "DatasetFixturesResource":
        """Fixture authoring — ``client.datasets.fixtures.create(...)`` builds a
        base DuckDB fixture for a ``(dataset, resource)`` pair from JSON rows."""
        return self._fixtures

    def items(self, dataset_id: str, *, page_size: Optional[int] = None) -> Iterator[DatasetItem]:
        """Lazily iterate a dataset's items, newest first (GET
        /sdk/v2/datasets/{id}/items; needs ``dataset-item:read``). Cursor-paginated
        — transparently walks every page. Unknown / cross-org dataset →
        ``NotFoundError`` on first iteration."""
        base = {"page_size": page_size} if page_size is not None else {}
        path = f"sdk/v2/datasets/{dataset_id}/items"
        return paginate(lambda c: self._v2_page_get(path, base, c), model=DatasetItem)

    def aitems(self, dataset_id: str, *, page_size: Optional[int] = None) -> AsyncIterator[DatasetItem]:
        """Async sibling of ``items``."""
        base = {"page_size": page_size} if page_size is not None else {}
        path = f"sdk/v2/datasets/{dataset_id}/items"
        return apaginate(lambda c: self._v2_apage_get(path, base, c), model=DatasetItem)

    def items_page(
        self, dataset_id: str, *, cursor: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of a dataset's items."""
        base = {"page_size": page_size} if page_size is not None else {}
        body = self._v2_page_get(f"sdk/v2/datasets/{dataset_id}/items", base, cursor)
        return CursorPage.from_body(body, model=DatasetItem)

    async def aitems_page(
        self, dataset_id: str, *, cursor: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``items_page``."""
        base = {"page_size": page_size} if page_size is not None else {}
        body = await self._v2_apage_get(f"sdk/v2/datasets/{dataset_id}/items", base, cursor)
        return CursorPage.from_body(body, model=DatasetItem)

    def _v2_page_get(self, path: str, base: Dict[str, Any], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base)
        if cursor:
            params["cursor"] = cursor
        return self.http.get(path, params or None)

    async def _v2_apage_get(self, path: str, base: Dict[str, Any], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base)
        if cursor:
            params["cursor"] = cursor
        return await self.http.aget(path, params or None)

    # ==================== generation ====================
    #
    # The async dataset-generation pipeline: trigger → poll → (retry). Trigger and
    # retry kick off a Temporal workflow and return the run; status reads its
    # progress; wait_for_generation blocks on the wait_for poll helper until the run
    # is terminal. Org-scoped by the key; agent-bound via the run's experiment.

    def generate(
        self, *, agent_id: str, experiment_id: str, schema_id: str, dataset_name: str,
        dataset_description: Optional[str] = None, target_count: Optional[int] = None,
        combo_cap: Optional[int] = None, selected_dimension_ids: Optional[List[str]] = None,
        resource_ids: Optional[List[str]] = None,
    ) -> DatasetGenerationRun:
        """Trigger dataset generation (POST /sdk/v2/datasets/generate; needs
        ``dataset:generate``). Fills a new dataset from ``schema_id`` using the
        ``experiment_id``'s trace taxonomy (the experiment must belong to
        ``agent_id`` and have a completed taxonomy run). ``target_count`` (default
        50, 1–500) and ``combo_cap`` (default 50, 1–200) bound the output; omit to
        take the backend defaults. Returns the created run (``run_id`` +
        ``dataset_id`` + initial ``"queued"`` status) — poll it with
        ``generation_status`` / ``wait_for_generation``.

        Raises ``ValidationError`` (bad refs / no taxonomy / unknown dimensions),
        ``NotFoundError`` (agent / experiment / schema not visible to the key), or
        ``ServiceUnavailableError`` (503 — the workflow couldn't start; retry)."""
        return DatasetGenerationRun.from_dict(self.http.post(_GENERATE, self._generate_body(
            agent_id, experiment_id, schema_id, dataset_name, dataset_description,
            target_count, combo_cap, selected_dimension_ids, resource_ids)))

    async def agenerate(
        self, *, agent_id: str, experiment_id: str, schema_id: str, dataset_name: str,
        dataset_description: Optional[str] = None, target_count: Optional[int] = None,
        combo_cap: Optional[int] = None, selected_dimension_ids: Optional[List[str]] = None,
        resource_ids: Optional[List[str]] = None,
    ) -> DatasetGenerationRun:
        """Async sibling of ``generate``."""
        return DatasetGenerationRun.from_dict(await self.http.apost(_GENERATE, self._generate_body(
            agent_id, experiment_id, schema_id, dataset_name, dataset_description,
            target_count, combo_cap, selected_dimension_ids, resource_ids)))

    def generation_status(self, run_id: str) -> DatasetGenerationRun:
        """Read a generation run's status + progress (GET
        /sdk/v2/datasets/generate/{run_id}/status; needs ``dataset:read``). Unknown
        / cross-org / out-of-binding run → ``NotFoundError``."""
        return DatasetGenerationRun.from_dict(self.http.get(f"{_GENERATE}/{run_id}/status"))

    async def ageneration_status(self, run_id: str) -> DatasetGenerationRun:
        """Async sibling of ``generation_status``."""
        return DatasetGenerationRun.from_dict(await self.http.aget(f"{_GENERATE}/{run_id}/status"))

    def retry_generation(self, run_id: str) -> DatasetGenerationRun:
        """Retry a **failed** generation run (POST
        /sdk/v2/datasets/generate/{run_id}/retry; needs ``dataset:generate``). Only
        a run whose status is ``"failed"`` can be retried (else ``ValidationError``);
        this clones a brand-new dataset + run (the failed one is left in place) and
        returns the new run (new ``run_id`` / ``dataset_id``). May 503 like
        ``generate``."""
        return DatasetGenerationRun.from_dict(self.http.post(f"{_GENERATE}/{run_id}/retry"))

    async def aretry_generation(self, run_id: str) -> DatasetGenerationRun:
        """Async sibling of ``retry_generation``."""
        return DatasetGenerationRun.from_dict(await self.http.apost(f"{_GENERATE}/{run_id}/retry"))

    def wait_for_generation(
        self, run_id: str, *, timeout: float = 1800.0, interval: float = 3.0
    ) -> DatasetGenerationRun:
        """Block until a generation run is terminal (``completed`` or ``failed``) or
        ``timeout`` seconds elapse, polling ``generation_status`` every ``interval``
        seconds. Returns the terminal run — inspect ``.succeeded`` / ``.status`` /
        ``.error_message`` (a **failed** run is returned, not raised; retry it with
        ``retry_generation``). Raises ``WaitTimeout`` if the deadline passes first.

        The default ``timeout`` (30 min) matches the backend's own generation budget
        so a full-length healthy run isn't cut off client-side; raise it only if that
        server budget has been raised."""
        return wait_for(
            lambda: self.generation_status(run_id),
            is_terminal=lambda run: run.is_terminal,
            timeout=timeout, interval=interval,
        )

    async def await_for_generation(
        self, run_id: str, *, timeout: float = 1800.0, interval: float = 3.0
    ) -> DatasetGenerationRun:
        """Async sibling of ``wait_for_generation``."""
        return await await_for(
            lambda: self.ageneration_status(run_id),
            is_terminal=lambda run: run.is_terminal,
            timeout=timeout, interval=interval,
        )

    @staticmethod
    def _generate_body(
        agent_id: str, experiment_id: str, schema_id: str, dataset_name: str,
        dataset_description: Optional[str], target_count: Optional[int],
        combo_cap: Optional[int], selected_dimension_ids: Optional[List[str]],
        resource_ids: Optional[List[str]],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "agent_id": agent_id, "experiment_id": experiment_id,
            "schema_id": schema_id, "dataset_name": dataset_name,
        }
        # omit-None: let the backend apply its defaults (dataset_description="",
        # target_count=50, combo_cap=50, selected_dimension_ids=[], resource_ids=[]).
        if dataset_description is not None:
            body["dataset_description"] = dataset_description
        if target_count is not None:
            body["target_count"] = target_count
        if combo_cap is not None:
            body["combo_cap"] = combo_cap
        if selected_dimension_ids is not None:
            body["selected_dimension_ids"] = selected_dimension_ids
        if resource_ids is not None:
            body["resource_ids"] = resource_ids
        return body

    # ==================== eval runs ====================
    #
    # An eval run sends a Dataset's cases (every one, or those ``item_ids`` names) to
    # the agent's runner ``repeats`` times, then grades and freezes them. start_run
    # kicks off the Temporal workflow (202) and returns the run's ids; runs lists the
    # Dataset's runs newest first, World runs included. Data-bearing → no swallow.

    def start_run(
        self, dataset_id: str, *, webhook_url: str, repeats: Optional[int] = None,
        role: Optional[str] = None, checkpoint_id: Optional[str] = None,
        item_ids: Optional[Union[str, Iterable[str]]] = None, window: Optional[int] = None,
        repeat_deadline_seconds: Optional[int] = None,
    ) -> EvalRun:
        """Start an eval run on a Dataset (POST /sdk/v2/datasets/{id}/runs; needs
        ``evosim:run``): send every case, or the cases ``item_ids`` names (one id or a
        list; they run in the Dataset's order), to the agent's runner at
        ``webhook_url``, ``repeats`` times, ``window`` sessions at a time, each repeat
        within ``repeat_deadline_seconds``. Omit an option to take the backend
        default: ``repeats`` 3 (1–10), ``role`` ``"baseline"`` (or ``"candidate"``),
        no checkpoint, every case, ``window`` 10 (1–50), ``repeat_deadline_seconds``
        1800 (60–43200). A run holds at most 1000 sessions (cases × repeats). Returns
        the started run (``status="running"``); follow it with ``runs``, cancel it
        with ``client.evosims.cancel(run.evosim_id)``.

        Raises ``ValidationError`` (an option out of range; empty, repeated or foreign
        ``item_ids``; a World's entry item; no cases; over 1000 sessions; case inputs
        over 1 MB; a refused runner URL), ``NotFoundError`` (Dataset not visible to the
        key, or a checkpoint not on its agent), ``RateLimitError`` (the fan-out
        throttle) or ``ServiceUnavailableError`` (503 — the workflow didn't start).
        It's a POST, so none of these is retried and a failed call never starts a
        second run. On the 503, ``exc.response_body`` holds the run's ids with
        ``status="start_failed"``; if its message says the run may have started,
        cancel it by ``evosim_id`` before retrying. After a timeout, look for a new
        ``running`` row in ``runs(dataset_id)`` before retrying."""
        return EvalRun.from_dict(self.http.post(
            f"sdk/v2/datasets/{dataset_id}/runs",
            self._run_body(webhook_url, repeats, role, checkpoint_id, item_ids, window,
                           repeat_deadline_seconds),
        ))

    async def astart_run(
        self, dataset_id: str, *, webhook_url: str, repeats: Optional[int] = None,
        role: Optional[str] = None, checkpoint_id: Optional[str] = None,
        item_ids: Optional[Union[str, Iterable[str]]] = None, window: Optional[int] = None,
        repeat_deadline_seconds: Optional[int] = None,
    ) -> EvalRun:
        """Async sibling of ``start_run``."""
        return EvalRun.from_dict(await self.http.apost(
            f"sdk/v2/datasets/{dataset_id}/runs",
            self._run_body(webhook_url, repeats, role, checkpoint_id, item_ids, window,
                           repeat_deadline_seconds),
        ))

    def runs(
        self, dataset_id: str, *, role: Optional[str] = None, status: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> Iterator[EvalRun]:
        """Lazily iterate a Dataset's eval runs, newest first, World runs included (GET
        /sdk/v2/datasets/{id}/runs; needs ``evosim:read``). Filters: ``role``
        (``"baseline"`` / ``"candidate"``) and ``status`` (only ``"complete"`` today);
        another value → ``ValidationError``. A complete run's row carries its compact
        ``summary``. Unknown / not-visible Dataset → ``NotFoundError`` on first
        iteration."""
        base = self._runs_params(role, status, page_size)
        path = f"sdk/v2/datasets/{dataset_id}/runs"
        return paginate(lambda c: self._v2_page_get(path, base, c), model=EvalRun)

    def aruns(
        self, dataset_id: str, *, role: Optional[str] = None, status: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> AsyncIterator[EvalRun]:
        """Async sibling of ``runs``."""
        base = self._runs_params(role, status, page_size)
        path = f"sdk/v2/datasets/{dataset_id}/runs"
        return apaginate(lambda c: self._v2_apage_get(path, base, c), model=EvalRun)

    def runs_page(
        self, dataset_id: str, *, cursor: Optional[str] = None, role: Optional[str] = None,
        status: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of a Dataset's eval runs."""
        body = self._v2_page_get(
            f"sdk/v2/datasets/{dataset_id}/runs", self._runs_params(role, status, page_size), cursor)
        return CursorPage.from_body(body, model=EvalRun)

    async def aruns_page(
        self, dataset_id: str, *, cursor: Optional[str] = None, role: Optional[str] = None,
        status: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``runs_page``."""
        body = await self._v2_apage_get(
            f"sdk/v2/datasets/{dataset_id}/runs", self._runs_params(role, status, page_size), cursor)
        return CursorPage.from_body(body, model=EvalRun)

    @staticmethod
    def _run_body(
        webhook_url: str, repeats: Optional[int], role: Optional[str],
        checkpoint_id: Optional[str], item_ids: Optional[Union[str, Iterable[str]]],
        window: Optional[int], repeat_deadline_seconds: Optional[int],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"webhook_url": webhook_url}
        # omit-None: let the backend apply its defaults (repeats=3, role="baseline", no
        # checkpoint, every case, window=10, repeat_deadline_seconds=1800).
        if repeats is not None:
            body["repeats"] = repeats
        if role is not None:
            body["role"] = role
        if checkpoint_id is not None:
            body["checkpoint_id"] = checkpoint_id
        if item_ids is not None:
            body["item_ids"] = _id_list(item_ids)
        if window is not None:
            body["window"] = window
        if repeat_deadline_seconds is not None:
            body["repeat_deadline_seconds"] = repeat_deadline_seconds
        return body

    @staticmethod
    def _runs_params(
        role: Optional[str], status: Optional[str], page_size: Optional[int]
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if role is not None:
            params["role"] = role
        if status is not None:
            params["status"] = status
        if page_size is not None:
            params["page_size"] = page_size
        return params

    # ==================== case evaluators ====================
    #
    # A case evaluator is the evaluator a case points at: it grades that case first
    # in the grading order, and many cases can share one. Writes go through the
    # many-cases PUT for one id or a list, so they always return a
    # CaseEvaluatorUpdate; case_evaluator reads one case's through the one-case GET.
    # Data-bearing → no swallow. PUT is idempotent (and retried on 429/503).

    def set_case_evaluator(
        self, dataset_id: str, item_ids: Union[str, Iterable[str]], *,
        evaluator_id: Optional[str] = None, name: Optional[str] = None,
        criteria: Optional[List[Dict[str, Any]]] = None, result_type: Optional[str] = None,
        description: Optional[str] = None, pass_threshold: Optional[float] = None,
    ) -> CaseEvaluatorUpdate:
        """Point one case or many (``item_ids``: one id or a list, at most 1000) at the
        evaluator that grades them (PUT /sdk/v2/datasets/{id}/case-evaluator; needs
        ``evaluator:write``). Name an existing evaluator with ``evaluator_id``, **or**
        describe an LLM rubric evaluator with ``name`` + ``criteria`` + ``result_type``
        (+ ``description``, ``pass_threshold``), shaped as for
        ``client.evaluators.create``: it is reused when the agent has one by that name
        with the same rubric and pass mark, else created. A ``"number"`` spec needs a
        ``pass_threshold`` (the score passes at or above it); a ``"boolean"`` one takes
        none. Passing both modes, neither, or a spec without its three required keys
        raises ``InvalidOperationError`` before any request.

        Returns a ``CaseEvaluatorUpdate`` (``evaluator_id``, ``created``,
        ``item_count``, ``changed``). Idempotent: a repeat reads ``changed == 0``; a
        retried call whose first try created the evaluator reads ``created=False``.

        Raises ``ValidationError`` (an id not in the Dataset; the World entry item; an
        evaluator that can't grade cases — not LLM/CODE, a string result, a number
        without a pass mark, the agent's pass/fail metric, a World's grader; a
        malformed spec), ``NotFoundError`` (Dataset not visible to the key, or an
        ``evaluator_id`` that is unknown or another agent's) or ``ConflictError`` (a
        run of the Dataset is in progress — a run reads its cases' evaluators live, so
        change them after it finishes; or the agent has an evaluator by that name with
        another rubric — edit it with ``client.evaluators.update`` or pass its
        ``evaluator_id``)."""
        return CaseEvaluatorUpdate.from_dict(self.http.put(
            self._case_evaluators_path(dataset_id),
            self._case_evaluator_body(
                "datasets.set_case_evaluator", item_ids, evaluator_id, name, criteria,
                result_type, description, pass_threshold),
        ))

    async def aset_case_evaluator(
        self, dataset_id: str, item_ids: Union[str, Iterable[str]], *,
        evaluator_id: Optional[str] = None, name: Optional[str] = None,
        criteria: Optional[List[Dict[str, Any]]] = None, result_type: Optional[str] = None,
        description: Optional[str] = None, pass_threshold: Optional[float] = None,
    ) -> CaseEvaluatorUpdate:
        """Async sibling of ``set_case_evaluator``."""
        return CaseEvaluatorUpdate.from_dict(await self.http.aput(
            self._case_evaluators_path(dataset_id),
            self._case_evaluator_body(
                "datasets.set_case_evaluator", item_ids, evaluator_id, name, criteria,
                result_type, description, pass_threshold),
        ))

    def case_evaluator(self, dataset_id: str, item_id: str) -> Evaluator:
        """Read the evaluator one case points at (GET
        /sdk/v2/datasets/{id}/items/{item_id}/case-evaluator; needs
        ``evaluator:read``), in detail shape with ``datasetitem_id`` set. A case with
        no case evaluator raises ``NotFoundError``, as do an unknown Dataset or item;
        to check without an exception, read ``case_evaluator_id`` on ``items()``, or
        ``graded_by`` / ``grader`` on ``cases()``. A link to another agent's evaluator
        reads as none, since grading ignores it."""
        return Evaluator.from_dict(self.http.get(self._case_evaluator_path(dataset_id, item_id)))

    async def acase_evaluator(self, dataset_id: str, item_id: str) -> Evaluator:
        """Async sibling of ``case_evaluator``."""
        return Evaluator.from_dict(await self.http.aget(self._case_evaluator_path(dataset_id, item_id)))

    def remove_case_evaluator(
        self, dataset_id: str, item_ids: Union[str, Iterable[str]]
    ) -> CaseEvaluatorUpdate:
        """Unlink one case or many from their case evaluator (PUT
        /sdk/v2/datasets/{id}/case-evaluator with ``evaluator_id: null``; needs
        ``evaluator:write``). The evaluator itself is kept — delete it with
        ``client.evaluators.delete``. Idempotent: ``changed`` counts only the cases
        that had one. An id not in the Dataset → ``ValidationError``; Dataset not
        visible to the key → ``NotFoundError``; a run of the Dataset in progress →
        ``ConflictError``."""
        return CaseEvaluatorUpdate.from_dict(self.http.put(
            self._case_evaluators_path(dataset_id),
            {"item_ids": _id_list(item_ids), "evaluator_id": None},
        ))

    async def aremove_case_evaluator(
        self, dataset_id: str, item_ids: Union[str, Iterable[str]]
    ) -> CaseEvaluatorUpdate:
        """Async sibling of ``remove_case_evaluator``."""
        return CaseEvaluatorUpdate.from_dict(await self.http.aput(
            self._case_evaluators_path(dataset_id),
            {"item_ids": _id_list(item_ids), "evaluator_id": None},
        ))

    @staticmethod
    def _case_evaluators_path(dataset_id: str) -> str:
        return f"sdk/v2/datasets/{dataset_id}/case-evaluator"

    @staticmethod
    def _case_evaluator_path(dataset_id: str, item_id: str) -> str:
        return f"sdk/v2/datasets/{dataset_id}/items/{item_id}/case-evaluator"

    @staticmethod
    def _case_evaluator_body(
        operation: str, item_ids: Union[str, Iterable[str]], evaluator_id: Optional[str],
        name: Optional[str], criteria: Optional[List[Dict[str, Any]]],
        result_type: Optional[str], description: Optional[str], pass_threshold: Optional[float],
    ) -> Dict[str, Any]:
        # check the evaluator choice before touching item_ids, so a bad call never
        # consumes a caller's generator.
        choice = _evaluator_choice(
            operation, evaluator_id, name, criteria, result_type, description, pass_threshold)
        return {"item_ids": _id_list(item_ids), **choice}

    # ==================== fallback evaluator ====================
    #
    # The Dataset's fallback evaluator grades a case that has no case evaluator and no
    # agent-reported result; it runs only then. One per Dataset. Set it like a case
    # evaluator (by id or by spec); clearing is remove_fallback_evaluator, so a bare
    # set call can never clear it. Data-bearing → no swallow.

    def set_fallback_evaluator(
        self, dataset_id: str, evaluator_id: Optional[str] = None, *,
        name: Optional[str] = None, criteria: Optional[List[Dict[str, Any]]] = None,
        result_type: Optional[str] = None, description: Optional[str] = None,
        pass_threshold: Optional[float] = None,
    ) -> Evaluator:
        """Set the Dataset's fallback evaluator (PUT
        /sdk/v2/datasets/{id}/fallback-evaluator; needs ``evaluator:write``). Name an
        existing evaluator with ``evaluator_id``, **or** describe an LLM rubric
        evaluator with ``name`` + ``criteria`` + ``result_type`` (+ ``description``,
        ``pass_threshold``), as for ``set_case_evaluator``: reused by name when the
        rubric and pass mark match, else created. A ``"number"`` spec needs a
        ``pass_threshold``. Passing both modes, neither, or a partial spec raises
        ``InvalidOperationError`` before any request — to clear, call
        ``remove_fallback_evaluator``.

        Returns the evaluator in detail shape with ``dataset_id`` set and ``created``
        True when a spec made it. Idempotent; a retried call whose first try created
        it reads ``created=False``.

        Raises ``ValidationError`` (an evaluator that can't grade cases — the same
        refusals as ``set_case_evaluator``; a malformed spec), ``NotFoundError``
        (Dataset not visible to the key, or an ``evaluator_id`` that is unknown or
        another agent's) or
        ``ConflictError`` (a run of the Dataset is in progress — change it after the
        run finishes; or the agent has an evaluator by that name with another
        rubric)."""
        return Evaluator.from_dict(self.http.put(
            self._fallback_path(dataset_id),
            _evaluator_choice("datasets.set_fallback_evaluator", evaluator_id, name, criteria,
                              result_type, description, pass_threshold),
        ))

    async def aset_fallback_evaluator(
        self, dataset_id: str, evaluator_id: Optional[str] = None, *,
        name: Optional[str] = None, criteria: Optional[List[Dict[str, Any]]] = None,
        result_type: Optional[str] = None, description: Optional[str] = None,
        pass_threshold: Optional[float] = None,
    ) -> Evaluator:
        """Async sibling of ``set_fallback_evaluator``."""
        return Evaluator.from_dict(await self.http.aput(
            self._fallback_path(dataset_id),
            _evaluator_choice("datasets.set_fallback_evaluator", evaluator_id, name, criteria,
                              result_type, description, pass_threshold),
        ))

    def fallback_evaluator(self, dataset_id: str) -> Evaluator:
        """Read the Dataset's fallback evaluator (GET
        /sdk/v2/datasets/{id}/fallback-evaluator; needs ``evaluator:read``), in detail
        shape with ``dataset_id`` set. None set raises ``NotFoundError``, as does an
        unknown Dataset; to check without an exception, read ``fallback_evaluator_id``
        from ``client.datasets.get(dataset_id)``. A link to another agent's evaluator
        reads as none, since grading ignores it."""
        return Evaluator.from_dict(self.http.get(self._fallback_path(dataset_id)))

    async def afallback_evaluator(self, dataset_id: str) -> Evaluator:
        """Async sibling of ``fallback_evaluator``."""
        return Evaluator.from_dict(await self.http.aget(self._fallback_path(dataset_id)))

    def remove_fallback_evaluator(self, dataset_id: str) -> None:
        """Clear the Dataset's fallback evaluator (DELETE
        /sdk/v2/datasets/{id}/fallback-evaluator; needs ``evaluator:write``). The
        evaluator itself is kept. Idempotent: clearing an unset fallback succeeds.
        Dataset not visible to the key → ``NotFoundError``; a run of the Dataset in
        progress → ``ConflictError``."""
        self.http.delete(self._fallback_path(dataset_id))

    async def aremove_fallback_evaluator(self, dataset_id: str) -> None:
        """Async sibling of ``remove_fallback_evaluator``."""
        await self.http.adelete(self._fallback_path(dataset_id))

    @staticmethod
    def _fallback_path(dataset_id: str) -> str:
        return f"sdk/v2/datasets/{dataset_id}/fallback-evaluator"

    # ==================== cases list ====================
    #
    # A Dataset's cases, each with what will grade it (the same pick grading makes),
    # in run order. Cursor-paginated like items; a page holds at most 50, since each
    # row carries its grader's criteria. Data-bearing → no swallow.

    def cases(self, dataset_id: str, *, page_size: Optional[int] = None) -> Iterator[DatasetCase]:
        """Lazily iterate a Dataset's cases with what will grade each one, in run order
        (oldest first, as a run takes them; GET /sdk/v2/datasets/{id}/cases; needs
        ``dataset-item:read``). Each ``DatasetCase`` is the item plus ``scored``,
        ``graded_by``, ``grader`` and ``if_not_reported``, decided from the current
        case evaluators, the agent's pass/fail metric and the Dataset's fallback.
        ``page_size`` is capped at 50. A grader's ``criteria`` is ``None`` unless the
        key also has ``evaluator:read``. Unknown / not-visible Dataset →
        ``NotFoundError`` on first iteration."""
        base = {"page_size": page_size} if page_size is not None else {}
        path = f"sdk/v2/datasets/{dataset_id}/cases"
        return paginate(lambda c: self._v2_page_get(path, base, c), model=DatasetCase)

    def acases(self, dataset_id: str, *, page_size: Optional[int] = None) -> AsyncIterator[DatasetCase]:
        """Async sibling of ``cases``."""
        base = {"page_size": page_size} if page_size is not None else {}
        path = f"sdk/v2/datasets/{dataset_id}/cases"
        return apaginate(lambda c: self._v2_apage_get(path, base, c), model=DatasetCase)

    def cases_page(
        self, dataset_id: str, *, cursor: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of a Dataset's cases."""
        base = {"page_size": page_size} if page_size is not None else {}
        body = self._v2_page_get(f"sdk/v2/datasets/{dataset_id}/cases", base, cursor)
        return CursorPage.from_body(body, model=DatasetCase)

    async def acases_page(
        self, dataset_id: str, *, cursor: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``cases_page``."""
        base = {"page_size": page_size} if page_size is not None else {}
        body = await self._v2_apage_get(f"sdk/v2/datasets/{dataset_id}/cases", base, cursor)
        return CursorPage.from_body(body, model=DatasetCase)

    # ==================== Dataset Methods ====================

    def list(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """List all datasets for agent.

        Args:
            agent_id: Optional agent ID to filter by (uses default if not provided)

        Returns:
            Dictionary with num_datasets and datasets list
        """
        try:
            params = {}
            if agent_id or self._agent_id:
                params["agent_id"] = agent_id or self._agent_id
            return self.http.get("sdk/datasets", params)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to list datasets: {e}")
                return {"num_datasets": 0, "datasets": []}
            raise

    def create(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        suggested_flag_config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create new dataset.

        Args:
            name: Dataset name (must be unique per agent)
            description: Optional description
            tags: Optional list of tags
            suggested_flag_config: Optional flag configuration
            agent_id: Optional agent ID (uses default if not provided)

        Returns:
            Dictionary with dataset_id
        """
        try:
            data: Dict[str, Any] = {"name": name}
            if description is not None:
                data["description"] = description
            if tags is not None:
                data["tags"] = tags
            if suggested_flag_config is not None:
                data["suggested_flag_config"] = suggested_flag_config
            data["agent_id"] = agent_id or self._agent_id
            return self.http.post("sdk/datasets/create", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to create dataset: {e}")
                return {}
            raise

    def get(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset with all items.

        Args:
            dataset_id: Dataset UUID

        Returns:
            Full dataset data including all items
        """
        try:
            return self.http.get("getdataset", {"dataset_id": dataset_id})
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to get dataset: {e}")
                return {}
            raise

    def update(self, dataset_id: str, **kwargs) -> Dict[str, Any]:
        """Update dataset metadata.

        Args:
            dataset_id: Dataset UUID
            **kwargs: Fields to update (name, description, tags, suggested_flag_config)

        Returns:
            Updated dataset data
        """
        try:
            data = {"dataset_id": dataset_id}
            data.update(kwargs)
            return self.http.put("sdk/datasets/update", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to update dataset: {e}")
                return {}
            raise

    def delete(self, dataset_id: str) -> Dict[str, Any]:
        """Delete dataset and all items.

        Args:
            dataset_id: Dataset UUID

        Returns:
            Success message
        """
        try:
            return self.http.delete("sdk/datasets/delete", {"dataset_id": dataset_id})
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to delete dataset: {e}")
                return {}
            raise

    # ==================== Dataset Item Methods ====================

    def create_item(
        self,
        dataset_id: str,
        name: str,
        input_data: Dict[str, Any],
        expected_output: Optional[Any] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        flag_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create dataset item.

        Args:
            dataset_id: Dataset UUID
            name: Item name
            input_data: Input data dictionary
            expected_output: Optional expected output
            description: Optional description
            tags: Optional list of tags
            metadata: Optional metadata dictionary
            flag_overrides: Optional flag overrides

        Returns:
            Dictionary with datasetitem_id
        """
        try:
            data: Dict[str, Any] = {
                "dataset_id": dataset_id,
                "name": name,
                "input": input_data
            }

            if expected_output is not None:
                data["expected_output"] = expected_output
            if description is not None:
                data["description"] = description
            if tags is not None:
                data["tags"] = tags
            if metadata is not None:
                data["metadata"] = metadata
            if flag_overrides is not None:
                data["flag_overrides"] = flag_overrides

            return self.http.post("sdk/datasets/items/create", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to create item: {e}")
                return {}
            raise

    def get_item(self, dataset_id: str, item_id: str) -> Dict[str, Any]:
        """Get specific dataset item.

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Dataset item data
        """
        try:
            return self.http.get("sdk/datasets/items/get", {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            })
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to get item: {e}")
                return {}
            raise

    def update_item(self, dataset_id: str, item_id: str, **kwargs) -> Dict[str, Any]:
        """Update dataset item.

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID
            **kwargs: Fields to update

        Returns:
            Updated item data
        """
        try:
            data = {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            }
            data.update(kwargs)
            return self.http.put("sdk/datasets/items/update", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to update item: {e}")
                return {}
            raise

    def delete_item(self, dataset_id: str, item_id: str) -> Dict[str, Any]:
        """Delete dataset item.

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Success message
        """
        try:
            return self.http.delete("sdk/datasets/items/delete", {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            })
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to delete item: {e}")
                return {}
            raise

    def list_item_sessions(self, dataset_id: str, item_id: str) -> Dict[str, Any]:
        """List all sessions for a dataset item.

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Dictionary with num_sessions and sessions list
        """
        try:
            return self.http.get("sdk/datasets/items/sessions", {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            })
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to list item sessions: {e}")
                return {"num_sessions": 0, "sessions": []}
            raise

    # ==================== Asynchronous Dataset Methods ====================

    async def alist(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """List all datasets for agent (asynchronous).

        Args:
            agent_id: Optional agent ID to filter by (uses default if not provided)

        Returns:
            Dictionary with num_datasets and datasets list
        """
        try:
            params = {}
            if agent_id or self._agent_id:
                params["agent_id"] = agent_id or self._agent_id
            return await self.http.aget("sdk/datasets", params)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to list datasets: {e}")
                return {"num_datasets": 0, "datasets": []}
            raise

    async def acreate(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        suggested_flag_config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create new dataset (asynchronous).

        Args:
            name: Dataset name (must be unique per agent)
            description: Optional description
            tags: Optional list of tags
            suggested_flag_config: Optional flag configuration
            agent_id: Optional agent ID (uses default if not provided)

        Returns:
            Dictionary with dataset_id
        """
        try:
            data: Dict[str, Any] = {"name": name}
            if description is not None:
                data["description"] = description
            if tags is not None:
                data["tags"] = tags
            if suggested_flag_config is not None:
                data["suggested_flag_config"] = suggested_flag_config
            data["agent_id"] = agent_id or self._agent_id
            return await self.http.apost("sdk/datasets/create", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to create dataset: {e}")
                return {}
            raise

    async def aget(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset with all items (asynchronous).

        Args:
            dataset_id: Dataset UUID

        Returns:
            Full dataset data including all items
        """
        try:
            return await self.http.aget("getdataset", {"dataset_id": dataset_id})
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to get dataset: {e}")
                return {}
            raise

    async def aupdate(self, dataset_id: str, **kwargs) -> Dict[str, Any]:
        """Update dataset metadata (asynchronous).

        Args:
            dataset_id: Dataset UUID
            **kwargs: Fields to update (name, description, tags, suggested_flag_config)

        Returns:
            Updated dataset data
        """
        try:
            data = {"dataset_id": dataset_id}
            data.update(kwargs)
            return await self.http.aput("sdk/datasets/update", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to update dataset: {e}")
                return {}
            raise

    async def adelete(self, dataset_id: str) -> Dict[str, Any]:
        """Delete dataset and all items (asynchronous).

        Args:
            dataset_id: Dataset UUID

        Returns:
            Success message
        """
        try:
            return await self.http.adelete("sdk/datasets/delete", {"dataset_id": dataset_id})
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to delete dataset: {e}")
                return {}
            raise

    # ==================== Asynchronous Item Methods ====================

    async def acreate_item(
        self,
        dataset_id: str,
        name: str,
        input_data: Dict[str, Any],
        expected_output: Optional[Any] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        flag_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create dataset item (asynchronous).

        Args:
            dataset_id: Dataset UUID
            name: Item name
            input_data: Input data dictionary
            expected_output: Optional expected output
            description: Optional description
            tags: Optional list of tags
            metadata: Optional metadata dictionary
            flag_overrides: Optional flag overrides

        Returns:
            Dictionary with datasetitem_id
        """
        try:
            data: Dict[str, Any] = {
                "dataset_id": dataset_id,
                "name": name,
                "input": input_data
            }

            if expected_output is not None:
                data["expected_output"] = expected_output
            if description is not None:
                data["description"] = description
            if tags is not None:
                data["tags"] = tags
            if metadata is not None:
                data["metadata"] = metadata
            if flag_overrides is not None:
                data["flag_overrides"] = flag_overrides

            return await self.http.apost("sdk/datasets/items/create", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to create item: {e}")
                return {}
            raise

    async def aget_item(self, dataset_id: str, item_id: str) -> Dict[str, Any]:
        """Get specific dataset item (asynchronous).

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Dataset item data
        """
        try:
            return await self.http.aget("sdk/datasets/items/get", {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            })
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to get item: {e}")
                return {}
            raise

    async def aupdate_item(self, dataset_id: str, item_id: str, **kwargs) -> Dict[str, Any]:
        """Update dataset item (asynchronous).

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID
            **kwargs: Fields to update

        Returns:
            Updated item data
        """
        try:
            data = {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            }
            data.update(kwargs)
            return await self.http.aput("sdk/datasets/items/update", data)
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to update item: {e}")
                return {}
            raise

    async def adelete_item(self, dataset_id: str, item_id: str) -> Dict[str, Any]:
        """Delete dataset item (asynchronous).

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Success message
        """
        try:
            return await self.http.adelete("sdk/datasets/items/delete", {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            })
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to delete item: {e}")
                return {}
            raise

    async def alist_item_sessions(self, dataset_id: str, item_id: str) -> Dict[str, Any]:
        """List all sessions for a dataset item (asynchronous).

        Args:
            dataset_id: Dataset UUID
            item_id: Item UUID

        Returns:
            Dictionary with num_sessions and sessions list
        """
        try:
            return await self.http.aget("sdk/datasets/items/sessions", {
                "dataset_id": dataset_id,
                "datasetitem_id": item_id
            })
        except Exception as e:
            if self._production:
                logger.error(f"[DatasetResource] Failed to list item sessions: {e}")
                return {"num_sessions": 0, "sessions": []}
            raise


class DatasetSchemasResource:
    """``client.datasets.schemas`` — dataset schema CRUD.

    A dataset schema is an org-scoped typed definition of a dataset item's ``input``
    shape (required for dataset generation). Org-scoped: a bound key reaches them
    all. Data-bearing — no production swallow. Each method has an async sibling.
    """

    def __init__(self, http: HttpClient):
        self.http = http

    def list(
        self, *, ordering: Optional[str] = None, page_size: Optional[int] = None
    ) -> Iterator[DatasetSchema]:
        """Lazily iterate the org's dataset schemas, newest first. ``ordering``
        accepts ``id`` (± prefix)."""
        base = self._params(ordering, page_size)
        return paginate(lambda c: self._page_get(base, c), model=DatasetSchema)

    def alist(
        self, *, ordering: Optional[str] = None, page_size: Optional[int] = None
    ) -> AsyncIterator[DatasetSchema]:
        """Async sibling of ``list``."""
        base = self._params(ordering, page_size)
        return apaginate(lambda c: self._apage_get(base, c), model=DatasetSchema)

    def list_page(
        self, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of dataset schemas."""
        return CursorPage.from_body(
            self._page_get(self._params(ordering, page_size), cursor), model=DatasetSchema
        )

    async def alist_page(
        self, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``list_page``."""
        body = await self._apage_get(self._params(ordering, page_size), cursor)
        return CursorPage.from_body(body, model=DatasetSchema)

    def get(self, schema_id: str) -> DatasetSchema:
        """Read one schema by id (raises ``NotFoundError`` if absent/cross-org)."""
        return DatasetSchema.from_dict(self.http.get(f"{_SCHEMAS}/{schema_id}"))

    async def aget(self, schema_id: str) -> DatasetSchema:
        """Async sibling of ``get``."""
        return DatasetSchema.from_dict(await self.http.aget(f"{_SCHEMAS}/{schema_id}"))

    def create(
        self, name: str, *, fields: List[Dict[str, Any]], description: Optional[str] = None
    ) -> DatasetSchema:
        """Create a dataset schema in the key's org (POST /sdk/v2/datasets/schemas;
        needs ``dataset-schema:write``). ``fields`` is a non-empty list of typed
        field defs — each ``{key, type, description?, required?, options?,
        children?}`` where ``type`` is ``string`` / ``number`` / ``categorical`` /
        ``object`` (``categorical`` needs ``options``; ``object`` needs
        ``children``). ``name`` is unique per org — a duplicate → ``ConflictError``;
        a malformed field → ``ValidationError``."""
        return DatasetSchema.from_dict(
            self.http.post(_SCHEMAS, self._create_body(name, fields, description))
        )

    async def acreate(
        self, name: str, *, fields: List[Dict[str, Any]], description: Optional[str] = None
    ) -> DatasetSchema:
        """Async sibling of ``create``."""
        return DatasetSchema.from_dict(
            await self.http.apost(_SCHEMAS, self._create_body(name, fields, description))
        )

    def update(
        self, schema_id: str, *, name: Optional[str] = None,
        description: Optional[str] = None, fields: Optional[List[Dict[str, Any]]] = None,
    ) -> DatasetSchema:
        """Partially update a schema (PUT /sdk/v2/datasets/schemas/{id}; needs
        ``dataset-schema:write``). Send only the fields to change. A rename
        collision → ``ConflictError``."""
        return DatasetSchema.from_dict(
            self.http.put(f"{_SCHEMAS}/{schema_id}", self._update_body(name, description, fields))
        )

    async def aupdate(
        self, schema_id: str, *, name: Optional[str] = None,
        description: Optional[str] = None, fields: Optional[List[Dict[str, Any]]] = None,
    ) -> DatasetSchema:
        """Async sibling of ``update``."""
        return DatasetSchema.from_dict(
            await self.http.aput(f"{_SCHEMAS}/{schema_id}", self._update_body(name, description, fields))
        )

    def delete(self, schema_id: str) -> None:
        """Delete a schema (DELETE /sdk/v2/datasets/schemas/{id}; needs
        ``dataset-schema:delete``). Datasets/generation runs pointing at it are
        un-linked (their schema pointer is set null), not deleted. A missing /
        cross-org id → ``NotFoundError``."""
        self.http.delete(f"{_SCHEMAS}/{schema_id}")

    async def adelete(self, schema_id: str) -> None:
        """Async sibling of ``delete``."""
        await self.http.adelete(f"{_SCHEMAS}/{schema_id}")

    # ---- internals ----

    @staticmethod
    def _params(ordering: Optional[str], page_size: Optional[int]) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if ordering is not None:
            params["ordering"] = ordering
        if page_size is not None:
            params["page_size"] = page_size
        return params or None

    def _page_get(self, base: Optional[Dict[str, Any]], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base or {})
        if cursor:
            params["cursor"] = cursor
        return self.http.get(_SCHEMAS, params or None)

    async def _apage_get(self, base: Optional[Dict[str, Any]], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base or {})
        if cursor:
            params["cursor"] = cursor
        return await self.http.aget(_SCHEMAS, params or None)

    @staticmethod
    def _create_body(
        name: str, fields: List[Dict[str, Any]], description: Optional[str]
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name, "fields": fields}
        if description is not None:
            body["description"] = description
        return body

    @staticmethod
    def _update_body(
        name: Optional[str], description: Optional[str], fields: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if fields is not None:
            body["fields"] = fields
        return body


class DatasetFixturesResource:
    """``client.datasets.fixtures`` — author base fixtures from JSON rows.

    A fixture is the hydrated mock data (a DuckDB blob) for one ``(dataset,
    resource)`` pair. ``create`` builds one from explicit rows. Data-bearing — no
    production swallow. Has an async sibling.
    """

    def __init__(self, http: HttpClient):
        self.http = http

    def create(
        self, *, resource_id: str, dataset_id: str, tables: List[Dict[str, Any]]
    ) -> Fixture:
        """Author a base fixture from JSON rows (POST /sdk/v2/fixtures; needs
        ``fixture:write``). ``tables`` is a list of ``{"name": <table>, "rows":
        [<row dict>, ...]}`` validated against the resource's ``spec``. Exactly one
        fixture may exist per ``(dataset, resource)`` — a duplicate →
        ``ConflictError``; a row/spec mismatch → ``ValidationError``. Unknown
        resource or dataset → ``NotFoundError``."""
        return Fixture.from_dict(self.http.post(_FIXTURES, self._body(resource_id, dataset_id, tables)))

    async def acreate(
        self, *, resource_id: str, dataset_id: str, tables: List[Dict[str, Any]]
    ) -> Fixture:
        """Async sibling of ``create``."""
        return Fixture.from_dict(
            await self.http.apost(_FIXTURES, self._body(resource_id, dataset_id, tables))
        )

    @staticmethod
    def _body(resource_id: str, dataset_id: str, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"resource_id": resource_id, "dataset_id": dataset_id, "tables": tables}
