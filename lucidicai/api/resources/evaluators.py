"""client.evaluators — evaluator reads (LUC-910) + writes (LUC-916).

A new namespace (today only ``client.evals.emit`` — a different concept, ad-hoc
score submission — exists). Reads the agent's evaluator definitions, one
evaluator's eval distribution across runs, and a single evaluator result by id.
Writes create / update / delete evaluator definitions.

The per-result reads reuse the ``EvalResult`` model (both endpoints serialize
with the backend's ``EvaluatorResultSerializer``). Data-bearing reads + writes —
they do NOT swallow in production. Each has an ``a``-prefixed async sibling.
"""
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from ..client import HttpClient
from ..models.base import CursorPage
from ..models.evaluator import Evaluator
from ..models.session import EvalResult
from ..pagination import apaginate, paginate
from ...core.errors import InvalidOperationError, require_agent_id

_EVALUATORS = "sdk/v2/evaluators"


class EvaluatorsResource:
    """Handle for the ``/sdk/v2/evaluators`` read endpoints."""

    def __init__(self, http: HttpClient, agent_id: Optional[str] = None):
        self.http = http
        self._agent_id = agent_id

    # ==================== evaluators (definitions) ====================

    def list(
        self, agent_id: Optional[str] = None, *,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> Iterator[Evaluator]:
        """Lazily iterate the agent's evaluators, newest first. ``agent_id``
        defaults to the configured agent; ``ordering`` accepts ``id`` (± prefix)."""
        base = self._agent_params(agent_id, ordering, page_size)
        return paginate(lambda c: self._page_get(_EVALUATORS, base, c), model=Evaluator)

    def alist(
        self, agent_id: Optional[str] = None, *,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> AsyncIterator[Evaluator]:
        """Async sibling of ``list``."""
        base = self._agent_params(agent_id, ordering, page_size)
        return apaginate(lambda c: self._apage_get(_EVALUATORS, base, c), model=Evaluator)

    def list_page(
        self, agent_id: Optional[str] = None, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of evaluators."""
        body = self._page_get(_EVALUATORS, self._agent_params(agent_id, ordering, page_size), cursor)
        return CursorPage.from_body(body, model=Evaluator)

    async def alist_page(
        self, agent_id: Optional[str] = None, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``list_page``."""
        body = await self._apage_get(_EVALUATORS, self._agent_params(agent_id, ordering, page_size), cursor)
        return CursorPage.from_body(body, model=Evaluator)

    def get(self, evaluator_id: str) -> Evaluator:
        """Read one evaluator's detail (incl. ``rubric_json`` / ``config``)."""
        return Evaluator.from_dict(self.http.get(f"{_EVALUATORS}/{evaluator_id}"))

    async def aget(self, evaluator_id: str) -> Evaluator:
        """Async sibling of ``get``."""
        return Evaluator.from_dict(await self.http.aget(f"{_EVALUATORS}/{evaluator_id}"))

    # ==================== writes (LUC-916) ====================
    #
    # create / update / delete evaluator definitions. Data-bearing writes → no
    # production swallow. Each has an async sibling. ``create`` is LLM/rubric only
    # (CODE creation is deferred backend-side, LUC-829) and rejects a non-LLM
    # ``type`` client-side, since the backend silently coerces every create to LLM.

    def create(
        self, name: str, *, criteria: List[Dict[str, Any]], result_type: str,
        description: Optional[str] = None, icon: Optional[str] = None,
        is_default: bool = False, type: str = "llm", agent_id: Optional[str] = None,
    ) -> Evaluator:
        """Create an LLM (rubric) evaluator (POST /sdk/v2/evaluators; needs
        ``evaluator:write``). ``result_type`` is ``"boolean"`` (pass/fail) or
        ``"number"`` (score); ``criteria`` is a non-empty list whose per-item shape
        depends on ``result_type`` — boolean: ``{name, pass_definition,
        fail_definition, pass_definition_details?, fail_definition_details?}``;
        number: ``{name, weight?, score_definitions:[{score, definition}]}``. The
        backend builds the rubric from ``criteria`` (don't pass ``rubric_json``). A
        duplicate ``name`` for the agent → ``ConflictError``; a malformed criterion
        or unsupported ``result_type`` → ``ValidationError``. Returns the created
        evaluator in **list shape** — ``criteria`` is populated, but ``rubric_json``
        / ``config`` are not (call ``get(id)`` for the full built rubric).

        Only LLM/rubric evaluators can be created from the SDK — CODE creation is
        deferred (LUC-829). Passing any ``type`` other than ``"llm"`` raises
        ``InvalidOperationError`` (create CODE evaluators in the dashboard)."""
        self._reject_non_llm(type)
        return Evaluator.from_dict(self.http.post(
            _EVALUATORS,
            self._create_body(name, criteria, result_type, description, icon, is_default, agent_id),
        ))

    async def acreate(
        self, name: str, *, criteria: List[Dict[str, Any]], result_type: str,
        description: Optional[str] = None, icon: Optional[str] = None,
        is_default: bool = False, type: str = "llm", agent_id: Optional[str] = None,
    ) -> Evaluator:
        """Async sibling of ``create``."""
        self._reject_non_llm(type)
        return Evaluator.from_dict(await self.http.apost(
            _EVALUATORS,
            self._create_body(name, criteria, result_type, description, icon, is_default, agent_id),
        ))

    def update(
        self, evaluator_id: str, *, name: Optional[str] = None,
        description: Optional[str] = None, icon: Optional[str] = None,
        is_default: Optional[bool] = None, criteria: Optional[List[Dict[str, Any]]] = None,
        result_type: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
    ) -> Evaluator:
        """Partially update an evaluator (PUT /sdk/v2/evaluators/{id}; needs
        ``evaluator:write``). Send only the fields to change: ``name`` /
        ``description`` / ``icon`` / ``is_default`` apply to any type; ``criteria``
        applies to LLM evaluators; ``result_type`` / ``config`` (incl.
        ``config["user_code"]``) apply to CODE evaluators — fields inapplicable to
        the evaluator's type are ignored, so a round-tripped detail body is safe.
        A rename collision → ``ConflictError``. Returns the updated evaluator
        (detail shape, incl. ``rubric_json`` / ``config``)."""
        return Evaluator.from_dict(self.http.put(
            f"{_EVALUATORS}/{evaluator_id}",
            self._update_body(name, description, icon, is_default, criteria, result_type, config),
        ))

    async def aupdate(
        self, evaluator_id: str, *, name: Optional[str] = None,
        description: Optional[str] = None, icon: Optional[str] = None,
        is_default: Optional[bool] = None, criteria: Optional[List[Dict[str, Any]]] = None,
        result_type: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
    ) -> Evaluator:
        """Async sibling of ``update``."""
        return Evaluator.from_dict(await self.http.aput(
            f"{_EVALUATORS}/{evaluator_id}",
            self._update_body(name, description, icon, is_default, criteria, result_type, config),
        ))

    def delete(self, evaluator_id: str) -> None:
        """Hard-delete an evaluator (DELETE /sdk/v2/evaluators/{id}; needs
        ``evaluator:delete``). This also removes its historical evaluator results.
        Unknown / cross-org / out-of-binding id → ``NotFoundError``."""
        self.http.delete(f"{_EVALUATORS}/{evaluator_id}")

    async def adelete(self, evaluator_id: str) -> None:
        """Async sibling of ``delete``."""
        await self.http.adelete(f"{_EVALUATORS}/{evaluator_id}")

    # ==================== one evaluator's eval distribution ====================

    def evals(
        self, evaluator_id: str, *,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> Iterator[EvalResult]:
        """Lazily iterate one evaluator's results across runs (the metric's
        distribution). Returns ``EvalResult`` items."""
        base = self._page_params(ordering, page_size)
        return paginate(
            lambda c: self._page_get(f"{_EVALUATORS}/{evaluator_id}/evals", base, c),
            model=EvalResult,
        )

    def aevals(
        self, evaluator_id: str, *,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> AsyncIterator[EvalResult]:
        """Async sibling of ``evals``."""
        base = self._page_params(ordering, page_size)
        return apaginate(
            lambda c: self._apage_get(f"{_EVALUATORS}/{evaluator_id}/evals", base, c),
            model=EvalResult,
        )

    def evals_page(
        self, evaluator_id: str, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of one evaluator's results."""
        body = self._page_get(f"{_EVALUATORS}/{evaluator_id}/evals", self._page_params(ordering, page_size), cursor)
        return CursorPage.from_body(body, model=EvalResult)

    async def aevals_page(
        self, evaluator_id: str, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``evals_page``."""
        body = await self._apage_get(f"{_EVALUATORS}/{evaluator_id}/evals", self._page_params(ordering, page_size), cursor)
        return CursorPage.from_body(body, model=EvalResult)

    # NOTE: a *single* evaluator result by its own id lives on the separate
    # ``client.evaluator_results`` namespace (get(eval_id)) — keying it off an
    # EvalResult id, not an evaluator id, so it isn't confused with the
    # evaluator-keyed reads above.

    # ==================== internals ====================

    @staticmethod
    def _reject_non_llm(type_: str) -> None:
        """Guard: the SDK creates LLM evaluators only. The backend has no ``type``
        field and silently coerces every create to LLM, so a CODE ``type`` would
        otherwise be honored as an LLM create — reject it here instead. A non-string
        or whitespace-padded value is normalized rather than crashing on ``.lower()``."""
        normalized = type_.strip().lower() if isinstance(type_, str) else type_
        if normalized != "llm":
            raise InvalidOperationError(
                f"creating a '{type_}' evaluator from the SDK is not supported — only "
                "LLM/rubric evaluators can be created (CODE creation is deferred, "
                "LUC-829; create CODE evaluators in the dashboard)."
            )

    def _create_body(
        self, name: str, criteria: List[Dict[str, Any]], result_type: str,
        description: Optional[str], icon: Optional[str], is_default: bool,
        agent_id: Optional[str],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "agent_id": require_agent_id(agent_id or self._agent_id, "evaluators.create"),
            "name": name, "result_type": result_type, "criteria": criteria,
            "is_default": is_default,
        }
        # omit-None: let the backend default description/icon (icon="compass").
        if description is not None:
            body["description"] = description
        if icon is not None:
            body["icon"] = icon
        return body

    @staticmethod
    def _update_body(
        name: Optional[str], description: Optional[str], icon: Optional[str],
        is_default: Optional[bool], criteria: Optional[List[Dict[str, Any]]],
        result_type: Optional[str], config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # omit-None: PUT is a partial update — send only the fields being changed.
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if icon is not None:
            body["icon"] = icon
        if is_default is not None:
            body["is_default"] = is_default
        if criteria is not None:
            body["criteria"] = criteria
        if result_type is not None:
            body["result_type"] = result_type
        if config is not None:
            body["config"] = config
        # An all-None update would send an empty body and silently no-op server-side;
        # surface the likely caller mistake instead.
        if not body:
            raise InvalidOperationError(
                "evaluators.update requires at least one field to change."
            )
        return body

    def _agent_params(
        self, agent_id: Optional[str], ordering: Optional[str], page_size: Optional[int]
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "agent_id": require_agent_id(agent_id or self._agent_id, "evaluators.list"),
        }
        if ordering is not None:
            params["ordering"] = ordering
        if page_size is not None:
            params["page_size"] = page_size
        return params

    @staticmethod
    def _page_params(ordering: Optional[str], page_size: Optional[int]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if ordering is not None:
            params["ordering"] = ordering
        if page_size is not None:
            params["page_size"] = page_size
        return params

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
