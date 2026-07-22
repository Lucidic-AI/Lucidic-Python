"""client.evaluators — evaluator reads (LUC-910).

A new namespace (today only ``client.evals.emit`` — a different concept, ad-hoc
score submission — exists). Reads the agent's evaluator definitions, one
evaluator's eval distribution across runs, and a single evaluator result by id.

The per-result reads reuse the ``EvalResult`` model (both endpoints serialize
with the backend's ``EvaluatorResultSerializer``). Data-bearing reads — they do
NOT swallow in production. Each has an ``a``-prefixed async sibling.
"""
from typing import Any, AsyncIterator, Dict, Iterator, Optional

from ..client import HttpClient
from ..models.base import CursorPage
from ..models.evaluator import Evaluator
from ..models.session import EvalResult
from ..pagination import apaginate, paginate

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

    def _agent_params(
        self, agent_id: Optional[str], ordering: Optional[str], page_size: Optional[int]
    ) -> Dict[str, Any]:
        resolved = agent_id or self._agent_id
        params: Dict[str, Any] = {}
        if resolved is not None:
            params["agent_id"] = resolved
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
