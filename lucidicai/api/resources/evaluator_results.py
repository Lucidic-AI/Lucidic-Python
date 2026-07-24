"""client.evaluator_results — a single evaluator result by id (LUC-910).

A small namespace keyed by an ``EvaluatorResult`` id (distinct from an evaluator
id), so ``evaluator_results.get(eval_id)`` can't be confused with the
evaluator-keyed reads on ``client.evaluators``. Reuses the ``EvalResult`` model
(the ``EvaluatorResultSerializer`` shape). Data-bearing read — does NOT swallow
in production.
"""
from ..client import HttpClient
from ..models.session import EvalResult

_EVAL_RESULTS = "sdk/v2/evaluator-results"


class EvaluatorResultsResource:
    """Handle for the ``/sdk/v2/evaluator-results`` read endpoint."""

    def __init__(self, http: HttpClient):
        self.http = http

    def get(self, eval_id: str) -> EvalResult:
        """Read one evaluator result by its id (a CODE result is synced from S3
        on read). ``eval_id`` is an ``EvalResult.eval_id``."""
        return EvalResult.from_dict(self.http.get(f"{_EVAL_RESULTS}/{eval_id}"))

    async def aget(self, eval_id: str) -> EvalResult:
        """Async sibling of ``get``."""
        return EvalResult.from_dict(await self.http.aget(f"{_EVAL_RESULTS}/{eval_id}"))
