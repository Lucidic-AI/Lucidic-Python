"""LUC-910 — client.evaluator_results.get (single evaluator result by id)."""
import httpx
import pytest
import respx

from lucidicai.api.models.session import EvalResult
from lucidicai.api.resources.evaluator_results import EvaluatorResultsResource
from lucidicai.core.errors import NotFoundError

_BASE = "https://stub.lucidic.test"
_EVAL_RESULTS = f"{_BASE}/sdk/v2/evaluator-results"


@pytest.fixture
def results(http):
    return EvaluatorResultsResource(http)


def _result(i, **over):
    d = {"eval_id": f"r{i}", "evaluator_name": "eval-1", "result": True,
         "result_type": "boolean", "is_pending": False}
    d.update(over)
    return d


class TestGet:
    @respx.mock
    def test_single_result_typed(self, results):
        respx.get(f"{_EVAL_RESULTS}/r1").mock(return_value=httpx.Response(
            200, json=_result(1, result=0.9, result_type="number")))
        r = results.get("r1")
        assert isinstance(r, EvalResult)
        assert r.eval_id == "r1" and r.result == 0.9

    @respx.mock
    def test_404_raises(self, results):
        respx.get(f"{_EVAL_RESULTS}/missing").mock(
            return_value=httpx.Response(404, json={"error": "not found"}))
        with pytest.raises(NotFoundError):
            results.get("missing")

    @respx.mock
    @pytest.mark.asyncio
    async def test_aget(self, results):
        respx.get(f"{_EVAL_RESULTS}/r1").mock(return_value=httpx.Response(200, json=_result(1)))
        r = await results.aget("r1")
        assert r.eval_id == "r1"
