"""LUC-1195 — evals.emit(pass_threshold=...): the pass mark for a number result.

The backend's POST /sdk/evals takes ``pass_threshold`` only with a number result and
only as a finite number (``scoring_services._check_threshold``). emit posts from a
background thread, so a backend 4xx would never reach the caller: the SDK checks the
same rule up front, as it does for ``result_type``.
"""
import json
import math

import httpx
import pytest
import respx

from lucidicai.api.resources.evals import EvalsResource

_EVALS = "https://stub.lucidic.test/sdk/evals"
_SESSION_ID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def evals(http):
    return EvalsResource(http, production=False)


def _sent(route):
    body = json.loads(route.calls.last.request.read())
    body.pop("current_time")  # the transport stamps every POST
    return body


class TestPassThresholdSent:
    @respx.mock
    def test_sent_when_given(self, evals):
        route = respx.post(_EVALS).mock(return_value=httpx.Response(201))
        evals.emit(0.85, name="criterion_pass_rate", pass_threshold=0.8, session_id=_SESSION_ID)
        assert evals.flush(timeout=5) is True
        assert _sent(route) == {
            "session_id": _SESSION_ID, "result": 0.85, "result_type": "number",
            "name": "criterion_pass_rate", "pass_threshold": 0.8,
        }

    @respx.mock
    def test_zero_is_sent(self, evals):
        route = respx.post(_EVALS).mock(return_value=httpx.Response(201))
        evals.emit(0.0, name="criterion_pass_rate", pass_threshold=0, session_id=_SESSION_ID)
        assert evals.flush(timeout=5) is True
        assert _sent(route)["pass_threshold"] == 0

    @respx.mock
    def test_omitted_when_not_given(self, evals):
        route = respx.post(_EVALS).mock(return_value=httpx.Response(201))
        evals.emit(0.85, name="criterion_pass_rate", session_id=_SESSION_ID)
        assert evals.flush(timeout=5) is True
        assert "pass_threshold" not in _sent(route)

    @respx.mock
    def test_int_mark_and_explicit_number_type(self, evals):
        route = respx.post(_EVALS).mock(return_value=httpx.Response(201))
        evals.emit(7, name="score", result_type="number", pass_threshold=5, session_id=_SESSION_ID)
        assert evals.flush(timeout=5) is True
        body = _sent(route)
        assert body["pass_threshold"] == 5 and body["result_type"] == "number"


class TestPassThresholdWithoutName:
    @respx.mock
    @pytest.mark.parametrize("name", [None, ""])
    def test_warns_about_the_shared_default_evaluator(self, evals, caplog, name):
        route = respx.post(_EVALS).mock(return_value=httpx.Response(201))
        with caplog.at_level("WARNING", logger="Lucidic"):
            evals.emit(0.9, name=name, pass_threshold=0.5, session_id=_SESSION_ID)
        assert evals.flush(timeout=5) is True
        assert "User Supplied (Number)" in caplog.text
        assert _sent(route)["pass_threshold"] == 0.5  # still sent

    @respx.mock
    def test_no_warning_with_a_name(self, evals, caplog):
        respx.post(_EVALS).mock(return_value=httpx.Response(201))
        with caplog.at_level("WARNING", logger="Lucidic"):
            evals.emit(0.9, name="criterion_pass_rate", pass_threshold=0.5, session_id=_SESSION_ID)
        assert evals.flush(timeout=5) is True
        assert "User Supplied (Number)" not in caplog.text


class TestPassThresholdRefused:
    @respx.mock
    @pytest.mark.parametrize("result", [True, "excellent"])
    def test_only_with_a_number_result(self, evals, result):
        route = respx.post(_EVALS).mock(return_value=httpx.Response(201))
        with pytest.raises(ValueError, match="only applies to a number result"):
            evals.emit(result, name="m", pass_threshold=0.8, session_id=_SESSION_ID)
        assert not route.called

    @respx.mock
    @pytest.mark.parametrize("mark", [math.nan, math.inf, -math.inf, 10 ** 400, True, "0.8"])
    def test_must_be_a_finite_number(self, evals, mark):
        route = respx.post(_EVALS).mock(return_value=httpx.Response(201))
        with pytest.raises(ValueError, match="finite number"):
            evals.emit(0.85, name="m", pass_threshold=mark, session_id=_SESSION_ID)
        assert not route.called

    @respx.mock
    def test_production_logs_and_sends_nothing(self, http, caplog):
        evals = EvalsResource(http, production=True)
        route = respx.post(_EVALS).mock(return_value=httpx.Response(201))
        with caplog.at_level("ERROR", logger="Lucidic"):
            assert evals.emit(True, name="m", pass_threshold=0.8, session_id=_SESSION_ID) is None
        assert evals.flush(timeout=5) is True
        assert not route.called
        assert "only applies to a number result" in caplog.text

    @respx.mock
    def test_production_swallows_an_overflowing_mark(self, http):
        evals = EvalsResource(http, production=True)
        route = respx.post(_EVALS).mock(return_value=httpx.Response(201))
        assert evals.emit(0.85, name="m", pass_threshold=10 ** 400, session_id=_SESSION_ID) is None
        assert evals.flush(timeout=5) is True
        assert not route.called
