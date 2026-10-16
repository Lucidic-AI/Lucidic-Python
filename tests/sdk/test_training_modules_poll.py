"""Characterization tests for TrainingModulesResource's inference poll loop.

These pin the observable behavior of ``_poll_to_tool_result`` /
``_apoll_to_tool_result`` (terminal mapping, SDK timeout, malformed-response and
poll-error soft failures, poll progression) so the LUC-920 refactor onto the
generic ``wait_for`` primitive provably changes no behavior.
"""
from unittest.mock import MagicMock

import pytest

from lucidicai.sdk.training_modules.resource import TrainingModulesResource


@pytest.fixture
def tm():
    # A MagicMock client is enough — we drive _api directly per test.
    resource = TrainingModulesResource(client=MagicMock())
    resource._api = MagicMock()
    return resource


def _run(status, **over):
    d = {"status": status, "inference_run_id": "ir1", "tool_name": "t", "session_id": "sess"}
    d.update(over)
    return d


def _poll(tm, run, *, timeout=5.0, interval=0.001):
    return tm._poll_to_tool_result(
        run, session_id="sess", timeout_seconds=timeout, poll_interval_seconds=interval)


class TestTerminalMapping:
    def test_initial_succeeded_unwraps_return_value(self, tm):
        out = _poll(tm, _run("SUCCEEDED", result={"return_value": 42}))
        assert out == 42
        tm._api.get_inference.assert_not_called()  # already terminal, no poll

    def test_succeeded_without_return_value_returns_result(self, tm):
        out = _poll(tm, _run("SUCCEEDED", result={"other": "x"}))
        assert out == {"other": "x"}

    def test_failed_is_soft_failure(self, tm):
        out = _poll(tm, _run("FAILED", error={"code": "x", "message": "bad"}))
        assert out["error"]["message"] == "bad"

    def test_timed_out_maps_to_code(self, tm):
        out = _poll(tm, _run("TIMED_OUT"))
        assert out["error"]["code"] == "training_module_inference_timed_out"

    def test_cancelled_maps_to_code(self, tm):
        out = _poll(tm, _run("CANCELLED"))
        assert out["error"]["code"] == "training_module_inference_cancelled"

    def test_unknown_terminal_status_maps_to_code(self, tm):
        out = _poll(tm, _run("WEIRD"))
        assert out["error"]["code"] == "training_module_inference_unknown_status"


class TestPollProgression:
    def test_polls_running_until_succeeded(self, tm):
        tm._api.get_inference.side_effect = [
            _run("RUNNING"), _run("SUCCEEDED", result={"return_value": "ok"})]
        out = _poll(tm, _run("RUNNING"))
        assert out == "ok"
        assert tm._api.get_inference.call_count == 2

    def test_poll_passes_inference_run_id_and_session(self, tm):
        tm._api.get_inference.side_effect = [_run("SUCCEEDED", result={"return_value": 1})]
        _poll(tm, _run("RUNNING", inference_run_id="ir-42"))
        tm._api.get_inference.assert_called_once_with(inference_run_id="ir-42", session_id="sess")


class TestFailurePaths:
    def test_sdk_timeout_before_first_poll(self, tm):
        out = _poll(tm, _run("RUNNING"), timeout=0)
        assert out["error"]["code"] == "training_module_sdk_timeout"
        tm._api.get_inference.assert_not_called()

    def test_missing_inference_run_id_is_malformed(self, tm):
        out = _poll(tm, _run("RUNNING", inference_run_id=None))
        assert out["error"]["code"] == "training_module_malformed_response"
        tm._api.get_inference.assert_not_called()  # never fetches without an id

    def test_poll_exception_is_soft_failure(self, tm):
        tm._api.get_inference.side_effect = RuntimeError("boom")
        out = _poll(tm, _run("RUNNING"))
        assert out["error"]["code"] == "training_module_poll_failed"
        assert "boom" in out["error"]["message"]


class TestAsyncParity:
    @pytest.mark.asyncio
    async def test_apoll_running_until_succeeded(self, tm):
        async def _aget(*, inference_run_id, session_id):
            return _run("SUCCEEDED", result={"return_value": "async-ok"})
        tm._api.aget_inference = _aget
        out = await tm._apoll_to_tool_result(
            _run("RUNNING"), session_id="sess", timeout_seconds=5.0, poll_interval_seconds=0.001)
        assert out == "async-ok"

    @pytest.mark.asyncio
    async def test_apoll_sdk_timeout(self, tm):
        out = await tm._apoll_to_tool_result(
            _run("RUNNING"), session_id="sess", timeout_seconds=0, poll_interval_seconds=0.001)
        assert out["error"]["code"] == "training_module_sdk_timeout"
