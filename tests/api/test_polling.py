"""LUC-920 — the generic wait_for / await_for poll primitive."""
import pytest

from lucidicai.api.polling import await_for, wait_for
from lucidicai.core.errors import LucidicError, WaitTimeout


def _terminal(state):
    return state == "done"


class TestWaitForSync:
    def test_returns_initial_without_polling_when_already_terminal(self):
        def poll():
            raise AssertionError("poll must not run when initial is terminal")
        assert wait_for(poll, is_terminal=_terminal, timeout=5, interval=0,
                        initial="done") == "done"

    def test_polls_until_terminal(self):
        states = iter(["running", "running", "done"])
        calls = {"n": 0}

        def poll():
            calls["n"] += 1
            return next(states)

        # no initial -> poll() supplies the first state too
        assert wait_for(poll, is_terminal=_terminal, timeout=5, interval=0) == "done"
        assert calls["n"] == 3

    def test_initial_supplied_skips_first_poll(self):
        states = iter(["done"])
        calls = {"n": 0}

        def poll():
            calls["n"] += 1
            return next(states)

        # initial is non-terminal, so exactly one poll is needed to reach "done"
        assert wait_for(poll, is_terminal=_terminal, timeout=5, interval=0,
                        initial="running") == "done"
        assert calls["n"] == 1

    def test_timeout_raises_with_last_state_and_budget(self):
        def poll():
            return "running"  # never terminal

        with pytest.raises(WaitTimeout) as exc:
            wait_for(poll, is_terminal=_terminal, timeout=0, interval=0, initial="running")
        assert exc.value.last_state == "running"
        assert exc.value.timeout == 0
        assert isinstance(exc.value, LucidicError)  # catchable as a LucidicError

    def test_timeout_last_state_is_most_recent_poll(self):
        # no initial: the first poll runs, then the deadline (0) trips
        def poll():
            return "step-1"

        with pytest.raises(WaitTimeout) as exc:
            wait_for(poll, is_terminal=_terminal, timeout=0, interval=0)
        assert exc.value.last_state == "step-1"

    def test_negative_timeout_clamps_reported_budget_to_zero(self):
        # a misused negative timeout raises immediately AND reports a sane budget
        # (not "within -5s") — the deadline is clamped, so is the WaitTimeout.
        def poll():
            return "running"
        with pytest.raises(WaitTimeout) as exc:
            wait_for(poll, is_terminal=_terminal, timeout=-5, interval=0, initial="running")
        assert exc.value.timeout == 0
        assert "-5" not in str(exc.value)

    def test_falsy_initial_state_is_honored_not_treated_as_unset(self):
        # initial=None must NOT trigger a first poll (None != the _UNSET sentinel)
        def poll():
            raise AssertionError("poll must not run for a terminal None initial")
        assert wait_for(poll, is_terminal=lambda s: s is None, timeout=5, interval=0,
                        initial=None) is None


class TestAwaitForAsync:
    @pytest.mark.asyncio
    async def test_returns_initial_when_terminal(self):
        async def poll():
            raise AssertionError("poll must not run when initial is terminal")
        assert await await_for(poll, is_terminal=_terminal, timeout=5, interval=0,
                               initial="done") == "done"

    @pytest.mark.asyncio
    async def test_polls_until_terminal(self):
        states = iter(["running", "done"])
        calls = {"n": 0}

        async def poll():
            calls["n"] += 1
            return next(states)

        assert await await_for(poll, is_terminal=_terminal, timeout=5, interval=0) == "done"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        async def poll():
            return "running"

        with pytest.raises(WaitTimeout) as exc:
            await await_for(poll, is_terminal=_terminal, timeout=0, interval=0, initial="running")
        assert exc.value.last_state == "running"
