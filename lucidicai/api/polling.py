"""Generic trigger-then-poll ``wait_for`` primitive (LUC-920).

The reusable poll loop behind the SDK's async, workflow-backed resources (dataset
generation, taxonomy / failure-modes, evosim runs, training-module inference). A
resource triggers a backend workflow, then blocks on ``wait_for`` — polling a
status callable on a fixed interval until it reports a terminal state or a deadline
elapses.

This owns only the loop mechanics (deadline, interval, optional pre-fetched initial
state, final-sleep clamping). Terminal-state *interpretation* — which states are
terminal, and whether a terminal state means success or failure — stays with the
caller via ``is_terminal`` and its handling of the returned state.
"""
import asyncio
import time
from typing import Any, Awaitable, Callable, TypeVar

from ..core.errors import WaitTimeout

T = TypeVar("T")

# Distinguishes "no initial state supplied" from a legitimately falsy/None state.
_UNSET: Any = object()


def wait_for(
    poll: Callable[[], T],
    *,
    is_terminal: Callable[[T], bool],
    timeout: float,
    interval: float,
    initial: Any = _UNSET,
) -> T:
    """Poll ``poll()`` until ``is_terminal(state)`` is true or the ``timeout``
    deadline elapses.

    Returns the first terminal state. Raises :class:`WaitTimeout` (carrying the
    last polled state) if the deadline elapses first. ``poll`` is called once up
    front unless an ``initial`` state is supplied (e.g. a trigger response already
    carries the first status), then once per ``interval`` thereafter. The final
    sleep is clamped so polling never overshoots the deadline.

    Args:
        poll: Zero-arg callable returning the current state.
        is_terminal: Predicate — true when a state should stop the loop.
        timeout: Total budget in seconds (clamped to >= 0).
        interval: Seconds to sleep between polls (clamped to >= 0). Note: ``0``
            busy-polls with no delay between calls — pass a real interval (e.g.
            a couple of seconds) for anything backed by a network status request.
        initial: A state already in hand; when given, ``poll`` is not called until
            after the first interval.
    """
    budget = max(0.0, timeout)
    deadline = time.monotonic() + budget
    state: T = poll() if initial is _UNSET else initial
    while not is_terminal(state):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WaitTimeout(state, budget)
        time.sleep(min(max(0.0, interval), remaining))
        state = poll()
    return state


async def await_for(
    poll: Callable[[], Awaitable[T]],
    *,
    is_terminal: Callable[[T], bool],
    timeout: float,
    interval: float,
    initial: Any = _UNSET,
) -> T:
    """Async sibling of :func:`wait_for` — ``poll`` is an ``async`` callable and the
    inter-poll wait uses ``asyncio.sleep``. Same semantics otherwise (including the
    ``interval=0`` busy-poll caveat)."""
    budget = max(0.0, timeout)
    deadline = time.monotonic() + budget
    state: T = (await poll()) if initial is _UNSET else initial
    while not is_terminal(state):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WaitTimeout(state, budget)
        await asyncio.sleep(min(max(0.0, interval), remaining))
        state = await poll()
    return state
