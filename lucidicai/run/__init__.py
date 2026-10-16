"""One object per run: ``lucidicai.load()`` connects any agent to a Lucidic eval run.

The wrapper an FDE copies next to the customer's agent is examples/lucidic_entry.py:
``with run:`` around their agent, with ``run.prompt(...)`` and ``run.tools.list()``.

Webhook runs: ``lucidicai.serve(run_case)`` (``lucidicai/runner.py``) verifies and
dedupes each delivery, replies 202 and calls ``run_case`` on a bounded pool.
A subprocess gets the same run through ``run.to_env()``.
"""
from .load import RUN_ARG, load
from .run import RUN_ENV_VAR, Run, RunIds
from .tools import RunTools

__all__ = ["load", "Run", "RunIds", "RunTools", "RUN_ENV_VAR", "RUN_ARG"]
