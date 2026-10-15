"""One object per run: ``lucidicai.load()`` connects any agent to a Lucidic eval run.

The wrapper an FDE copies next to the customer's agent is examples/lucidic_entry.py:
``with run:`` around their agent, with ``run.prompt(...)`` and ``run.tools.list()``.

A webhook receiver calls ``load(body, signature, client=client)`` in the request
(it does no network I/O), replies 202 and hands the run to ``run_case`` on a
bounded pool; see examples/lucidic_runner.py.
A subprocess gets the same run through ``run.to_env()``.
"""
from .load import RUN_ARG, load
from .run import RUN_ENV_VAR, Run, RunIds
from .tools import RunTools

__all__ = ["load", "Run", "RunIds", "RunTools", "RUN_ENV_VAR", "RUN_ARG"]
