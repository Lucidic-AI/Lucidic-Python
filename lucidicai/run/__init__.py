"""One object per run: ``lucidicai.load()`` connects any agent to a Lucidic eval run.

The wrapper an FDE writes next to the customer's agent (examples/lucidic_entry.py)::

    import lucidicai as lucidic
    from their_pkg import their_agent                       # the only line the customer edits

    client = lucidic.LucidicAI(providers=["openai"])        # api key / agent id from env

    def run_case(run: lucidic.Run):
        with run:                                           # session open, fixtures bound; ends + flushes evals on exit
            # todo(LUC-1141): run.download(workdir) for the case documents
            return their_agent(
                run.input,
                system=run.prompt("system", default=their_agent.SYSTEM_PROMPT, register=True),
                extra_tools=run.tools.list(),
                call_extra_tool=run.tools.call,
            )

    if __name__ == "__main__":
        run_case(lucidic.load(client=client))               # LUCIDIC_RUN, else --lucidic-run

A webhook receiver calls ``load(body, signature, client=client)`` in the request
(it does no network I/O), replies 202 and hands the run to ``run_case`` on a
bounded pool; see examples/lucidic_runner.py.
A subprocess gets the same run through ``run.to_env()``.
"""
from .load import RUN_ARG, load
from .run import RUN_ENV_VAR, Run, RunIds
from .tools import RunTools

__all__ = ["load", "Run", "RunIds", "RunTools", "RUN_ENV_VAR", "RUN_ARG"]
