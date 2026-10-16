"""The Lucidic wrapper an FDE copies next to the customer's agent.

Webhook runs: `lucidic.serve(run_case)` (see examples/lucidic_runner.py) calls run_case per case.
Local or subprocess runs: `python lucidic_entry.py --lucidic-run case.json`
(or LUCIDIC_RUN set to the JSON or a path to it).
"""
import lucidicai as lucidic
from their_pkg import their_agent                       # the only line the customer edits

client = lucidic.LucidicAI(providers=["openai"])        # api key / agent id from env


def run_case(run: lucidic.Run):
    with run:                                           # session open, fixtures bound; ends + flushes evals on exit
        run.download("workdir")                         # the case documents, at workdir/<path>
        return their_agent(
            run.input,
            system=run.prompt("system", default=their_agent.SYSTEM_PROMPT, register=True),
            extra_tools=run.tools.list(),
            call_extra_tool=run.tools.call,
        )


if __name__ == "__main__":
    run_case(lucidic.load(client=client))               # LUCIDIC_RUN, else --lucidic-run
