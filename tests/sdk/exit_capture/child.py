"""Subprocess under test: ``child.py <scenario>``.

Builds a real client, makes one real openai call against the fake backend
(instrumentation -> span -> exporter), then ends the way the scenario says.
Signal scenarios print READY and block until the parent signals them.
"""
import asyncio
import os
import signal
import sys
import threading
import time

import openai

from lucidicai import LucidicAI

AGENT_ID = "aaaa1111-0000-0000-0000-000000000000"
API_KEY = "secret-key"
MESSAGES = [{"role": "user", "content": "hi"}]
BATCH_TIMER_WAIT_SECONDS = 7  # past the 5 s span batch timer


def new_client():
    return LucidicAI(api_key=API_KEY, agent_id=AGENT_ID, providers=["openai"])


def llm():
    oa = openai.OpenAI(api_key="sk-test", base_url=os.environ["OPENAI_BASE_URL"])
    oa.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES)


async def allm():
    oa = openai.AsyncOpenAI(api_key="sk-test", base_url=os.environ["OPENAI_BASE_URL"])
    await oa.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES)


def block(message="READY"):
    print(message, flush=True)
    while True:
        time.sleep(0.1)


def app_handler(signum, frame):
    print("APP_HANDLER", flush=True)
    sys.exit(7)


def session_then_llm(client, name):
    session = client.sessions.create(session_name=name)
    llm()
    return session


def in_thread(name):
    errors = []

    def work():
        try:
            session_then_llm(new_client(), name)
        except Exception as e:
            errors.append(repr(e))

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    print(f"THREAD_ERRORS={errors}", flush=True)


async def async_scenario(name):
    client = new_client()
    session = await client.sessions.acreate(session_name=name)
    await allm()
    if name == "async_aend":
        await session.aend()
    else:
        await client.aclose()
        os._exit(0)  # skip atexit: aclose alone must have delivered everything


def drain_then_second_signal(client):
    from lucidicai.runner import _drain_on_signal

    session_then_llm(client, "serve")
    with _drain_on_signal(lambda: None) as stopped:
        print("READY", flush=True)
        stopped.wait()
        block("DRAINING")  # a case that never finishes


def main(name):
    if name in ("thread_client", "thread_client_sigterm"):
        in_thread(name)
        if name == "thread_client_sigterm":
            block()
        return
    if name.startswith("async_"):
        asyncio.run(async_scenario(name))
        return
    if name == "app_handler_before":
        signal.signal(signal.SIGTERM, app_handler)
    client = new_client()
    if name == "app_handler_after":
        signal.signal(signal.SIGTERM, app_handler)
    if name == "serve_second_signal":
        drain_then_second_signal(client)
    if name == "with":
        with client.sessions.create(session_name=name):
            llm()
        return
    session = session_then_llm(client, name)
    if name == "manual_end":
        session.end()
    elif name == "manual_end_linger":
        session.end()
        time.sleep(BATCH_TIMER_WAIT_SECONDS)
    elif name == "sessions_end":
        client.sessions.end(session.session_id)
    elif name == "close":
        client.close()
        os._exit(0)  # skip atexit: close alone must have delivered everything
    elif name in ("sigint", "sigterm", "app_handler_before", "app_handler_after"):
        block()
    # exit, last_event_exit: fall off the end


if __name__ == "__main__":
    main(sys.argv[1])
