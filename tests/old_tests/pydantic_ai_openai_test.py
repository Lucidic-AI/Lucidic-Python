import os
import asyncio
import sys
from typing import List, Optional
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

# import Lucidic and PydanticAI
import lucidicai as lai
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from openai import OpenAI


class StreamResponse(BaseModel):
    """StreamResponse class as mentioned by user"""
    agent_name: str
    instructions: str
    steps: List[str]
    output: str
    status_code: int


def get_client():
    """Get OpenAI client"""
    return OpenAI(api_key=OPENAI_API_KEY)


orchestrator_system_prompt = """You are an expert project orchestrator. 
Your job is to break down complex tasks into clear, actionable steps and provide detailed output.
Always be thorough and structured in your responses."""

orchestrator_deps = None  # No dependencies for this simple test


async def run_test():
    # init Lucidic with PydanticAI handler
    lai.init(
        session_name="PydanticAI OpenAI Handler Test",
        providers=["pydantic_ai"],
    )

    # open a step so `active_step` exists for event logging
    lai.create_step(
        state="PydanticAI OpenAI handler smoke-test",
        action="Call agent.run with OpenAIModel",
        goal="Verify events & responses with PydanticAI"
    )

    # Create the OpenAIModel as specified by user
    model = OpenAIModel(
        model_name="gpt-4o-mini"
    )

    # Create the orchestrator agent (model tracking is handled by monkey-patching)
    orchestrator_agent = Agent(
        model=model,
        name="Orchestrator Agent",
        system_prompt=orchestrator_system_prompt,
        deps_type=orchestrator_deps
    )

    # --- Non-streaming call ---
    print("=== Non-streaming PydanticAI response ===")
    task = "Summarize 'Romeo and Juliet' in one sentence."
    
    # Create StreamResponse as mentioned by user
    planner_stream_output = StreamResponse(
        agent_name="Planner Agent",
        instructions=task,
        steps=[],
        output="",
        status_code=0
    )
    
    # Make the call as suggested by user
    planner_response = await orchestrator_agent.run(user_prompt=task)
    print("Response:", planner_response.output)
    
    # Check the last event for cost information
    event_history = lai.Client().session.event_history
    last_event_id = list(event_history.keys())[-1] if event_history else None
    last_event = event_history[last_event_id] if last_event_id else None
    if last_event:
        print(f"Non-streaming cost: {getattr(last_event, 'cost_added', 'N/A')}")
    else:
        print("No event found for non-streaming")

    # --- Streaming call ---
    print("\n=== Streaming PydanticAI response ===")
    task2 = "Tell me a quick joke."
    
    async with orchestrator_agent.run_stream(user_prompt=task2) as stream_result:
        async for chunk in stream_result.stream_text(delta=True):
            print(chunk, end="")
        print("\n")
    
    # Check the last event for streaming cost information
    event_history = lai.Client().session.event_history
    last_event_id = list(event_history.keys())[-1] if event_history else None
    last_event = event_history[last_event_id] if last_event_id else None
    if last_event:
        print(f"Streaming cost: {getattr(last_event, 'cost_added', 'N/A')}")
    else:
        print("No event found for streaming")

    # tear down
    lai.end_step()
    lai.end_session()


if __name__ == "__main__":
    asyncio.run(run_test())