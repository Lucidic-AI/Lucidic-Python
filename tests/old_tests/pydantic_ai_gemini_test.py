import os
import asyncio
import sys
from typing import List, Optional
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY")

# import Lucidic and PydanticAI
import lucidicai as lai
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel


class StreamResponse(BaseModel):
    """StreamResponse class as mentioned by user"""
    agent_name: str
    instructions: str
    steps: List[str]
    output: str
    status_code: int


orchestrator_system_prompt = """You are an expert project orchestrator. 
Your job is to break down complex tasks into clear, actionable steps and provide detailed output.
Always be thorough and structured in your responses."""

orchestrator_deps = None  # No dependencies for this simple test


async def run_test():
    # init Lucidic with PydanticAI handler
    lai.init(
        session_name="PydanticAI Gemini Handler Test",
        providers=["pydantic_ai"],
    )

    # open a step so `active_step` exists for event logging
    lai.create_step(
        state="PydanticAI Gemini handler smoke-test",
        action="Call agent.run with GeminiModel",
        goal="Verify events & responses with PydanticAI"
    )

    # Create the GeminiModel as specified by user
    model = GeminiModel(
        model_name="gemini-2.5-flash-preview-05-20"
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

    # --- Streaming call ---
    print("\n=== Streaming PydanticAI response ===")
    task2 = "Tell me a quick joke."
    
    async with orchestrator_agent.run_stream(user_prompt=task2) as stream_result:
        async for chunk in stream_result.stream_text(delta=True):
            print(chunk, end="")
        print("\n")

    # tear down
    lai.end_step()
    lai.end_session()


if __name__ == "__main__":
    asyncio.run(run_test())