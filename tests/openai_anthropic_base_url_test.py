import os
import asyncio
import sys
import base64
from pydantic import BaseModel
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError("Missing ANTHROPIC_API_KEY")

# import Lucidic and the OpenAI SDK
import lucidicai as lai
from openai import OpenAI

# Define Pydantic models for structured output
class MathStep(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: List[MathStep]
    final_answer: str

class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str
    skills: List[str]

class ImageDescription(BaseModel):
    description: str
    objects_seen: List[str]

async def run_test():
    # init Lucidic with OpenAI handler (this will be used even with Anthropic base URL)
    lai.init(
        session_name="OpenAI SDK + Anthropic Base URL Test",
        provider="openai",  # Using OpenAI handler with Anthropic API
    )

    # open a step so `active_step` exists for event logging
    lai.create_step(
        state="OpenAI SDK with Anthropic base URL test",
        action="Call OpenAI SDK with Anthropic base URL",
        goal="Test OpenAI handler compatibility with Anthropic API"
    )

    # instantiate OpenAI SDK client with Anthropic base URL
    client = OpenAI(
        api_key=ANTHROPIC_API_KEY,
        base_url="https://api.anthropic.com/v1",
        default_headers={
            "anthropic-version": "2023-06-01"
        }
    )

    # --- Regular chat completion call ---
    print("=== Regular chat completion ===")
    resp = client.chat.completions.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role":"user","content":"What is 25 * 4 + 10? Show your work."}],
        max_tokens=150
    )
    
    # This should work but may have issues with the Lucidic handler
    result = resp.choices[0].message.content if hasattr(resp, "choices") else str(resp)
    print("Response:", result)
    print("Usage:", resp.usage if hasattr(resp, 'usage') else "No usage info")

    # --- Test streaming response ---
    print("\n=== Streaming response test ===")
    stream_resp = client.chat.completions.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role":"user","content":"Count from 1 to 5"}],
        max_tokens=50,
        stream=True
    )
    
    print("Stream response:")
    chunk_count = 0
    for chunk in stream_resp:
        if chunk_count > 10:  # Prevent hanging
            print("\n(truncated)")
            break
            
        if (hasattr(chunk, 'choices') and chunk.choices is not None and 
            len(chunk.choices) > 0):
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                print(delta.content, end='', flush=True)
        chunk_count += 1
    print("\n")

    # --- Test structured output (parse) - this will likely fail ---
    print("\n=== Structured output (parse) test ===")
    try:
        resp = client.beta.chat.completions.parse(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role":"user","content":"Solve step by step: What is 25 * 4 + 10?"}],
            response_format=MathReasoning,
        )
        result = resp.choices[0].message.parsed if hasattr(resp, "choices") else str(resp)
        print("Final answer:", result.final_answer)
        print("Steps:")
        for step in result.steps:
            print(f"  - {step.explanation}: {step.output}")
    except Exception as parse_error:
        print(f"PARSE ERROR: {type(parse_error).__name__}: {str(parse_error)}")
        print("This demonstrates that structured output doesn't work with Anthropic base URL")

    # --- Test with image (if supported) ---
    print("\n=== Image analysis test ===")
    with open("tests/ord_runways.jpg", "rb") as f:
        img_bytes = f.read()
    data_uri = f"data:image/jpeg;base64,{base64.standard_b64encode(img_bytes).decode()}"

    # OpenAI format message with image - this may cause issues with Anthropic API
    image_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image briefly:"},
            {"type": "image_url", "image_url": {"url": data_uri}}
        ]
    }

    resp = client.chat.completions.create(
        model="claude-3-5-sonnet-20241022",
        messages=[image_message],
        max_tokens=100
    )

    result = resp.choices[0].message.content
    print("Image description:", result)
    print("Usage:", resp.usage if hasattr(resp, 'usage') else "No usage info")

    # tear down
    lai.end_step()
    lai.end_session()


if __name__ == "__main__":
    asyncio.run(run_test())