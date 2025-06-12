import os
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

def run_test():
    # init Lucidic with OpenAI handler (this will be used even with Anthropic base URL)
    lai.init(
        session_name="OpenAI SDK + Anthropic Base URL Simple Test",
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

    print("=== Test 1: Regular chat completion ===")
    try:
        resp = client.chat.completions.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role":"user","content":"What is 25 * 4 + 10? Show your work."}],
            max_tokens=100
        )
        
        result = resp.choices[0].message.content if hasattr(resp, "choices") else str(resp)
        print("✓ Regular chat completion works")
        print("Response:", result[:50] + "..." if len(result) > 50 else result)
        print("Usage:", resp.usage if hasattr(resp, 'usage') else "No usage info")
    except Exception as e:
        print(f"✗ Regular chat completion failed: {type(e).__name__}: {str(e)}")

    print("\n=== Test 2: Streaming response (limited) ===")
    try:
        stream_resp = client.chat.completions.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role":"user","content":"Count from 1 to 3"}],
            max_tokens=20,
            stream=True
        )
        
        print("Stream response: ", end="")
        chunk_count = 0
        for chunk in stream_resp:
            if chunk_count > 10:  # Limit chunks to prevent hanging
                print("\n(truncated after 10 chunks)")
                break
            
            if (hasattr(chunk, 'choices') and chunk.choices is not None and 
                len(chunk.choices) > 0):
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    print(delta.content, end='', flush=True)
            chunk_count += 1
        print("\n✓ Streaming works (with null checks)")
    except Exception as e:
        print(f"✗ Streaming failed: {type(e).__name__}: {str(e)}")

    print("\n=== Test 3: Structured output (parse) - Should now work! ===")
    try:
        resp = client.beta.chat.completions.parse(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role":"user","content":"Solve step by step: 25 * 4 + 10"}],
            response_format=MathReasoning,
        )
        result = resp.choices[0].message.parsed if hasattr(resp, "choices") else str(resp)
        print("✓ Structured output works with Anthropic workaround!")
        print("Final answer:", result.final_answer)
        print("Steps:")
        for step in result.steps:
            print(f"  - {step.explanation}: {step.output}")
    except Exception as parse_error:
        print(f"✗ Structured output failed: {type(parse_error).__name__}: {str(parse_error)}")

    print("\n=== Test 4: Image analysis ===")
    try:
        with open("tests/ord_runways.jpg", "rb") as f:
            img_bytes = f.read()
        data_uri = f"data:image/jpeg;base64,{base64.standard_b64encode(img_bytes).decode()}"

        # OpenAI format message with image
        image_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see? One word answer."},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }

        resp = client.chat.completions.create(
            model="claude-3-5-sonnet-20241022",
            messages=[image_message],
            max_tokens=50
        )

        result = resp.choices[0].message.content
        print("✓ Image analysis works")
        print("Description:", result[:50] + "..." if len(result) > 50 else result)
        print("Usage:", resp.usage if hasattr(resp, 'usage') else "No usage info")
    except Exception as e:
        print(f"✗ Image analysis failed: {type(e).__name__}: {str(e)}")

    # tear down
    lai.end_step()
    lai.end_session()
    print("\n=== Test completed ===")


if __name__ == "__main__":
    run_test()