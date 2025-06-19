"""Test Anthropic streaming finalization"""
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Enable info logging
logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv
load_dotenv()

import lucidicai as lai
from openai import OpenAI

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

print("=== Testing Anthropic Streaming Finalization ===")

lai.init(
    session_name="Anthropic Stream Fix Test",
    providers=["openai"]
)

lai.create_step(
    state="Testing Anthropic streaming",
    action="Stream from Anthropic",
    goal="Check event finalization"
)

# Use OpenAI SDK with Anthropic base URL
client = OpenAI(
    api_key=ANTHROPIC_API_KEY,
    base_url="https://api.anthropic.com/v1",
    default_headers={
        "anthropic-version": "2023-06-01"
    }
)

print("\nCreating Anthropic streaming request...")
stream = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Count from 1 to 3"}],
    stream=True,
    max_tokens=20
)

print("\nIterating over stream...")
full_response = ""
chunk_count = 0

for chunk in stream:
    chunk_count += 1
    print(f"\nChunk {chunk_count}: {chunk}")
    if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'content') and delta.content:
            content = delta.content
            full_response += content
            print(f"  Content: '{content}'")

print(f"\nTotal chunks: {chunk_count}")
print(f"Full response: '{full_response}'")

lai.end_step()
lai.end_session()
print("\nTest completed")