"""Minimal streaming test"""
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set up logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Also set Lucidic logger to DEBUG
lucidic_logger = logging.getLogger("Lucidic")
lucidic_logger.setLevel(logging.DEBUG)

from dotenv import load_dotenv
load_dotenv()

import lucidicai as lai
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize
lai.init(session_name="Minimal Stream Test", providers=["openai"])
lai.create_step(state="Test", action="Stream", goal="Debug")

# Create client and stream
client = OpenAI(api_key=OPENAI_API_KEY)
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi"}],
    stream=True,
    max_tokens=5
)

# Try to iterate
print(f"Stream object type: {type(stream)}")
print(f"Stream object class: {stream.__class__.__name__}")

for chunk in stream:
    if hasattr(chunk, 'choices') and chunk.choices and chunk.choices[0].delta.content:
        print(f"Chunk content: {chunk.choices[0].delta.content}")

lai.end_step()
lai.end_session()