"""Debug streaming test"""
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

import lucidicai as lai
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def test_simple_streaming():
    print("\n=== Simple Streaming Debug Test ===")
    
    lai.init(
        session_name="Debug Streaming Test",
        providers=["openai"]
    )
    
    lai.create_step(
        state="Testing streaming",
        action="Debug stream",
        goal="See what happens"
    )
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    print("\nCreating streaming request...")
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Hello World'"}],
        stream=True,
        max_tokens=10
    )
    
    print("\nIterating over stream...")
    full_response = ""
    for i, chunk in enumerate(stream):
        print(f"Chunk {i}: {chunk}")
        if hasattr(chunk, 'choices') and chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            print(f"  Content: '{content}'")
    
    print(f"\nFull response: '{full_response}'")
    
    lai.end_step()
    lai.end_session()
    print("\nTest completed")

if __name__ == "__main__":
    test_simple_streaming()