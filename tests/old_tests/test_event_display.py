"""Test to show what events look like before stream consumption"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import lucidicai as lai
from openai import OpenAI
from lucidicai.client import Client

# Initialize
lai.init('Event Display Test', providers=['openai'])

# Create a step
step_id = lai.create_step(state="Testing", action="Multiple calls", goal="Show events")

client = OpenAI()

# Make several calls including streaming
print("Making API calls...\n")

# 1. Regular call
response1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say 'test 1'"}],
    max_tokens=10
)

# 2. Another regular call
response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say 'test 2'"}],
    max_tokens=10
)

# 3. Streaming call - DON'T CONSUME YET
stream3 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Count to 3"}],
    stream=True,
    max_tokens=20
)

# 4. Another streaming call - DON'T CONSUME YET
stream4 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True,
    max_tokens=30
)

# 5. Another streaming call - DON'T CONSUME YET
stream5 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "List A, B, C"}],
    stream=True,
    max_tokens=20
)

# Now let's check the events BEFORE consuming streams
print("=== EVENTS BEFORE CONSUMING STREAMS ===")
session = lai.get_session()

# Get event data from the backend
client_obj = Client()
for i, (event_id, event) in enumerate(session.event_history.items()):
    print(f"\nEvent {i+1} ({event_id[:8]}...):")
    try:
        # Try to get event data from backend
        event_data = client_obj.make_request('getevent', 'GET', {'event_id': event_id})
        print(f"  Description: {event_data.get('description', 'N/A')[:50]}...")
        result = event_data.get('result', 'N/A')
        result_str = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
        print(f"  Result: {result_str}")
        print(f"  Finished: {event_data.get('is_finished', False)}")
    except Exception as e:
        # Fallback to local data
        print(f"  Could not get from backend: {e}")
        print(f"  Local finished state: {event.is_finished}")

# This is what the test output shows - unconsumed streams
print(f"\n=== STREAM OBJECTS (what tests might print) ===")
print(f"Stream 3: {stream3}")
print(f"Stream 4: {stream4}")
print(f"Stream 5: {stream5}")

# Now consume the streams
print("\n=== CONSUMING STREAMS ===")
for i, stream in enumerate([stream3, stream4, stream5], 3):
    print(f"\nConsuming stream {i}...")
    response = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
    print(f"Response: {response}")

# Check events AFTER consuming
print("\n=== EVENTS AFTER CONSUMING STREAMS ===")
for i, (event_id, event) in enumerate(session.event_history.items()):
    print(f"\nEvent {i+1} ({event_id[:8]}...):")
    print(f"  Finished: {event.is_finished}")

lai.end_session()