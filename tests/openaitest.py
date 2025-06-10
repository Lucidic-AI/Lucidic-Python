import os
import asyncio
import sys
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

# import Lucidic and the OpenAI SDK
import lucidicai as lai
from openai import OpenAI

async def run_test():
    # init Lucidic with your OpenAI handler
    lai.init(
        session_name="OpenAI Handler Test",
        provider="openai",
    )

    # open a step so `active_step` exists for event logging
    lai.create_step(
        state="OpenAI handler smoke-test",
        action="Call client.chat.completions.create",
        goal="Verify events & responses"
    )

    # instantiate the SDK client (it'll pick up your patched handler)
    client = OpenAI(api_key=OPENAI_API_KEY)

    # --- Non-streaming call ---
    print("=== Non-streaming response ===")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":"Hi there! Summarize 'Romeo and Juliet' in one sentence."}],
        max_tokens=100
    )
    # extract text in the same way your handler does
    text = resp.choices[0].message.content if hasattr(resp, "choices") else str(resp)
    print(text)

    # --- Streaming call ---
    print("\n=== Streaming response ===")
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":"Tell me a quick joke."}],
        max_tokens=100,
        stream=True
    )
    # print each delta as it arrives
    for chunk in stream:
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                print(delta.content, end="")
    print("\n")

    # == test image ==
    with open("tests/ord_runways.jpg", "rb") as f:
        img_bytes = f.read()
    data_uri = f"data:image/jpeg;base64,{base64.standard_b64encode(img_bytes).decode()}"

    # build the OpenAI message payload with an image_url piece
    image_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Here's the image I want you to describe:"},
            {"type": "image_url", "image_url": {"url": data_uri}}
        ]
    }

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[image_message],
        max_tokens=50
    )

    # your handler should have torn off that data_uri into `screenshots=[…]`
    print("Response text:", resp.choices[0].message.content)

    # tear down
    lai.end_step()
    lai.end_session()


if __name__ == "__main__":
    asyncio.run(run_test())