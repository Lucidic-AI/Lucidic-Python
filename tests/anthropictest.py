import os
import asyncio
import sys
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError("Missing ANTHROPIC_API_KEY")

# import Lucidic and the Anthropic SDK
import lucidicai as lai
from anthropic import Anthropic

async def run_test():
    # init Lucidic with your Anthropic handler
    lai.init(
        session_name="Anthropic Handler Test",
        provider="anthropic",
    )

    # open a step so `active_step` exists for event logging
    lai.create_step(
        state="Anthropic handler smoke-test",
        action="Call client.messages.create",
        goal="Verify events & responses"
    )

    # 5) instantiate the SDK client (it’ll pick up your patched handler)
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # --- Non-streaming call ---
    print("=== Non-streaming response ===")
    resp = client.messages.create(
        model="claude-3-7-sonnet-latest",
        messages=[{"role":"user","content":"Hi there! Summarize 'Romeo and Juliet' in one sentence."}],
        max_tokens=100
    )
    # extract text in the same way your handler does
    text = resp.content[0].text if hasattr(resp, "content") else str(resp)
    print(text)

    # --- Streaming call ---
    print("\n=== Streaming response ===")
    stream = client.messages.create(
        model="claude-3-5-sonnet-latest",
        messages=[{"role":"user","content":"Tell me a quick joke."}],
        max_tokens=100,
        stream=True
    )
    # print each delta as it arrives
    for chunk in stream:
        if hasattr(chunk, "delta") and getattr(chunk.delta, "text", None):
            print(chunk.delta.text, end="")
    print("\n")

    # == test image ==
    with open("tests/ord_runways.jpg", "rb") as f:
        img_bytes = f.read()
    data = base64.standard_b64encode(img_bytes).decode() # can also use url instead of bytes here

    # build the Anthropic message payload with an image_url piece
    image_message = {
        "role": "user",
        "content": [
            {"type": "text",      "text": "Here's the image I want you to describe:"},
            {"type": "image", "source": {"type": "base64", "media_type":"image/jpeg", "data": data}}
        ]
    }

    resp = client.messages.create(
        model="claude-3-5-sonnet-latest",
        messages=[image_message],
        max_tokens=50
    )

    # your handler should have torn off that data_uri into `screenshots=[…]`
    print("Response text:", resp.content[0].text)

    # tear down
    lai.end_step()
    lai.end_session()


if __name__ == "__main__":
    asyncio.run(run_test())
