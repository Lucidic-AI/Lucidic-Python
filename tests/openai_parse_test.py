import os
import asyncio
import sys
import base64
from pydantic import BaseModel
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

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

async def run_test():
    # init Lucidic with your OpenAI handler
    lai.init(
        session_name="OpenAI Parse Handler Test",
        provider="openai",
    )

    # open a step so `active_step` exists for event logging
    lai.create_step(
        state="OpenAI parse handler smoke-test",
        action="Call client.beta.chat.completions.parse",
        goal="Verify structured output events & responses"
    )

    # instantiate the SDK client (it'll pick up your patched handler)
    client = OpenAI(api_key=OPENAI_API_KEY)

    # --- Structured output call ---
    print("=== Structured output response ===")
    
    resp = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[{"role":"user","content":"Solve step by step: What is 25 * 4 + 10?"}],
        response_format=MathReasoning,
    )
    # extract structured data in the same way your handler does
    result = resp.choices[0].message.parsed if hasattr(resp, "choices") else str(resp)
    print("Final answer:", result.final_answer)
    print("Steps:")
    for step in result.steps:
        print(f"  - {step.explanation}: {step.output}")

    # --- Another structured output call ---
    print("\n=== Person extraction response ===")
    resp = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[{"role":"user","content":"Extract info: John Smith is a 30-year-old software engineer who knows Python, JavaScript, and SQL."}],
        response_format=PersonInfo,
    )
    # extract structured data
    result = resp.choices[0].message.parsed if hasattr(resp, "choices") else str(resp)
    print(f"Name: {result.name}, Age: {result.age}, Job: {result.occupation}")
    print(f"Skills: {', '.join(result.skills)}")

    # == test structured output with image ==
    with open("tests/ord_runways.jpg", "rb") as f:
        img_bytes = f.read()
    data_uri = f"data:image/jpeg;base64,{base64.standard_b64encode(img_bytes).decode()}"

    # Simple model for image description
    class ImageDescription(BaseModel):
        description: str
        objects_seen: List[str]

    # build the OpenAI message payload with an image_url piece
    image_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image and list objects you see:"},
            {"type": "image_url", "image_url": {"url": data_uri}}
        ]
    }

    resp = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[image_message],
        response_format=ImageDescription,
    )

    # your handler should have torn off that data_uri into `screenshots=[…]`
    result = resp.choices[0].message.parsed
    print("Image description:", result.description)
    print("Objects seen:", ', '.join(result.objects_seen))

    # tear down
    lai.end_step()
    lai.end_session()


if __name__ == "__main__":
    asyncio.run(run_test())