"""Comprehensive streaming tests for Lucidic AI SDK"""
import os
import sys
import asyncio
import base64
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import lucidicai as lai
from openai import OpenAI, AsyncOpenAI
import anthropic

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("Missing ANTHROPIC_API_KEY")


def test_openai_sync_streaming():
    """Test synchronous OpenAI streaming"""
    print("\n=== OpenAI Sync Streaming Test ===")
    
    lai.init(
        session_name="OpenAI Sync Streaming Test",
        providers=["openai"]
    )
    
    lai.create_step(
        state="Testing sync streaming",
        action="Stream response from OpenAI",
        goal="Verify streaming wrapper works correctly"
    )
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        # Test 1: Basic streaming
        print("\nTest 1: Basic streaming response")
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count from 1 to 5, one number per line"}],
            stream=True,
            max_tokens=50
        )
        
        chunks_received = 0
        full_response = ""
        for chunk in stream:
            chunks_received += 1
            if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end='', flush=True)
        
        print(f"\n✓ Received {chunks_received} chunks")
        print(f"✓ Full response length: {len(full_response)} chars")
        
        # Test 2: Streaming with larger response
        print("\n\nTest 2: Streaming with story generation")
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Write a 3-sentence story about a robot"}],
            stream=True,
            max_tokens=150
        )
        
        chunks_received = 0
        full_response = ""
        print("Story: ", end='')
        for chunk in stream:
            chunks_received += 1
            if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end='', flush=True)
        
        print(f"\n✓ Received {chunks_received} chunks for story")
        print(f"✓ Story length: {len(full_response)} chars")
        
    except Exception as e:
        print(f"✗ Sync streaming failed: {type(e).__name__}: {str(e)}")
    
    lai.end_step()
    lai.end_session()
    print("\n✓ OpenAI sync streaming test completed")


async def test_openai_async_streaming():
    """Test asynchronous OpenAI streaming"""
    print("\n=== OpenAI Async Streaming Test ===")
    
    lai.init(
        session_name="OpenAI Async Streaming Test",
        providers=["openai"]
    )
    
    lai.create_step(
        state="Testing async streaming",
        action="Async stream response from OpenAI",
        goal="Verify async streaming wrapper works correctly"
    )
    
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    try:
        # Test 1: Basic async streaming
        print("\nTest 1: Basic async streaming response")
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "List 3 colors"}],
            stream=True,
            max_tokens=50
        )
        
        chunks_received = 0
        full_response = ""
        async for chunk in stream:
            chunks_received += 1
            if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end='', flush=True)
        
        print(f"\n✓ Received {chunks_received} async chunks")
        print(f"✓ Response length: {len(full_response)} chars")
        
        # Test 2: Concurrent async streams
        print("\n\nTest 2: Concurrent async streams")
        
        async def stream_task(prompt: str, task_id: int):
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=30
            )
            
            chunks = 0
            response = ""
            async for chunk in stream:
                chunks += 1
                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
            
            print(f"✓ Task {task_id} completed: {chunks} chunks, response: {response[:30]}...")
            return chunks, response
        
        # Run 3 concurrent streaming requests
        tasks = [
            stream_task("Name a fruit", 1),
            stream_task("Name a vegetable", 2),
            stream_task("Name an animal", 3)
        ]
        
        results = await asyncio.gather(*tasks)
        print(f"✓ All {len(results)} async tasks completed")
        
    except Exception as e:
        print(f"✗ Async streaming failed: {type(e).__name__}: {str(e)}")
    
    lai.end_step()
    lai.end_session()
    print("\n✓ OpenAI async streaming test completed")


def test_anthropic_streaming():
    """Test Anthropic streaming via OpenAI SDK"""
    print("\n=== Anthropic Streaming Test (via OpenAI SDK) ===")
    
    lai.init(
        session_name="Anthropic Streaming Test",
        providers=["openai"]
    )
    
    lai.create_step(
        state="Testing Anthropic streaming",
        action="Stream response from Anthropic",
        goal="Verify streaming works with Anthropic base URL"
    )
    
    # Use OpenAI SDK with Anthropic base URL
    client = OpenAI(
        api_key=ANTHROPIC_API_KEY,
        base_url="https://api.anthropic.com/v1",
        default_headers={
            "anthropic-version": "2023-06-01"
        }
    )
    
    try:
        print("\nTest: Anthropic streaming response")
        stream = client.chat.completions.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Write a haiku about coding"}],
            stream=True,
            max_tokens=100
        )
        
        chunks_received = 0
        full_response = ""
        print("Haiku: ")
        for chunk in stream:
            chunks_received += 1
            if hasattr(chunk, 'choices') and chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end='', flush=True)
        
        print(f"\n\n✓ Received {chunks_received} chunks from Anthropic")
        print(f"✓ Haiku length: {len(full_response)} chars")
        
    except Exception as e:
        print(f"✗ Anthropic streaming failed: {type(e).__name__}: {str(e)}")
    
    lai.end_step()
    lai.end_session()
    print("\n✓ Anthropic streaming test completed")


def test_streaming_with_images():
    """Test streaming with image inputs"""
    print("\n=== Streaming with Images Test ===")
    
    lai.init(
        session_name="Streaming with Images Test",
        providers=["openai"]
    )
    
    lai.create_step(
        state="Testing streaming with images",
        action="Stream response with image input",
        goal="Verify streaming works with multimodal content"
    )
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        # Load test image
        image_path = os.path.join(os.path.dirname(__file__), "ord_runways.jpg")
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        data_uri = f"data:image/jpeg;base64,{base64.standard_b64encode(img_bytes).decode()}"
        
        print("\nTest: Streaming response with image analysis")
        
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what you see in 2-3 sentences:"},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }
        
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[message],
            stream=True,
            max_tokens=150
        )
        
        chunks_received = 0
        full_response = ""
        print("Description: ", end='')
        for chunk in stream:
            chunks_received += 1
            if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end='', flush=True)
        
        print(f"\n\n✓ Received {chunks_received} chunks with image input")
        print(f"✓ Description length: {len(full_response)} chars")
        
    except Exception as e:
        print(f"✗ Streaming with images failed: {type(e).__name__}: {str(e)}")
    
    lai.end_step()
    lai.end_session()
    print("\n✓ Streaming with images test completed")


def test_error_handling_in_streaming():
    """Test error handling during streaming"""
    print("\n=== Streaming Error Handling Test ===")
    
    lai.init(
        session_name="Streaming Error Handling Test",
        providers=["openai"]
    )
    
    lai.create_step(
        state="Testing error handling",
        action="Test streaming error scenarios",
        goal="Verify proper error handling in streaming"
    )
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Test 1: Invalid model
    print("\nTest 1: Invalid model name")
    try:
        stream = client.chat.completions.create(
            model="invalid-model-name",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        for chunk in stream:
            pass
        print("✗ Should have raised an error")
    except Exception as e:
        print(f"✓ Correctly caught error: {type(e).__name__}")
    
    # Test 2: Token limit with streaming
    print("\nTest 2: Very low token limit")
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Tell me a long story"}],
            stream=True,
            max_tokens=5  # Very low limit
        )
        
        chunks = 0
        response = ""
        for chunk in stream:
            chunks += 1
            if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
        
        print(f"✓ Handled low token limit: {chunks} chunks, '{response}'")
        
    except Exception as e:
        print(f"✗ Failed: {type(e).__name__}: {str(e)}")
    
    lai.end_step()
    lai.end_session()
    print("\n✓ Error handling test completed")


async def main():
    """Run all streaming tests"""
    print("=" * 60)
    print("LUCIDIC AI SDK - COMPREHENSIVE STREAMING TESTS")
    print("=" * 60)
    
    # Run sync tests
    test_openai_sync_streaming()
    
    # Run async tests
    await test_openai_async_streaming()
    
    # Run Anthropic test
    test_anthropic_streaming()
    
    # Run multimodal test
    test_streaming_with_images()
    
    # Run error handling tests
    test_error_handling_in_streaming()
    
    print("\n" + "=" * 60)
    print("ALL STREAMING TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())