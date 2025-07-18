#!/usr/bin/env python3
"""
Test script for Lucidic SDK with Anthropic Thinking mode.
This tests the SDK's ability to handle ThinkingBlock responses from Anthropic.

Required environment variables:
- ANTHROPIC_API_KEY: Your Anthropic API key
- LUCIDIC_API_KEY: Your Lucidic API key
- LUCIDIC_AGENT_ID: Your Lucidic agent ID

You can set these in a .env file or export them in your shell.

Note: Extended thinking is supported in these models:
- Claude Opus 4 (claude-opus-4-20250514)
- Claude Sonnet 4 (claude-sonnet-4-20250514)
- Claude Sonnet 3.7 (claude-3-7-sonnet-20250219)

This test uses Claude 3.7 Sonnet. For Claude 4 models, you can also use
the beta header "anthropic-beta: interleaved-thinking-2025-05-14" for
interleaved thinking between tool calls.
"""

import os
import asyncio
from anthropic import Anthropic
import lucidicai as lai
from dotenv import load_dotenv

load_dotenv()

def test_anthropic_thinking_sync():
    """Test synchronous Anthropic calls with thinking mode"""
    print("Testing Anthropic Thinking mode (synchronous)...")
    
    # Create Anthropic client - SDK will automatically handle it with the provider
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Create a step for this test
    lai.create_step(
        action="Test Anthropic Thinking Mode",
        goal="Testing ThinkingBlock handling in Anthropic responses"
    )
    
    try:
        # Test 1: Simple thinking mode request
        print("\nTest 1: Simple thinking request...")
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",  # Using Claude 3.7 Sonnet which supports extended thinking
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": "Think step by step about how to calculate the factorial of 5."
            }],
            thinking={
                "type": "enabled",
                "budget_tokens": 2048  # Minimum is 1024
            }
        )
        
        # Check for thinking blocks in response
        thinking_content = None
        text_content = None
        for block in response.content:
            if hasattr(block, 'type'):
                if block.type == 'thinking':
                    thinking_content = getattr(block, 'thinking', '')
                    print(f"Thinking block detected: {thinking_content[:100]}...")
                elif block.type == 'text':
                    text_content = block.text
        
        if text_content:
            print(f"Response: {text_content[:200]}...")
        else:
            print("No text content found in response")
        
        # Test 2: Complex reasoning with thinking
        print("\nTest 2: Complex reasoning with thinking...")
        response2 = client.messages.create(
            model="claude-3-7-sonnet-20250219",  # Using Claude 3.7 Sonnet which supports extended thinking
            max_tokens=5000,
            messages=[{
                "role": "user",
                "content": "Think through this problem: If a train leaves Station A at 2 PM traveling at 60 mph, and another train leaves Station B at 3 PM traveling at 80 mph, and the stations are 280 miles apart, when will they meet?"
            }],
            thinking={
                "type": "enabled",
                "budget_tokens": 3000
            }
        )
        
        # Check for thinking blocks in response2
        for block in response2.content:
            if hasattr(block, 'type'):
                if block.type == 'thinking':
                    print(f"Thinking block detected in response2")
                elif block.type == 'text':
                    print(f"Response: {block.text[:200]}...")
        
        # Test 3: Multi-turn conversation with thinking
        print("\nTest 3: Multi-turn conversation with thinking...")
        messages = [
            {"role": "user", "content": "Let's solve a logic puzzle. Think through this: Three houses are in a row. The red house is to the left of the green house. The blue house is to the right of the red house. What is the order of the houses?"},
            {"role": "assistant", "content": "I need to think through this step-by-step.\n\nGiven information:\n- Three houses in a row\n- Red house is to the left of green house\n- Blue house is to the right of red house\n\nFrom 'Red is left of Green': Red < Green\nFrom 'Blue is right of Red': Red < Blue\n\nSo we have: Red < Green and Red < Blue\n\nThis means Red must be the leftmost house. Now I need to determine the order of Blue and Green.\n\nIf the order were Red, Green, Blue, that would satisfy both conditions.\nIf the order were Red, Blue, Green, that would also satisfy both conditions.\n\nWait, let me reconsider. If Blue is to the right of Red, and Red is to the left of Green, we need to check if there's a unique solution.\n\nActually, the order must be: Red, Blue, Green\n\nThis satisfies:\n- Red is to the left of Green ✓\n- Blue is to the right of Red ✓"},
            {"role": "user", "content": "Think about whether your answer is correct. What if the order was Red, Green, Blue?"}
        ]
        
        response3 = client.messages.create(
            model="claude-3-7-sonnet-20250219",  # Using Claude 3.7 Sonnet which supports extended thinking
            max_tokens=4096,
            messages=messages,
            thinking={
                "type": "enabled",
                "budget_tokens": 2048
            }
        )
        
        # Check for thinking blocks in response3
        for block in response3.content:
            if hasattr(block, 'type'):
                if block.type == 'thinking':
                    print(f"Thinking block detected in response3")
                elif block.type == 'text':
                    print(f"Response: {block.text[:200]}...")
        
        # Update step with results
        lai.update_step(
            state="Test completed successfully",
            action="Ran 3 thinking mode tests",
            eval_score=1.0,
            eval_description="All thinking mode tests passed"
        )
        
    except Exception as e:
        print(f"Error during testing: {e}")
        lai.update_step(
            state="Test failed",
            action="Error during testing",
            eval_score=0.0,
            eval_description=f"Error: {str(e)}"
        )
        raise
    
    finally:
        lai.end_step()
        pass

def test_anthropic_thinking_streaming():
    """Test streaming Anthropic calls with thinking mode"""
    print("\n\nTesting Anthropic Thinking mode (streaming)...")
    
    # Create Anthropic client - SDK will automatically handle it with the provider
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Create a step for this test
    lai.create_step(
        action="Test Anthropic Thinking Mode Streaming",
        goal="Testing ThinkingBlock handling in streaming Anthropic responses"
    )
    
    try:
        print("\nTest 4: Streaming with thinking mode...")
        
        # Create a streaming request
        stream = client.messages.create(
            model="claude-3-7-sonnet-20250219",  # Using Claude 3.7 Sonnet which supports extended thinking
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": "Think through how to implement a binary search algorithm in Python, then provide the code."
            }],
            stream=True,
            thinking={
                "type": "enabled",
                "budget_tokens": 2500
            }
        )
        
        # Process the stream
        full_response = ""
        thinking_blocks = []
        
        for chunk in stream:
            if hasattr(chunk, 'type'):
                if chunk.type == 'content_block_start':
                    if hasattr(chunk, 'content_block') and hasattr(chunk.content_block, 'type'):
                        if chunk.content_block.type == 'thinking':
                            print('\nThinking block started...', end='', flush=True)
                elif chunk.type == 'content_block_delta':
                    if hasattr(chunk.delta, 'text'):
                        full_response += chunk.delta.text
                        print('.', end='', flush=True)
                    elif hasattr(chunk.delta, 'thinking'):
                        print('T', end='', flush=True)
                        thinking_blocks.append(chunk)
        
        print(f"\n\nStreaming response received ({len(full_response)} chars)")
        print(f"Thinking blocks detected: {len(thinking_blocks)}")
        
        lai.update_step(
            state="Streaming test completed",
            action=f"Processed {len(full_response)} chars with {len(thinking_blocks)} thinking blocks",
            eval_score=1.0,
            eval_description="Streaming with thinking blocks handled successfully"
        )
        
    except Exception as e:
        print(f"Error during streaming test: {e}")
        lai.update_step(
            state="Streaming test failed",
            action="Error during streaming test",
            eval_score=0.0,
            eval_description=f"Error: {str(e)}"
        )
        raise
    
    finally:
        lai.end_step()
        pass

async def test_anthropic_thinking_async():
    """Test asynchronous Anthropic calls with thinking mode"""
    print("\n\nTesting Anthropic Thinking mode (asynchronous)...")
    
    from anthropic import AsyncAnthropic
    
    # Create async Anthropic client - SDK will automatically handle it with the provider
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Create a step for this test
    lai.create_step(
        action="Test Anthropic Thinking Mode Async",
        goal="Testing ThinkingBlock handling in async Anthropic responses"
    )
    
    try:
        print("\nTest 5: Async thinking mode request...")
        
        response = await client.messages.create(
            model="claude-3-7-sonnet-20250219",  # Using Claude 3.7 Sonnet which supports extended thinking
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": "Think about the most efficient sorting algorithm for a nearly sorted array and explain why."
            }],
            thinking={
                "type": "enabled",
                "budget_tokens": 2048
            }
        )
        
        # Check for thinking blocks in async response
        for block in response.content:
            if hasattr(block, 'type'):
                if block.type == 'thinking':
                    print(f"Thinking block detected in async response")
                elif block.type == 'text':
                    print(f"Async response: {block.text[:200]}...")
        
        lai.update_step(
            state="Async test completed",
            action="Executed async thinking mode request",
            eval_score=1.0,
            eval_description="Async thinking mode handled successfully"
        )
        
    except Exception as e:
        print(f"Error during async test: {e}")
        lai.update_step(
            state="Async test failed",
            action="Error during async test",
            eval_score=0.0,
            eval_description=f"Error: {str(e)}"
        )
        raise
    
    finally:
        lai.end_step()
        pass

def main():
    """Run all tests"""
    print("Starting Lucidic SDK Anthropic Thinking Mode Tests")
    print("=" * 50)
    
    # Check for required environment variables
    required_vars = ["ANTHROPIC_API_KEY", "LUCIDIC_API_KEY", "LUCIDIC_AGENT_ID"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these environment variables or add them to a .env file.")
        return
    
    # Initialize the SDK with Anthropic provider
    session_id = lai.init(
        session_name="Anthropic Thinking Mode Test",
        providers=["anthropic"]
    )
    print(f"Session initialized: {session_id}")
    
    try:
        # Run synchronous tests
        test_anthropic_thinking_sync()
        
        # Run streaming tests
        test_anthropic_thinking_streaming()
        
        # Run async tests
        asyncio.run(test_anthropic_thinking_async())
        
        print("\n\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\n\nTest suite failed: {e}")
        raise
    
    finally:
        # End the session
        lai.end_session()
        print("\nSession ended.")

if __name__ == "__main__":
    main()