"""
Test script for sync and async event creation.

This script demonstrates:
- Synchronous event creation with create_event()
- Asynchronous event creation with acreate_event()
- Error event creation with create_error_event() and acreate_error_event()
- Blob offloading for large payloads
"""

import asyncio
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import lucidicai as lai


def test_sync_event_creation():
    """Test synchronous event creation."""
    print("\n" + "=" * 60)
    print("Testing SYNCHRONOUS event creation")
    print("=" * 60)
    
    # Test 1: Simple generic event
    print("\n1. Creating a simple generic event...")
    start = time.time()
    event_id = lai.create_event(
        type="generic",
        description="Test sync event",
        details={"test": "data", "number": 42}
    )
    elapsed = time.time() - start
    print(f"   ✓ Created event: {event_id[:8]}... ({elapsed:.3f}s)")
    
    # Test 2: Function call event
    print("\n2. Creating a function_call event...")
    start = time.time()
    event_id = lai.create_event(
        type="function_call",
        function_name="test_function",
        arguments={"arg1": "value1", "arg2": 123},
        return_value={"result": "success"}
    )
    elapsed = time.time() - start
    print(f"   ✓ Created event: {event_id[:8]}... ({elapsed:.3f}s)")
    
    # Test 3: Error event
    print("\n3. Creating an error_traceback event...")
    start = time.time()
    try:
        raise ValueError("This is a test error")
    except ValueError as e:
        event_id = lai.create_error_event(
            error=e,
            description="Test error event"
        )
    elapsed = time.time() - start
    print(f"   ✓ Created event: {event_id[:8]}... ({elapsed:.3f}s)")
    
    # Test 4: Large payload (should trigger blob offloading)
    print("\n4. Creating a large payload event (blob offloading)...")
    large_data = "x" * 100000  # 100KB of data
    start = time.time()
    event_id = lai.create_event(
        type="generic",
        description="Large payload test",
        details={"large_content": large_data}
    )
    elapsed = time.time() - start
    print(f"   ✓ Created event: {event_id[:8]}... ({elapsed:.3f}s)")
    print(f"   (Payload size: {len(large_data)} bytes - should use blob storage)")


async def test_async_event_creation():
    """Test asynchronous event creation."""
    print("\n" + "=" * 60)
    print("Testing ASYNCHRONOUS event creation")
    print("=" * 60)
    
    # Test 1: Simple async generic event
    print("\n1. Creating a simple async generic event...")
    start = time.time()
    event_id = await lai.acreate_event(
        type="generic",
        description="Test async event",
        details={"test": "async_data", "number": 99}
    )
    elapsed = time.time() - start
    print(f"   ✓ Created event: {event_id[:8]}... ({elapsed:.3f}s)")
    
    # Test 2: Async function call event
    print("\n2. Creating an async function_call event...")
    start = time.time()
    event_id = await lai.acreate_event(
        type="function_call",
        function_name="async_test_function",
        arguments={"async_arg": "async_value"},
        return_value={"async_result": "success"}
    )
    elapsed = time.time() - start
    print(f"   ✓ Created event: {event_id[:8]}... ({elapsed:.3f}s)")
    
    # Test 3: Async error event
    print("\n3. Creating an async error_traceback event...")
    start = time.time()
    try:
        raise RuntimeError("This is an async test error")
    except RuntimeError as e:
        event_id = await lai.acreate_error_event(
            error=e,
            description="Test async error event"
        )
    elapsed = time.time() - start
    print(f"   ✓ Created event: {event_id[:8]}... ({elapsed:.3f}s)")
    
    # Test 4: Large async payload (blob offloading in background)
    print("\n4. Creating a large async payload event (background blob upload)...")
    large_data = "y" * 100000  # 100KB of data
    start = time.time()
    event_id = await lai.acreate_event(
        type="generic",
        description="Large async payload test",
        details={"large_async_content": large_data}
    )
    elapsed = time.time() - start
    print(f"   ✓ Created event: {event_id[:8]}... ({elapsed:.3f}s)")
    print(f"   (Payload size: {len(large_data)} bytes - blob upload runs in background)")


async def test_concurrent_async_events():
    """Test creating multiple async events concurrently."""
    print("\n" + "=" * 60)
    print("Testing CONCURRENT async event creation")
    print("=" * 60)
    
    print("\nCreating 5 events concurrently...")
    start = time.time()
    
    tasks = [
        lai.acreate_event(
            type="generic",
            description=f"Concurrent event {i}",
            details={"index": i}
        )
        for i in range(5)
    ]
    
    event_ids = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    print(f"   ✓ Created {len(event_ids)} events in {elapsed:.3f}s")
    for i, event_id in enumerate(event_ids):
        print(f"     - Event {i}: {event_id[:8]}...")


async def run_all_async_tests():
    """Run all async tests in a single event loop."""
    await test_async_event_creation()
    await test_concurrent_async_events()


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Event Creation Test Script")
    print("#" * 60)
    
    # Initialize the SDK
    print("\nInitializing Lucidic SDK...")
    session_id = lai.init(
        session_name="Event Creation Test",
        providers=[]  # No telemetry providers needed for this test
    )
    print(f"Session ID: {session_id}")
    
    try:
        # Run sync tests
        test_sync_event_creation()
        
        # Run all async tests in a single event loop
        # (httpx AsyncClient is tied to an event loop, so we need to reuse it)
        asyncio.run(run_all_async_tests())
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    finally:
        # End the session
        print("\nEnding session...")
        lai.end_session(is_successful=True)
        print("Done!")


if __name__ == "__main__":
    main()

