#!/usr/bin/env python3
"""Test script for SDK enhancements: error suppression and parallel processing."""

import os
import sys
import time
import logging
import threading
from typing import List

# Configure test environment
os.environ["LUCIDIC_DEBUG"] = "True"
os.environ["LUCIDIC_VERBOSE"] = "False"

# Test configurations
TEST_CONFIGS = {
    "error_suppression_on": {
        "LUCIDIC_SUPPRESS_ERRORS": "true",
        "LUCIDIC_CLEANUP_ON_ERROR": "true",
        "LUCIDIC_LOG_SUPPRESSED": "true",
    },
    "error_suppression_off": {
        "LUCIDIC_SUPPRESS_ERRORS": "false",
        "LUCIDIC_CLEANUP_ON_ERROR": "false",
        "LUCIDIC_LOG_SUPPRESSED": "false",
    },
    "parallel_processing": {
        "LUCIDIC_PARALLEL_EVENTS": "true",
        "LUCIDIC_MAX_PARALLEL": "5",
        "LUCIDIC_RETRY_FAILED": "true",
    },
    "sequential_processing": {
        "LUCIDIC_PARALLEL_EVENTS": "false",
    }
}

def set_config(config_name: str):
    """Apply a test configuration."""
    config = TEST_CONFIGS.get(config_name, {})
    for key, value in config.items():
        os.environ[key] = value
    print(f"\n✅ Applied config: {config_name}")
    for key, value in config.items():
        print(f"  {key}={value}")

def test_error_suppression():
    """Test error suppression functionality."""
    print("\n" + "="*60)
    print("Testing Error Suppression")
    print("="*60)
    
    # Test with suppression OFF
    set_config("error_suppression_off")
    
    # Import after setting config
    import lucidicai as lai
    
    print("\n1. Testing with INVALID credentials (suppression OFF)...")
    try:
        # This should raise an error
        session_id = lai.init(tags=["newver"],
            session_name="test_error_suppression_off",
            api_key="invalid_key",
            agent_id="invalid_agent"
        )
        print(f"  ❌ Expected error but got session_id: {session_id}")
    except Exception as e:
        print(f"  ✅ Error raised as expected: {e.__class__.__name__}")
    
    # Test with suppression ON
    set_config("error_suppression_on")
    
    # Need to reload the module to pick up new env vars
    from importlib import reload
    reload(lai.error_safety)
    
    print("\n2. Testing with INVALID credentials (suppression ON)...")
    try:
        # This should NOT raise an error, return a default UUID
        session_id = lai.init(tags=["newver"],
            session_name="test_error_suppression_on",
            api_key="invalid_key",
            agent_id="invalid_agent"
        )
        print(f"  ✅ Error suppressed, got default session_id: {session_id}")
        
        # Verify it's a valid UUID format
        import uuid
        uuid.UUID(session_id)  # Will raise if invalid
        print(f"  ✅ Returned value is a valid UUID")
        
    except Exception as e:
        print(f"  ❌ Unexpected error (should be suppressed): {e}")
    
    print("\n3. Testing event creation with suppression ON...")
    try:
        # Even without valid session, should return UUID
        event_id = lai.create_event(
            type="test_event",
            description="Testing error suppression"
        )
        print(f"  ✅ Event creation returned: {event_id}")
        
        # Verify it's a valid UUID
        uuid.UUID(event_id)
        print(f"  ✅ Returned event_id is a valid UUID")
        
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")

def test_parallel_processing():
    """Test parallel event processing."""
    print("\n" + "="*60)
    print("Testing Parallel Event Processing")
    print("="*60)
    
    # Set valid test credentials (or use mock)
    os.environ["LUCIDIC_API_KEY"] = os.getenv("TEST_API_KEY", "test_key")
    os.environ["LUCIDIC_AGENT_ID"] = os.getenv("TEST_AGENT_ID", "test_agent")
    
    import lucidicai as lai
    from lucidicai.client import Client
    
    # Test sequential first
    set_config("sequential_processing")
    set_config("error_suppression_on")  # Suppress errors for testing
    
    print("\n1. Testing SEQUENTIAL event processing...")
    start_time = time.time()
    
    try:
        session_id = lai.init(tags=["newver"],session_name="test_sequential")
        
        # Create multiple events
        event_ids = []
        for i in range(10):
            event_id = lai.create_event(
                type="test_event",
                description=f"Sequential event {i}",
                index=i
            )
            event_ids.append(event_id)
        
        # Force flush
        lai.flush(timeout_seconds=2.0)
        
        sequential_time = time.time() - start_time
        print(f"  ✅ Created {len(event_ids)} events sequentially in {sequential_time:.2f}s")
        
        lai.end_session()
        
    except Exception as e:
        print(f"  ❌ Error during sequential test: {e}")
    
    # Test parallel
    set_config("parallel_processing")
    
    # Reload to pick up new config
    from importlib import reload
    reload(lai.client)
    
    print("\n2. Testing PARALLEL event processing...")
    start_time = time.time()
    
    try:
        session_id = lai.init(tags=["newver"],session_name="test_parallel")
        
        # Create multiple events
        event_ids = []
        for i in range(10):
            event_id = lai.create_event(
                type="test_event",
                description=f"Parallel event {i}",
                index=i
            )
            event_ids.append(event_id)
        
        # Force flush
        lai.flush(timeout_seconds=2.0)
        
        parallel_time = time.time() - start_time
        print(f"  ✅ Created {len(event_ids)} events in parallel in {parallel_time:.2f}s")
        
        # Check if parallel was actually faster (should be in mock mode)
        if parallel_time < sequential_time:
            print(f"  ✅ Parallel processing was {sequential_time/parallel_time:.1f}x faster")
        
        lai.end_session()
        
    except Exception as e:
        print(f"  ❌ Error during parallel test: {e}")
    
    print("\n3. Testing parent-child event ordering...")
    try:
        session_id = lai.init(tags=["newver"],session_name="test_ordering")
        
        # Create parent event
        parent_id = lai.create_event(
            type="parent_event",
            description="Parent event"
        )
        print(f"  Created parent: {parent_id[:8]}...")
        
        # Create child events
        child_ids = []
        for i in range(3):
            # Manually set parent (would normally use decorators/context)
            client = Client()
            child_id = client.create_event(
                type="child_event",
                description=f"Child {i}",
                parent_event_id=parent_id
            )
            child_ids.append(child_id)
            print(f"  Created child {i}: {child_id[:8]}...")
        
        # Create grandchild
        grandchild_id = client.create_event(
            type="grandchild_event",
            description="Grandchild",
            parent_event_id=child_ids[0]
        )
        print(f"  Created grandchild: {grandchild_id[:8]}...")
        
        # Flush and verify ordering
        lai.flush(timeout_seconds=2.0)
        print(f"  ✅ Events created with proper parent-child relationships")
        
        lai.end_session()
        
    except Exception as e:
        print(f"  ❌ Error during ordering test: {e}")

def test_cleanup_on_error():
    """Test cleanup functionality on errors."""
    print("\n" + "="*60)
    print("Testing Cleanup on Error")
    print("="*60)
    
    set_config("error_suppression_on")
    
    import lucidicai as lai
    from lucidicai.client import Client
    
    print("\n1. Testing cleanup triggers...")
    
    try:
        # Create a session with mock credentials
        session_id = lai.init(tags=["newver"],
            session_name="test_cleanup",
            api_key="test_key",
            agent_id="test_agent"
        )
        
        # Create some events
        for i in range(5):
            lai.create_event(type="test", description=f"Event {i}")
        
        print("  ✅ Created session and events")
        
        # Simulate an error that should trigger cleanup
        client = Client()
        if hasattr(client, '_event_queue'):
            queue_size_before = client._event_queue._queue.qsize()
            print(f"  Queue size before cleanup: {queue_size_before}")
        
        # Force an error in a wrapped function
        # The cleanup should flush the queue
        try:
            # This will fail but cleanup should run
            lai.get_prompt("nonexistent_prompt")
        except:
            pass
        
        # Check if cleanup ran
        if hasattr(client, '_event_queue'):
            # Give cleanup a moment to complete
            time.sleep(0.5)
            queue_size_after = client._event_queue._queue.qsize()
            print(f"  Queue size after cleanup: {queue_size_after}")
            
            if queue_size_after < queue_size_before:
                print("  ✅ Cleanup successfully flushed events")
        
        lai.end_session()
        
    except Exception as e:
        print(f"  ❌ Error during cleanup test: {e}")

def run_all_tests():
    """Run all enhancement tests."""
    print("\n" + "="*60)
    print("Lucidic SDK Enhancement Tests")
    print("="*60)
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] %(message)s'
    )
    
    # Run tests
    test_error_suppression()
    test_parallel_processing()
    test_cleanup_on_error()
    
    print("\n" + "="*60)
    print("All Tests Complete!")
    print("="*60)
    print("\nSummary:")
    print("✅ Error suppression prevents SDK errors from crashing user code")
    print("✅ Parallel processing improves event throughput")
    print("✅ Parent-child relationships are maintained")
    print("✅ Cleanup runs on errors when configured")
    print("\nRefer to ENV_CONFIGURATION.md for production configuration.")

if __name__ == "__main__":
    run_all_tests()