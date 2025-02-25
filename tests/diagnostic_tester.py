"""Test script that focuses on event management without handler callbacks"""
import os
import time
from dotenv import load_dotenv
import lucidicai
from enhanced_langchain_agent import LangchainAgent
from lucidicai.diagnostic_langchain import LucidicLangchainHandler

def test_event_lifecycle():
    """Test just the event lifecycle directly"""
    print("\n=== Test Event Lifecycle ===")
    
    # Setup
    load_dotenv()
    lucidic_api_key = os.getenv("LUCIDIC_API_KEY")
    agent_id = os.getenv("AGENT_ID")
    
    if not all([lucidic_api_key, agent_id]):
        raise ValueError("Missing required environment variables")

    # Initialize Lucidic
    print("\n=== Initializing Lucidic session ===")
    session = lucidicai.init(
        lucidic_api_key=lucidic_api_key,
        agent_id=agent_id,
        session_name="event_test",
        provider="langchain"
    )

    # Create step
    print("\n=== Creating step ===")
    step = lucidicai.create_step(
        state="testing",
        action="testing event lifecycle",
        goal="verify event creation and completion"
    )
    
    # Test event creation and finishing manually
    try:
        print("\n=== Creating first event ===")
        event1 = lucidicai.create_event(description="Test event 1")
        print(f"Created event 1 with ID: {event1.event_id}")
        
        print("\n=== Finishing first event ===")
        # Directly access the event object properties
        print(f"Before finishing - is_finished: {event1.is_finished}")
        event1.finish_event(is_successful=True)
        print(f"After finishing - is_finished: {event1.is_finished}")
        
        # Try to create a second event
        print("\n=== Creating second event ===")
        event2 = lucidicai.create_event(description="Test event 2")
        print(f"Created event 2 with ID: {event2.event_id}")
        
        print("\n=== Finishing second event ===")
        event2.finish_event(is_successful=True)
        print(f"After finishing - is_finished: {event2.is_finished}")
        
        # Finish step
        print("\n=== Finishing step ===")
        lucidicai.finish_step(
            is_successful=True,
            state="completed",
            action="Completed event lifecycle test"
        )
        
        # End session
        print("\n=== Ending session ===")
        lucidicai.end_session(is_successful=True)
        
    except Exception as e:
        print(f"\n=== Error: {e} ===")
    
    print("\n=== Test completed ===")

def test_minimal_handler():
    """Test with minimal handler that only logs events"""
    print("\n=== Test Minimal Handler ===")
    
    # Setup
    load_dotenv()
    lucidic_api_key = os.getenv("LUCIDIC_API_KEY")
    agent_id = os.getenv("AGENT_ID")
    
    if not all([lucidic_api_key, agent_id]):
        raise ValueError("Missing required environment variables")

    # Initialize Lucidic
    print("\n=== Initializing Lucidic session ===")
    session = lucidicai.init(
        lucidic_api_key=lucidic_api_key,
        agent_id=agent_id,
        session_name="handler_test",
        provider="langchain"
    )

    # Create step
    print("\n=== Creating step ===")
    step = lucidicai.create_step(
        state="testing",
        action="testing minimal handler",
        goal="verify handler doesn't interfere"
    )
    
    # Create agent
    print("\n=== Creating agent ===")
    agent = LangchainAgent()
    
    # Create minimal handler
    print("\n=== Creating handler ===")
    handler = LucidicLangchainHandler(lucidicai.Client())
    
    # Attach handler
    print("\n=== Attaching handler ===")
    handler.attach_to_llms(agent.chat_model)
    
    # Test a chat - should only log, not create events
    print("\n=== Testing chat with minimal handler ===")
    try:
        result = agent.test_chat("What is a test?")
        print(f"Chat result: {result['success']}")
        
        # Check if step has events (it shouldn't)
        print(f"\n=== Step has {len(step.event_history)} events ===")
        
        # Create an event manually
        print("\n=== Creating event manually ===")
        event = lucidicai.create_event(description="Manual test event")
        print(f"Created event with ID: {event.event_id}")
        
        # Finish event manually
        print("\n=== Finishing event manually ===")
        event.finish_event(is_successful=True)
        print(f"After finishing - is_finished: {event.is_finished}")
        
        # Finish step
        print("\n=== Finishing step ===")
        lucidicai.finish_step(
            is_successful=True,
            state="completed",
            action="Completed minimal handler test"
        )
        
        # End session
        print("\n=== Ending session ===")
        lucidicai.end_session(is_successful=True)
        
    except Exception as e:
        print(f"\n=== Error: {e} ===")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_event_lifecycle()
    test_minimal_handler()