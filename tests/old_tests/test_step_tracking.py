#!/usr/bin/env python3
"""Test script to verify step tracking is working correctly"""

import os
import lucidicai as lai

# Set up test environment
os.environ['LUCIDIC_API_KEY'] = 'test-key'
os.environ['LUCIDIC_AGENT_ID'] = 'test-agent'
os.environ['LUCIDIC_DEBUG'] = 'True'  # Use localhost

def test_step_tracking():
    """Test that multiple steps can be created and ended correctly"""
    print("Testing step tracking...")
    
    # Initialize session
    session_id = lai.init(session_name="Test Step Tracking")
    print(f"Session initialized: {session_id}")
    
    # Create and end first step
    print("\nCreating first step...")
    step1_id = lai.create_step(
        state="Starting first task",
        action="Initialize",
        goal="Complete first task"
    )
    print(f"First step created: {step1_id}")
    
    # Check active step
    session = lai.get_session()
    print(f"Active step after creating step 1: {session.active_step.step_id if session.active_step else 'None'}")
    
    # End first step
    print("\nEnding first step...")
    lai.end_step(
        state="First task completed",
        action="Finalize",
        goal="First task done"
    )
    print("First step ended")
    
    # Check if step is marked as finished
    step1 = session.step_history[step1_id]
    print(f"First step is_finished: {step1.is_finished}")
    
    # Create and end second step
    print("\nCreating second step...")
    step2_id = lai.create_step(
        state="Starting second task",
        action="Initialize",
        goal="Complete second task"
    )
    print(f"Second step created: {step2_id}")
    
    # Check active step
    print(f"Active step after creating step 2: {session.active_step.step_id if session.active_step else 'None'}")
    
    # End second step
    print("\nEnding second step...")
    lai.end_step(
        state="Second task completed",
        action="Finalize",
        goal="Second task done"
    )
    print("Second step ended")
    
    # Check if step is marked as finished
    step2 = session.step_history[step2_id]
    print(f"Second step is_finished: {step2.is_finished}")
    
    # Create third step with explicit step_id
    print("\nCreating third step...")
    step3_id = lai.create_step(
        state="Starting third task",
        action="Initialize",
        goal="Complete third task"
    )
    print(f"Third step created: {step3_id}")
    
    # End third step using explicit step_id
    print("\nEnding third step with explicit step_id...")
    lai.end_step(
        step_id=step3_id,
        state="Third task completed",
        action="Finalize",
        goal="Third task done"
    )
    print("Third step ended")
    
    # Check if step is marked as finished
    step3 = session.step_history[step3_id]
    print(f"Third step is_finished: {step3.is_finished}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Total steps created: {len(session.step_history)}")
    for step_id, step in session.step_history.items():
        print(f"Step {step_id}: is_finished = {step.is_finished}")
    
    # End session
    lai.end_session(is_successful=True, is_successful_reason="All steps completed successfully")
    print("\nSession ended successfully")

if __name__ == "__main__":
    try:
        test_step_tracking()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()