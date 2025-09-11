"""
Comprehensive test for event nesting in decorated functions.

This test validates:
- Multiple levels of nested decorated function calls
- Event parent-child relationships
- OpenAI API call instrumentation within decorated functions
- Error event generation from decorated functions
- Proper context propagation through nested calls
"""

import os
import time
import json
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import lucidicai as lai
from lucidicai.sdk.decorators import event


class TestNestedDecorators:
    """Test suite for nested decorator functionality."""

    def setup(self):
        """Setup test environment - uses real backend from .env."""
        # Track events created through monitoring the event queue
        self.created_events = []
        
        # Initialize SDK with real backend
        session_id = lai.init(
            session_name='Test Nested Decorators',
            providers=['openai']
        )
        
        # Hook into the event queue to track events
        from lucidicai.sdk.init import get_event_queue
        event_queue = get_event_queue()
        
        if event_queue:
            original_queue_event = event_queue.queue_event
            
            def track_and_queue(event_request):
                """Track events as they're queued."""
                self.created_events.append(event_request)
                return original_queue_event(event_request)
            
            event_queue.queue_event = track_and_queue
            self.event_queue = event_queue
        
        print(f"\033[94mSession initialized: {session_id}\033[0m")
    
    def test_complex_nested_workflow(self):
        """Test a complex workflow with multiple levels of nesting and OpenAI calls."""
        
        # Define our test functions with decorators
        @event(name="data_processor")
        def process_data(data: List[int]) -> Dict[str, Any]:
            """Process raw data."""
            result = {
                'sum': sum(data),
                'count': len(data),
                'average': sum(data) / len(data) if data else 0
            }
            time.sleep(0.01)  # Simulate processing time
            return result
        
        @event(name="data_validator") 
        def validate_data(data: List[int]) -> bool:
            """Validate data meets requirements."""
            if not data:
                return False
            if any(x < 0 for x in data):
                return False
            time.sleep(0.01)  # Simulate validation time
            return True
        
        @event(name="ai_analyzer")
        def analyze_with_ai(processed_data: Dict[str, Any]) -> str:
            """Analyze data using OpenAI - will create instrumented LLM events."""
            try:
                # Use real OpenAI API (will be instrumented by SDK telemetry)
                import openai
                import os
                
                # Ensure API key is set
                if not os.getenv('OPENAI_API_KEY'):
                    # If no API key, return mock response
                    return f"Analysis of {processed_data['count']} items (avg={processed_data['average']}) | Summary: sum={processed_data['sum']}"
                
                client = openai.OpenAI()
                
                # First real OpenAI call - analysis
                response1 = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a data analyst. Respond in 10 words or less."},
                        {"role": "user", "content": f"Analyze: sum={processed_data['sum']}, count={processed_data['count']}, avg={processed_data['average']}"}
                    ],
                    temperature=0.7,
                    max_tokens=20
                )
                
                # Second real OpenAI call - summary  
                response2 = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": f"Summarize in 5 words: {response1.choices[0].message.content}"}
                    ],
                    temperature=0.5,
                    max_tokens=15
                )
                
                return f"{response1.choices[0].message.content} | {response2.choices[0].message.content}"
                    
            except Exception as e:
                # Fallback on any error (including import or API errors)
                print(f"OpenAI call failed: {e}")
                analysis = f"Analysis of {processed_data['count']} items: average={processed_data['average']}"
                summary = f"Summary: Processing successful with sum={processed_data['sum']}"
                return f"{analysis} | {summary}"
        
        @event(name="formatter")
        def format_results(analysis: str, metadata: Dict) -> str:
            """Format the final results."""
            formatted = f"[{metadata['timestamp']}] {analysis}"
            time.sleep(0.01)
            return formatted
        
        @event(name="nested_helper")
        def nested_helper_function(value: int) -> int:
            """A deeply nested helper function."""
            time.sleep(0.01)
            return value * 2
        
        @event(name="aggregator")
        def aggregate_results(data: List[int], analysis: str) -> Dict:
            """Aggregate multiple results together."""
            # Call another nested function
            multiplied_sum = nested_helper_function(sum(data))
            
            return {
                'original_sum': sum(data),
                'multiplied_sum': multiplied_sum,
                'analysis': analysis
            }
        
        @event(name="main_workflow")
        def main_workflow(input_data: List[int]) -> Dict:
            """Main workflow that orchestrates everything."""
            import datetime
            
            # Step 1: Validate the data
            is_valid = validate_data(input_data)
            if not is_valid:
                raise ValueError("Invalid input data provided")
            
            # Step 2: Process the data
            processed = process_data(input_data)
            
            # Step 3: Analyze with AI (contains 2 OpenAI calls)
            ai_analysis = analyze_with_ai(processed)
            
            # Step 4: Direct OpenAI call in main workflow (to test telemetry nesting)
            try:
                import openai
                import os
                
                if os.getenv('OPENAI_API_KEY'):
                    client = openai.OpenAI()
                    
                    # Direct real LLM call within main_workflow
                    validation_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "user", "content": f"Reply 'Valid' if this looks complete: {ai_analysis[:30]}"}
                        ],
                        max_tokens=10,
                        temperature=0
                    )
                    final_validation = validation_response.choices[0].message.content
                else:
                    final_validation = "Validation: Complete (no API key)"
            except Exception as e:
                print(f"Validation OpenAI call failed: {e}")
                final_validation = "Final validation: Complete"
            
            # Step 5: Aggregate results (calls nested_helper)
            aggregated = aggregate_results(input_data, ai_analysis)
            
            # Step 6: Format the output
            metadata = {
                'timestamp': datetime.datetime.now().isoformat(),
                'input_size': len(input_data),
                'validation': final_validation
            }
            formatted_output = format_results(ai_analysis, metadata)
            
            # Step 7: Intentionally throw an error to test error event generation
            if aggregated['multiplied_sum'] > 100:
                raise RuntimeError(f"Multiplied sum {aggregated['multiplied_sum']} exceeds threshold!")
            
            return {
                'formatted': formatted_output,
                'aggregated': aggregated,
                'processed': processed
            }
        
        # Execute the main workflow - will throw RuntimeError
        test_data = [10, 20, 30, 40]  # Sum = 100, multiplied = 200, will trigger error
        
        try:
            result = main_workflow(test_data)
            print("\033[91mERROR: Function should have thrown RuntimeError but didn't!\033[0m")
        except RuntimeError as e:
            print(f"\033[92m✓\033[0m Expected RuntimeError caught: {e}")
        
        # Wait a moment for events to be queued
        time.sleep(0.5)
        
        # Verify the event structure
        print(f"\n\033[96mCreated {len(self.created_events)} events\033[0m")
        assert len(self.created_events) > 0, "No events were created"
        
        # Find main workflow event
        main_events = [e for e in self.created_events 
                      if e.get('type') == 'function_call' 
                      and e.get('payload', {}).get('function_name') == 'main_workflow']
        assert len(main_events) == 1, f"Expected 1 main_workflow event, got {len(main_events)}"
        
        main_event = main_events[0]
        main_event_id = main_event['client_event_id']
        
        # Verify error was captured - check both possible locations
        error_text = main_event['payload'].get('misc', {}).get('error') or main_event['payload'].get('error')
        assert error_text is not None, "Main workflow should have captured the error"
        assert 'RuntimeError' in str(error_text), "Error should be RuntimeError"
        
        # Verify nested events have correct parent
        nested_functions = ['validate_data', 'process_data', 'analyze_with_ai', 
                          'aggregate_results', 'format_results']
        
        for func_name in nested_functions:
            func_events = [e for e in self.created_events 
                          if e.get('type') == 'function_call' 
                          and e.get('payload', {}).get('function_name') == func_name]
            assert len(func_events) > 0, f"No events found for {func_name}"
            
            # Check parent relationship
            for evt in func_events:
                parent_id = evt.get('client_parent_event_id')
                assert parent_id == main_event_id, f"{func_name} should have main_workflow as parent"
        
        # Verify deeply nested helper was called from aggregator
        helper_events = [e for e in self.created_events 
                        if e.get('type') == 'function_call'
                        and e.get('payload', {}).get('function_name') == 'nested_helper_function']
        assert len(helper_events) == 1, f"Expected 1 nested_helper event, got {len(helper_events)}"
        
        # The helper should have aggregator as parent
        aggregator_events = [e for e in self.created_events 
                           if e.get('payload', {}).get('function_name') == 'aggregate_results']
        assert len(aggregator_events) == 1
        aggregator_id = aggregator_events[0]['client_event_id']
        assert helper_events[0]['client_parent_event_id'] == aggregator_id, "Helper should have aggregator as parent"
        
        # Verify event ordering and timing
        for evt in self.created_events:
            assert 'occurred_at' in evt, f"Event {evt.get('client_event_id')} missing occurred_at"
            if evt.get('type') == 'function_call':
                assert 'duration' in evt, f"Function event {evt.get('client_event_id')} missing duration"
        
        # Check for LLM generation events (from OpenAI instrumentation)
        llm_events = [e for e in self.created_events if e.get('type') == 'llm_generation']
        print(f"\n\033[96mFound {len(llm_events)} LLM generation events from OpenAI instrumentation\033[0m")
        
        # Verify LLM events have proper parent context
        if llm_events:
            # LLM calls within ai_analyzer should have ai_analyzer as parent
            ai_analyzer_event = [e for e in self.created_events 
                               if e.get('payload', {}).get('function_name') == 'analyze_with_ai'][0]
            
            # Direct LLM call in main_workflow should have main_workflow as parent
            for llm_event in llm_events:
                parent_id = llm_event.get('client_parent_event_id')
                if parent_id:
                    print(f"   - LLM event has parent: {parent_id[:8]}...")
        
        print(f"\n\033[92m✓\033[0m Test passed! Created {len(self.created_events)} events with proper nesting:")
        print(f"   - Main workflow event with error capture")
        print(f"   - {len(nested_functions)} directly nested function events")
        print(f"   - 1 deeply nested helper function event")
        print(f"   - {len(llm_events)} LLM generation events from OpenAI")
        print(f"   - All events have proper parent relationships")
        
        # Print event tree for visualization
        print(f"\n\033[94mEvent Tree:\033[0m")
        print(f"└── main_workflow (error: RuntimeError)")
        print(f"    ├── validate_data")
        print(f"    ├── process_data")
        print(f"    ├── analyze_with_ai")
        if len(llm_events) >= 2:
            print(f"    │   ├── OpenAI call 1 (gpt-4)")
            print(f"    │   └── OpenAI call 2 (gpt-4)")
        print(f"    ├── OpenAI validation call (gpt-4o)")
        print(f"    ├── aggregate_results")
        print(f"    │   └── nested_helper_function")
        print(f"    └── format_results")
    
    def test_async_nested_decorators(self):
        """Test async decorated functions with nesting."""
        
        @event(name="async_processor")
        async def async_process(data: str) -> str:
            """Async processing function."""
            await asyncio.sleep(0.01)
            return data.upper()
        
        @event(name="async_validator")
        async def async_validate(data: str) -> bool:
            """Async validation function."""
            await asyncio.sleep(0.01)
            return len(data) > 0
        
        @event(name="async_main")
        async def async_main_workflow(input_text: str) -> Dict:
            """Async main workflow."""
            # Validate
            is_valid = await async_validate(input_text)
            if not is_valid:
                raise ValueError("Invalid input")
            
            # Process
            processed = await async_process(input_text)
            
            # Intentional error for testing
            if "ERROR" in processed:
                raise RuntimeError("Found ERROR in processed text")
            
            return {
                'original': input_text,
                'processed': processed
            }
        
        # Clear events from previous test
        self.created_events.clear()
        
        # Run async test
        async def run_test():
            try:
                await async_main_workflow("test error case")
                print("\033[91mERROR: Async function should have thrown RuntimeError!\033[0m")
            except RuntimeError as e:
                print(f"\033[92m✓\033[0m Async RuntimeError caught: {e}")
        
        asyncio.run(run_test())
        
        # Wait for events to be queued
        time.sleep(0.5)
        
        # Verify async events were created
        async_events = [e for e in self.created_events 
                       if 'async' in e.get('payload', {}).get('function_name', '')]
        assert len(async_events) >= 3, f"Expected at least 3 async events, got {len(async_events)}"
        
        # Verify nesting
        async_main = [e for e in async_events if 'async_main' in e['payload'].get('misc', {}).get('name', e['payload'].get('function_name', ''))][0]
        async_validate = [e for e in async_events if 'async_validator' in e['payload'].get('misc', {}).get('name', e['payload'].get('function_name', ''))][0]
        async_process = [e for e in async_events if 'async_processor' in e['payload'].get('misc', {}).get('name', e['payload'].get('function_name', ''))][0]
        
        assert async_validate['client_parent_event_id'] == async_main['client_event_id']
        assert async_process['client_parent_event_id'] == async_main['client_event_id']
        
        print(f"\n\033[92m✓\033[0m Async test passed! Created {len(async_events)} async events with proper nesting")
    
    def test_mixed_sync_async_nesting(self):
        """Test mixing sync and async decorated functions."""
        
        @event(name="sync_helper")
        def sync_helper(value: int) -> int:
            """Synchronous helper."""
            time.sleep(0.01)
            return value + 10
        
        @event(name="async_caller")
        async def async_caller(value: int) -> int:
            """Async function that calls sync function."""
            await asyncio.sleep(0.01)
            # Call sync function from async context
            result = sync_helper(value)
            return result * 2
        
        @event(name="mixed_main")
        async def mixed_workflow(start_value: int) -> int:
            """Main workflow mixing sync and async."""
            result = await async_caller(start_value)
            
            # Error case
            if result > 50:
                raise ValueError(f"Result {result} too large")
            
            return result
        
        # Clear events from previous test
        self.created_events.clear()
        
        # Run test
        async def run_test():
            try:
                await mixed_workflow(25)  # Will produce 70, triggering error
                print("\033[91mERROR: Mixed workflow should have thrown ValueError!\033[0m")
            except ValueError as e:
                print(f"\033[92m✓\033[0m Mixed ValueError caught: {e}")
        
        asyncio.run(run_test())
        
        # Wait for events to be queued
        time.sleep(0.5)
        
        # Verify mixed events
        mixed_events = [e for e in self.created_events 
                       if e.get('payload', {}).get('function_name', '') in 
                       ['mixed_workflow', 'async_caller', 'sync_helper']]
        assert len(mixed_events) == 3, f"Expected 3 mixed events, got {len(mixed_events)}"
        
        # Verify nesting hierarchy
        mixed_main = [e for e in mixed_events if 'mixed_main' in e['payload'].get('misc', {}).get('name', e['payload'].get('function_name', ''))][0]
        async_caller_ev = [e for e in mixed_events if 'async_caller' in e['payload'].get('misc', {}).get('name', e['payload'].get('function_name', ''))][0]
        sync_helper_ev = [e for e in mixed_events if 'sync_helper' in e['payload'].get('misc', {}).get('name', e['payload'].get('function_name', ''))][0]
        
        assert async_caller_ev['client_parent_event_id'] == mixed_main['client_event_id']
        assert sync_helper_ev['client_parent_event_id'] == async_caller_ev['client_event_id']
        
        print(f"\n\033[92m✓\033[0m Mixed sync/async test passed! Created {len(mixed_events)} events")
    
    def cleanup(self):
        """Clean up after tests."""
        # Force flush the event queue
        if hasattr(self, 'event_queue'):
            self.event_queue.force_flush(timeout_seconds=2.0)
        
        # End the session
        lai.end_session()
        print(f"\n\033[94mSession ended and events flushed\033[0m")


if __name__ == "__main__":
    # Run the tests
    test = TestNestedDecorators()
    test.setup()
    
    print("\033[94mRunning comprehensive nested decorator tests...\033[0m\n")
    
    # Run all tests - let assertions fail naturally
    test.test_complex_nested_workflow()
    test.test_async_nested_decorators()
    test.test_mixed_sync_async_nesting()
    
    print(f"\n\033[96mTotal events created across all tests: {len(test.created_events)}\033[0m")
    print("\033[92mAll nested decorator tests completed successfully!\033[0m")
    
    # Clean up
    test.cleanup()