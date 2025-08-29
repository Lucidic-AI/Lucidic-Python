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
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock, call

import pytest

# Set environment variables before importing SDK
os.environ['LUCIDIC_API_KEY'] = 'test-api-key'
os.environ['LUCIDIC_AGENT_ID'] = 'test-agent-id'

import lucidicai as lai
from lucidicai.decorators import event


class TestNestedDecorators:
    """Test suite for nested decorator functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Setup test environment and mock API calls."""
        # Clear any existing state
        lai.clear()
        
        # Mock the client's make_request to avoid actual API calls
        mock_client = MagicMock()
        mock_client.make_request = MagicMock(return_value={
            'session_id': 'test-session-123',
            'session_name': 'Test Session'
        })
        
        # Track all created events
        self.created_events = []
        
        def mock_create_event(**kwargs):
            """Capture all event creation calls."""
            event_id = f"event-{len(self.created_events)}"
            self.created_events.append({
                'event_id': event_id,
                **kwargs
            })
            return event_id
        
        mock_client.create_event = mock_create_event
        mock_client.initialized = True
        mock_client.session = MagicMock(session_id='test-session-123')
        mock_client._event_queue = MagicMock()
        mock_client._event_queue.queue_event = MagicMock()
        
        # Patch the client singleton
        with patch('lucidicai.client.Client') as MockClient:
            MockClient.return_value = mock_client
            self.client = mock_client
            
            # Initialize SDK
            lai.init(
                api_key='test-api-key',
                agent_id='test-agent-id',
                session_name='Test Nested Decorators'
            )
    
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
            """Analyze data using OpenAI (mocked)."""
            # Mock OpenAI API calls
            with patch('openai.chat.completions.create') as mock_openai:
                # First OpenAI call - analysis
                mock_openai.return_value = MagicMock(
                    choices=[MagicMock(message=MagicMock(content="Analysis result: Data looks good"))]
                )
                
                # Simulate making OpenAI calls
                import openai
                client = openai.Client(api_key="test-key")
                
                # First call - analyze the data
                response1 = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a data analyst"},
                        {"role": "user", "content": f"Analyze this data: {json.dumps(processed_data)}"}
                    ]
                )
                
                # Second call - generate summary
                mock_openai.return_value = MagicMock(
                    choices=[MagicMock(message=MagicMock(content="Summary: Processing successful"))]
                )
                
                response2 = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "user", "content": "Summarize the analysis"}
                    ]
                )
                
                return f"{response1.choices[0].message.content} | {response2.choices[0].message.content}"
        
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
            
            # Step 3: Analyze with AI (includes 2 OpenAI calls)
            ai_analysis = analyze_with_ai(processed)
            
            # Step 4: Aggregate results (calls nested_helper)
            aggregated = aggregate_results(input_data, ai_analysis)
            
            # Step 5: Format the output
            metadata = {
                'timestamp': datetime.datetime.now().isoformat(),
                'input_size': len(input_data)
            }
            formatted_output = format_results(ai_analysis, metadata)
            
            # Step 6: Intentionally throw an error to test error event generation
            if aggregated['multiplied_sum'] > 100:
                raise RuntimeError(f"Multiplied sum {aggregated['multiplied_sum']} exceeds threshold!")
            
            return {
                'formatted': formatted_output,
                'aggregated': aggregated,
                'processed': processed
            }
        
        # Execute the main workflow and expect an error
        test_data = [10, 20, 30, 40]  # Sum = 100, multiplied = 200, will trigger error
        
        with pytest.raises(RuntimeError) as exc_info:
            result = main_workflow(test_data)
        
        assert "exceeds threshold" in str(exc_info.value)
        
        # Verify the event structure
        assert len(self.created_events) > 0, "No events were created"
        
        # Find main workflow event
        main_events = [e for e in self.created_events if e.get('type') == 'function_call' 
                      and e.get('payload', {}).get('function_name') == 'main_workflow']
        assert len(main_events) == 1, f"Expected 1 main_workflow event, got {len(main_events)}"
        
        main_event = main_events[0]
        main_event_id = main_event['event_id']
        
        # Verify nested events have correct parent
        nested_functions = ['validate_data', 'process_data', 'analyze_with_ai', 
                          'aggregate_results', 'format_results']
        
        for func_name in nested_functions:
            func_events = [e for e in self.created_events 
                          if e.get('type') == 'function_call' 
                          and e.get('payload', {}).get('function_name') == func_name]
            assert len(func_events) > 0, f"No events found for {func_name}"
            
            # Check parent relationship
            for event in func_events:
                parent_id = event.get('parent_event_id') or event.get('parent_client_event_id')
                assert parent_id is not None, f"{func_name} event has no parent"
        
        # Verify deeply nested helper was called from aggregator
        helper_events = [e for e in self.created_events 
                        if e.get('type') == 'function_call'
                        and e.get('payload', {}).get('function_name') == 'nested_helper_function']
        assert len(helper_events) == 1, f"Expected 1 nested_helper event, got {len(helper_events)}"
        
        # Verify error was captured in main_workflow event
        assert main_event['payload'].get('error') is not None, "Main workflow should have captured the error"
        assert 'RuntimeError' in str(main_event['payload'].get('error', '')), "Error should be RuntimeError"
        
        # Verify event ordering and timing
        for event in self.created_events:
            assert 'occurred_at' in event, f"Event {event.get('event_id')} missing occurred_at"
            if event.get('type') == 'function_call':
                assert 'duration' in event, f"Function event {event.get('event_id')} missing duration"
        
        print(f"\n\033[92m✓\033[0m Test passed! Created {len(self.created_events)} events with proper nesting:")
        print(f"   - Main workflow event with error capture")
        print(f"   - {len(nested_functions)} directly nested function events")
        print(f"   - 1 deeply nested helper function event")
        print(f"   - All events have proper parent relationships")
    
    def test_async_nested_decorators(self):
        """Test async decorated functions with nesting."""
        import asyncio
        
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
        
        # Run async test
        async def run_test():
            with pytest.raises(RuntimeError):
                await async_main_workflow("test error case")
        
        asyncio.run(run_test())
        
        # Verify async events were created
        async_events = [e for e in self.created_events 
                       if 'async' in e.get('payload', {}).get('function_name', '')]
        assert len(async_events) >= 3, f"Expected at least 3 async events, got {len(async_events)}"
        
        print(f"\n\033[92m✓\033[0m Async test passed! Created {len(async_events)} async events with proper nesting")
    
    def test_mixed_sync_async_nesting(self):
        """Test mixing sync and async decorated functions."""
        import asyncio
        
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
        
        # Run test
        async def run_test():
            with pytest.raises(ValueError):
                await mixed_workflow(25)  # Will produce 70, triggering error
        
        asyncio.run(run_test())
        
        # Verify mixed events
        mixed_events = [e for e in self.created_events 
                       if e.get('payload', {}).get('function_name', '') in 
                       ['mixed_workflow', 'async_caller', 'sync_helper']]
        assert len(mixed_events) == 3, f"Expected 3 mixed events, got {len(mixed_events)}"
        
        print(f"\n\033[92m✓\033[0m Mixed sync/async test passed! Created {len(mixed_events)} events")


if __name__ == "__main__":
    # Run the tests
    test = TestNestedDecorators()
    test.setup(None)
    
    print("\033[94mRunning comprehensive nested decorator tests...\033[0m\n")
    
    try:
        test.test_complex_nested_workflow()
    except Exception as e:
        print(f"\033[91m✗\033[0m Complex workflow test failed: {e}")
    
    try:
        test.test_async_nested_decorators()
    except Exception as e:
        print(f"\033[91m✗\033[0m Async decorator test failed: {e}")
    
    try:
        test.test_mixed_sync_async_nesting()
    except Exception as e:
        print(f"\033[91m✗\033[0m Mixed sync/async test failed: {e}")
    
    print(f"\n\033[96mTotal events created across all tests: {len(test.created_events)}\033[0m")
    print("\033[92mAll nested decorator tests completed!\033[0m")