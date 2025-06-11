import os
import sys
from pydantic import BaseModel
from typing import List
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock OpenAI SDK before importing Lucidic
import unittest.mock

# Mock the OpenAI resources
with patch('openai.resources.chat.completions') as mock_completions, \
     patch('openai.resources.beta.chat.completions') as mock_beta_completions:
    
    mock_completions.Completions = MagicMock()
    mock_completions.Completions.create = MagicMock()
    
    mock_beta_completions.Completions = MagicMock()
    mock_beta_completions.Completions.parse = MagicMock()
    
    # Now import our handler and singleton clearer
    from lucidicai.providers.openai_handler import OpenAIHandler
    from lucidicai.singleton import clear_singletons

def test_openai_handler_monkey_patching():
    """Test that the OpenAI handler correctly monkey-patches both create and parse methods"""
    
    # Clear any existing singleton instances
    clear_singletons()
    
    # Create a mock client
    mock_client = MagicMock()
    mock_client.session = MagicMock()
    mock_client.session.active_step = None
    
    # Create handler instance
    handler = OpenAIHandler(mock_client)
    
    # Test that original methods are None initially
    assert handler.original_create is None
    assert handler.original_parse is None
    
    print("✓ Handler initialized correctly")
    
    # Test override method
    with patch('openai.resources.chat.completions') as mock_completions, \
         patch('openai.resources.beta.chat.completions') as mock_beta_completions:
        
        # Set up mocks
        original_create = MagicMock()
        original_parse = MagicMock()
        
        mock_completions.Completions = MagicMock()
        mock_completions.Completions.create = original_create
        
        mock_beta_completions.Completions = MagicMock()
        mock_beta_completions.Completions.parse = original_parse
        
        # Call override
        handler.override()
        
        # Verify original methods are stored
        assert handler.original_create == original_create
        assert handler.original_parse == original_parse
        
        # Verify methods are patched (they should be different functions now)
        assert mock_completions.Completions.create != original_create
        assert mock_beta_completions.Completions.parse != original_parse
        
        print("✓ Methods successfully monkey-patched")
        
        # Test undo override
        handler.undo_override()
        
        # Verify methods are restored
        assert mock_completions.Completions.create == original_create
        assert mock_beta_completions.Completions.parse == original_parse
        
        # Verify stored references are cleared
        assert handler.original_create is None
        assert handler.original_parse is None
        
        print("✓ Methods successfully restored")

def test_structured_output_response_handling():
    """Test that structured output responses are handled correctly"""
    
    # Clear any existing singleton instances
    clear_singletons()
    
    # Create a mock client
    mock_client = MagicMock()
    
    # Create handler instance
    handler = OpenAIHandler(mock_client)
    
    # Create mock response with parsed attribute (structured output)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.parsed = {"name": "John", "age": 30}
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.model = "gpt-4o-2024-08-06"
    
    # Create mock event
    mock_event = MagicMock()
    
    # Test handling structured output response
    result = handler._handle_regular_response(mock_response, {"model": "gpt-4o-2024-08-06"}, mock_event)
    
    # Verify event was updated with structured output
    mock_event.update_event.assert_called_once()
    call_args = mock_event.update_event.call_args[1]
    
    assert call_args['is_finished'] == True
    assert call_args['is_successful'] == True
    assert call_args['model'] == "gpt-4o-2024-08-06"
    assert "John" in call_args['result']  # Structured output should be converted to string
    
    print("✓ Structured output response handled correctly")

def test_regular_response_handling():
    """Test that regular responses are still handled correctly"""
    
    # Clear any existing singleton instances
    clear_singletons()
    
    # Create a mock client
    mock_client = MagicMock()
    
    # Create handler instance
    handler = OpenAIHandler(mock_client)
    
    # Create mock response without parsed attribute (regular response)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "This is a regular response"
    # No parsed attribute
    del mock_response.choices[0].message.parsed
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.model = "gpt-4o-mini"
    
    # Create mock event
    mock_event = MagicMock()
    
    # Test handling regular response
    result = handler._handle_regular_response(mock_response, {"model": "gpt-4o-mini"}, mock_event)
    
    # Verify event was updated with regular content
    mock_event.update_event.assert_called_once()
    call_args = mock_event.update_event.call_args[1]
    
    assert call_args['is_finished'] == True
    assert call_args['is_successful'] == True
    assert call_args['model'] == "gpt-4o-mini"
    assert call_args['result'] == "This is a regular response"
    
    print("✓ Regular response handled correctly")

if __name__ == "__main__":
    try:
        test_openai_handler_monkey_patching()
        test_structured_output_response_handling()
        test_regular_response_handling()
        print("\n🎉 All tests passed! OpenAI parse method support has been successfully implemented.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()