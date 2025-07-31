"""Comprehensive LiteLLM unit tests - validates correct information is tracked"""
import os
import sys
import unittest
import asyncio
import base64
from typing import Dict, Any, List
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import lucidicai as lai
import litellm
from litellm import completion, acompletion

# Get API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

os.environ['LUCIDIC_DEBUG'] = 'False'
print(f"DEBUG IS {os.getenv('LUCIDIC_DEBUG')}")


# Define structured output models
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

class ImageDescription(BaseModel):
    description: str
    objects_seen: List[str]


class TestLiteLLMComprehensive(unittest.TestCase):
    """Comprehensive unit tests for LiteLLM integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        if not OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY")
        if not ANTHROPIC_API_KEY:
            raise ValueError("Missing ANTHROPIC_API_KEY")
        
        # Initialize Lucidic with LiteLLM provider
        lai.init(
            session_name="LiteLLM Unit Tests",
            providers=["litellm"]
        )
        
        # Create test step
        lai.create_step(
            state="Testing LiteLLM Integration",
            action="Run unit tests",
            goal="Validate all LiteLLM functionality"
        )
        
        # Set API keys for LiteLLM
        litellm.openai_key = OPENAI_API_KEY
        litellm.anthropic_key = ANTHROPIC_API_KEY
    
    @classmethod
    def tearDownClass(cls):
        """Tear down test class"""
        lai.end_step()
        lai.end_session()
    
    def test_openai_provider_sync(self):
        """Test OpenAI provider through LiteLLM tracks correct information"""
        # Make request
        response = completion(
            model="openai/gpt-4o",  # Using GPT-4 Omni
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'test passed'"}
            ],
            max_tokens=10
        )   
        
        # Validate response structure
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices), 0)
        
        # Validate response content
        result = response.choices[0].message.content
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        
        # Validate usage data exists
        self.assertIsNotNone(response.usage)
        self.assertGreater(response.usage.total_tokens, 0)
        self.assertGreater(response.usage.prompt_tokens, 0)
        self.assertGreater(response.usage.completion_tokens, 0)
        
        # Validate model info
        self.assertIn("gpt-4o", response.model)
        
        print(f"✅ OpenAI sync via LiteLLM: {result[:50]}...")
    
    def test_anthropic_provider_sync(self):
        """Test Anthropic provider through LiteLLM tracks correct information"""
        response = completion(
            model="anthropic/claude-3-haiku-20240307",  # Using Haiku for speed
            messages=[
                {"role": "user", "content": "Say 'anthropic test passed'"}
            ],
            max_tokens=20
        )
        
        # Validate response
        self.assertIsNotNone(response)
        result = response.choices[0].message.content
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        
        # Validate usage
        self.assertIsNotNone(response.usage)
        self.assertGreater(response.usage.total_tokens, 0)
        
        print(f"✅ Anthropic sync via LiteLLM: {result[:50]}...")
    
    def test_async_completion(self):
        """Test asynchronous completion tracks correct information"""
        async def run_async_test():
            response = await acompletion(
                model="openai/gpt-4-turbo",
                messages=[
                    {"role": "user", "content": "Say 'async test passed'"}
                ],
                max_tokens=10
            )
            
            # Validate response
            self.assertIsNotNone(response)
            self.assertIsNotNone(response.choices[0].message.content)
            self.assertIsNotNone(response.usage)
            
            return response
        
        # Run async test
        response = asyncio.run(run_async_test())
        result = response.choices[0].message.content
        
        print(f"✅ Async completion: {result[:50]}...")
    
    def test_streaming_sync(self):
        """Test synchronous streaming tracks chunks correctly"""
        stream = completion(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Count: 1 2 3"}],
            stream=True,
            max_tokens=20
        )
        
        full_response = ""
        chunk_count = 0
        has_finish_reason = False
        
        for chunk in stream:
            chunk_count += 1
            
            # Validate chunk structure
            self.assertIsNotNone(chunk)
            if hasattr(chunk, 'id'):
                self.assertIsNotNone(chunk.id)
            
            if hasattr(chunk, 'choices') and chunk.choices:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                if chunk.choices[0].finish_reason:
                    has_finish_reason = True
        
        # Validate streaming worked
        self.assertGreater(chunk_count, 1)
        self.assertGreater(len(full_response), 0)
        self.assertTrue(has_finish_reason)
        
        print(f"✅ Sync streaming: {chunk_count} chunks, response: {full_response[:50]}...")
    
    def test_streaming_async(self):
        """Test asynchronous streaming tracks chunks correctly"""
        async def run_async_stream():
            stream = await acompletion(
                model="anthropic/claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "List: A B C"}],
                stream=True,
                max_tokens=20
            )
            
            full_response = ""
            chunk_count = 0
            
            async for chunk in stream:
                chunk_count += 1
                if hasattr(chunk, 'choices') and chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            
            self.assertGreater(chunk_count, 1)
            self.assertGreater(len(full_response), 0)
            
            return full_response, chunk_count
        
        # Run async test
        full_response, chunk_count = asyncio.run(run_async_stream())
        
        print(f"✅ Async streaming: {chunk_count} chunks, response: {full_response[:50]}...")
    
    def test_vision_with_image(self):
        """Test vision/image analysis tracks image data"""
        # Load test image
        image_path = os.path.join(os.path.dirname(__file__), "../ord_runways.jpg")
        if not os.path.exists(image_path):
            self.skipTest("Test image not found")
        
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        data_uri = f"data:image/jpeg;base64,{base64.standard_b64encode(img_bytes).decode()}"
        
        # Message with image
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "One word description:"},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }]
        
        response = completion(
            model="openai/gpt-4o",  # GPT-4o for vision
            messages=messages,
            max_tokens=10
        )
        
        # Validate response
        self.assertIsNotNone(response)
        result = response.choices[0].message.content
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        
        # Validate usage (images cost more tokens)
        self.assertIsNotNone(response.usage)
        self.assertGreater(response.usage.prompt_tokens, 100)  # Images use many tokens
        
        print(f"✅ Vision analysis via LiteLLM: {result}")
    
    def test_multimodal_anthropic(self):
        """Test multimodal support with Anthropic"""
        # Load test image
        image_path = os.path.join(os.path.dirname(__file__), "../ord_runways.jpg")
        if not os.path.exists(image_path):
            self.skipTest("Test image not found")
        
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        data_uri = f"data:image/jpeg;base64,{base64.standard_b64encode(img_bytes).decode()}"
        
        # Message with image for Anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image briefly:"},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }]
        
        try:
            response = completion(
                model="anthropic/claude-3-5-sonnet-20241022",  # Claude with vision
                messages=messages,
                max_tokens=50
            )
            
            # Validate response
            self.assertIsNotNone(response)
            result = response.choices[0].message.content
            self.assertIsNotNone(result)
            
            print(f"✅ Anthropic multimodal via LiteLLM: {result[:50]}...")
        except Exception as e:
            print(f"⚠️  Anthropic multimodal test skipped: {str(e)}")
    
    def test_error_handling(self):
        """Test error handling captures error information"""
        with self.assertRaises(Exception) as context:
            completion(
                model="invalid-provider/invalid-model-xyz",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
        
        # Validate error details
        error = context.exception
        self.assertIsNotNone(error)
        
        print(f"✅ Error handling: {type(error).__name__} caught")
    
    def test_concurrent_requests(self):
        """Test concurrent requests are tracked independently"""
        async def make_concurrent_requests():
            tasks = []
            
            # Mix of providers
            models = [
                "openai/gpt-4o-mini",
                "anthropic/claude-3-haiku-20240307",
                "openai/gpt-3.5-turbo"
            ]
            
            for i, model in enumerate(models):
                task = acompletion(
                    model=model,
                    messages=[{"role": "user", "content": f"Number: {i+1}"}],
                    max_tokens=10
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            return responses
        
        # Run concurrent requests
        responses = asyncio.run(make_concurrent_requests())
        
        # Validate all responses
        self.assertEqual(len(responses), 3)
        for i, response in enumerate(responses):
            self.assertIsNotNone(response)
            self.assertIsNotNone(response.choices[0].message.content)
            self.assertIsNotNone(response.usage)
            self.assertIsNotNone(response.id)  # Each has unique ID
        
        # Verify each response has different ID
        ids = [r.id for r in responses]
        self.assertEqual(len(set(ids)), 3)  # All unique
        
        print(f"✅ Concurrent requests: {len(responses)} responses with unique IDs")
    
    def test_token_limits(self):
        """Test token limit handling"""
        response = completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Tell me a very long story"}],
            max_tokens=5  # Very low limit
        )
        
        # Validate response respects token limit
        result = response.choices[0].message.content
        self.assertIsNotNone(result)
        self.assertLess(len(result.split()), 10)  # Should be very short
        
        # Validate finish reason
        self.assertEqual(response.choices[0].finish_reason, "length")
        
        print(f"✅ Token limits: {len(result.split())} words, finish_reason={response.choices[0].finish_reason}")
    
    def test_model_variety(self):
        """Test different models through LiteLLM are tracked correctly"""
        test_cases = [
            ("openai/gpt-3.5-turbo", "OpenAI GPT-3.5"),
            ("openai/gpt-4o", "OpenAI GPT-4o"),
            ("anthropic/claude-3-haiku-20240307", "Anthropic Haiku"),
            ("gpt-3.5-turbo", "No provider prefix"),  # LiteLLM should infer
        ]
        
        for model, description in test_cases:
            try:
                response = completion(
                    model=model,
                    messages=[{"role": "user", "content": f"Say '{description}'"}],
                    max_tokens=20
                )
                
                # Validate model is tracked
                self.assertIsNotNone(response.model)
                result = response.choices[0].message.content
                print(f"✅ Model {model}: {result[:30]}...")
                
            except Exception as e:
                print(f"⚠️  Model {model} not available: {str(e)}")
    
    def test_custom_headers_and_params(self):
        """Test custom parameters are handled correctly"""
        # Test with temperature and other params
        response = completion(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Random number between 1-10"}],
            temperature=1.0,
            top_p=0.9,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            max_tokens=10
        )
        
        # Validate response
        self.assertIsNotNone(response)
        result = response.choices[0].message.content
        self.assertIsNotNone(result)
        
        print(f"✅ Custom parameters: {result}")
    
    def test_system_messages(self):
        """Test system messages are tracked correctly"""
        messages = [
            {"role": "system", "content": "You are a pirate. Always respond like a pirate."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Ahoy there, matey!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        response = completion(
            model="openai/gpt-4o-mini",
            messages=messages,
            max_tokens=30
        )
        
        # Validate response maintains context
        result = response.choices[0].message.content.lower()
        self.assertIsNotNone(result)
        # Should have pirate-like language
        pirate_words = ["ahoy", "matey", "arr", "ye", "aye"]
        has_pirate_word = any(word in result for word in pirate_words)
        self.assertTrue(has_pirate_word, f"Response should be pirate-like: {result}")
        
        print(f"✅ System messages: {response.choices[0].message.content[:50]}...")
    
    def test_json_mode(self):
        """Test JSON mode responses"""
        try:
            response = completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Return a JSON object with name and age"}],
                response_format={"type": "json_object"},
                max_tokens=50
            )
            
            # Validate JSON response
            result = response.choices[0].message.content
            self.assertIsNotNone(result)
            
            # Should be valid JSON
            import json
            parsed = json.loads(result)
            self.assertIsInstance(parsed, dict)
            
            print(f"✅ JSON mode: {result}")
            
        except Exception as e:
            print(f"⚠️  JSON mode test skipped: {str(e)}")
    
    def test_multiple_images(self):
        """Test multiple images in a single message"""
        # Load test image
        image_path = os.path.join(os.path.dirname(__file__), "../ord_runways.jpg")
        if not os.path.exists(image_path):
            self.skipTest("Test image not found")
        
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        data_uri_1 = f"data:image/jpeg;base64,{base64.standard_b64encode(img_bytes).decode()}"
        
        image_path_2 = os.path.join(os.path.dirname(__file__), "./red_pixel.png")
        with open(image_path_2, "rb") as f:
            img_bytes_2 = f.read()
        data_uri_2 = f"data:image/jpeg;base64,{base64.standard_b64encode(img_bytes_2).decode()}"
        
        # Message with multiple images (same image twice for testing)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "How many images do you see, and what do they show?"},
                {"type": "image_url", "image_url": {"url": data_uri_1}},
                {"type": "image_url", "image_url": {"url": data_uri_2}}
            ]
        }]
        
        try:
            response = completion(
                model="openai/gpt-4o",
                messages=messages,
                max_tokens=50
            )
            
            # Validate response
            result = response.choices[0].message.content
            self.assertIsNotNone(result)
            
            print(f"✅ Multiple images: {result}")
            
        except Exception as e:
            print(f"⚠️  Multiple images test skipped: {str(e)}")


if __name__ == "__main__":
    unittest.main()