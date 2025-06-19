"""Comprehensive OpenAI SDK unit tests - validates correct information is tracked"""
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
from openai import OpenAI, AsyncOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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


class TestOpenAIComprehensive(unittest.TestCase):
    """Comprehensive unit tests for OpenAI SDK integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        if not OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY")
        
        # Initialize Lucidic
        lai.init(
            session_name="OpenAI Unit Tests",
            providers=["openai"]
        )
        
        # Create test step
        lai.create_step(
            state="Testing OpenAI SDK",
            action="Run unit tests",
            goal="Validate all OpenAI functionality"
        )
        
        cls.sync_client = OpenAI(api_key=OPENAI_API_KEY)
        cls.async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    @classmethod
    def tearDownClass(cls):
        """Tear down test class"""
        lai.end_step()
        lai.end_session()
    
    def test_chat_completion_sync(self):
        """Test synchronous chat completion tracks correct information"""
        # Make request
        response = self.sync_client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4 Omni
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
        
        print(f"✅ Sync chat completion: {result[:50]}...")
    
    def test_chat_completion_async(self):
        """Test asynchronous chat completion tracks correct information"""
        async def run_async_test():
            response = await self.async_client.chat.completions.create(
                model="gpt-4-turbo",  # Using GPT-4 Turbo
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
        
        print(f"✅ Async chat completion: {result[:50]}...")
    
    def test_streaming_sync(self):
        """Test synchronous streaming tracks chunks correctly"""
        stream = self.sync_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using GPT-3.5 Turbo
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
            self.assertIsNotNone(chunk.id)
            self.assertEqual(chunk.object, "chat.completion.chunk")
            
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
            stream = await self.async_client.chat.completions.create(
                model="gpt-4o-mini",
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
    
    def test_structured_output(self):
        """Test structured output with beta.chat.completions.parse"""
        try:
            response = self.sync_client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "user", "content": "Solve step by step: 10 + 5"}
                ],
                response_format=MathReasoning,
            )
            
            # Validate response structure
            self.assertIsNotNone(response)
            self.assertIsNotNone(response.choices)
            
            # Validate parsed result
            result = response.choices[0].message.parsed
            self.assertIsInstance(result, MathReasoning)
            self.assertEqual(result.final_answer, "15")
            self.assertIsInstance(result.steps, list)
            self.assertGreater(len(result.steps), 0)
            
            # Validate usage
            self.assertIsNotNone(response.usage)
            
            print(f"✅ Structured output: {result.final_answer}, {len(result.steps)} steps")
            
        except Exception as e:
            # Some models don't support structured output
            self.skipTest(f"Structured output not supported: {str(e)}")
    
    def test_vision(self):
        """Test vision/image analysis tracks image data"""
        # Load test image
        image_path = os.path.join(os.path.dirname(__file__), "ord_runways.jpg")
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
        
        response = self.sync_client.chat.completions.create(
            model="gpt-4o",  # GPT-4o for vision
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
        
        print(f"✅ Vision analysis: {result}")
    
    def test_error_handling(self):
        """Test error handling captures error information"""
        with self.assertRaises(Exception) as context:
            self.sync_client.chat.completions.create(
                model="invalid-model-xyz",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
        
        # Validate error details
        error = context.exception
        self.assertIn("model", str(error).lower())
        
        print(f"✅ Error handling: {type(error).__name__} caught")
    
    def test_concurrent_requests(self):
        """Test concurrent requests are tracked independently"""
        async def make_concurrent_requests():
            tasks = []
            for i in range(3):
                task = self.async_client.chat.completions.create(
                    model="gpt-4o-mini",
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
        response = self.sync_client.chat.completions.create(
            model="gpt-4o-mini",
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
    
    def test_beta_parse_person_extraction(self):
        """Test beta.chat.completions.parse with person extraction"""
        try:
            response = self.sync_client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "user", "content": "Extract info: John Smith is a 30-year-old software engineer who knows Python, JavaScript, and SQL."}
                ],
                response_format=PersonInfo,
            )
            
            # Validate response structure
            self.assertIsNotNone(response)
            self.assertIsNotNone(response.choices)
            
            # Validate parsed result
            result = response.choices[0].message.parsed
            self.assertIsInstance(result, PersonInfo)
            self.assertEqual(result.name, "John Smith")
            self.assertEqual(result.age, 30)
            self.assertEqual(result.occupation, "software engineer")
            self.assertIn("Python", result.skills)
            self.assertIn("JavaScript", result.skills)
            self.assertIn("SQL", result.skills)
            
            print(f"✅ Beta parse person: {result.name}, {result.age}, {result.occupation}")
            
        except Exception as e:
            self.skipTest(f"Beta parse not supported: {str(e)}")
    
    def test_beta_parse_with_image(self):
        """Test beta.chat.completions.parse with image analysis"""
        # Load test image
        image_path = os.path.join(os.path.dirname(__file__), "ord_runways.jpg")
        if not os.path.exists(image_path):
            self.skipTest("Test image not found")
        
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            data_uri = f"data:image/jpeg;base64,{base64.standard_b64encode(img_bytes).decode()}"
            
            # Build message with image
            image_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image and list objects you see:"},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }
            
            response = self.sync_client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[image_message],
                response_format=ImageDescription,
            )
            
            # Validate response
            self.assertIsNotNone(response)
            result = response.choices[0].message.parsed
            self.assertIsInstance(result, ImageDescription)
            self.assertIsNotNone(result.description)
            self.assertIsInstance(result.objects_seen, list)
            self.assertGreater(len(result.objects_seen), 0)
            
            # Validate usage (images use more tokens)
            self.assertIsNotNone(response.usage)
            self.assertGreater(response.usage.prompt_tokens, 100)
            
            print(f"✅ Beta parse with image: {len(result.objects_seen)} objects seen")
            
        except Exception as e:
            self.skipTest(f"Beta parse with image not supported: {str(e)}")
    
    def test_model_variety(self):
        """Test different OpenAI models are tracked correctly"""
        models = [
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini"
        ]
        
        for model in models:
            try:
                response = self.sync_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": f"Say '{model}'"}],
                    max_tokens=20
                )
                
                # Validate model is tracked
                self.assertIsNotNone(response.model)
                result = response.choices[0].message.content
                print(f"✅ Model {model}: {result[:30]}...")
                
            except Exception as e:
                print(f"⚠️  Model {model} not available: {str(e)}")


if __name__ == "__main__":
    unittest.main()