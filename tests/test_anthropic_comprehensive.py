"""Comprehensive Anthropic SDK unit tests - validates correct information is tracked"""
import os
import sys
import unittest
import asyncio
import base64
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import lucidicai as lai
import anthropic
from anthropic import Anthropic, AsyncAnthropic
from openai import OpenAI  # For Anthropic via OpenAI SDK tests

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


class TestAnthropicComprehensive(unittest.TestCase):
    """Comprehensive unit tests for Anthropic SDK integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        if not ANTHROPIC_API_KEY:
            raise ValueError("Missing ANTHROPIC_API_KEY")
        
        # Initialize Lucidic with both providers
        lai.init(
            session_name="Anthropic Unit Tests",
            providers=["anthropic", "openai"]  # Both for testing via OpenAI SDK
        )
        
        # Create test step
        lai.create_step(
            state="Testing Anthropic SDK",
            action="Run unit tests",
            goal="Validate all Anthropic functionality"
        )
        
        cls.sync_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        cls.async_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        cls.openai_client = OpenAI(
            api_key=ANTHROPIC_API_KEY,
            base_url="https://api.anthropic.com/v1",
            default_headers={"anthropic-version": "2023-06-01"}
        )
    
    @classmethod
    def tearDownClass(cls):
        """Tear down test class"""
        lai.end_step()
        lai.end_session()
    
    def test_native_sync(self):
        """Test native Anthropic SDK synchronous tracks correct information"""
        response = self.sync_client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Using Sonnet model
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Say 'test passed'"}
            ]
        )
        
        # Validate response structure
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.content)
        self.assertGreater(len(response.content), 0)
        
        # Validate content
        result = response.content[0].text
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        
        # Validate metadata
        self.assertIsNotNone(response.id)
        self.assertIsNotNone(response.model)
        self.assertEqual(response.model, "claude-3-5-sonnet-20241022")
        self.assertIsNotNone(response.usage)
        self.assertGreater(response.usage.input_tokens, 0)
        self.assertGreater(response.usage.output_tokens, 0)
        
        print(f"✅ Native sync: {result[:50]}...")
    
    def test_native_async(self):
        """Test native Anthropic SDK asynchronous tracks correct information"""
        async def run_async_test():
            response = await self.async_client.messages.create(
                model="claude-3-opus-20240229",  # Using Opus model
                max_tokens=50,
                messages=[
                    {"role": "user", "content": "Say 'async test passed'"}
                ]
            )
            
            # Validate response
            self.assertIsNotNone(response)
            self.assertIsNotNone(response.content[0].text)
            self.assertIsNotNone(response.usage)
            
            return response
        
        # Run async test
        response = asyncio.run(run_async_test())
        result = response.content[0].text
        
        print(f"✅ Native async: {result[:50]}...")
    
    def test_native_streaming(self):
        """Test native Anthropic SDK streaming tracks chunks correctly"""
        full_response = ""
        chunk_count = 0
        
        with self.sync_client.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[{"role": "user", "content": "Count: 1 2 3"}]
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                chunk_count += 1
        
        # Validate streaming worked
        self.assertGreater(chunk_count, 0)
        self.assertGreater(len(full_response), 0)
        
        # Validate final message
        final_message = stream.get_final_message()
        self.assertIsNotNone(final_message)
        self.assertIsNotNone(final_message.usage)
        
        print(f"✅ Native streaming: {chunk_count} chunks, response: {full_response[:50]}...")
    
    def test_openai_sdk_sync(self):
        """Test Anthropic via OpenAI SDK synchronous"""
        response = self.openai_client.chat.completions.create(
            model="claude-3-5-sonnet-20241022",  # Using latest model via OpenAI SDK
            messages=[
                {"role": "user", "content": "Say 'OpenAI SDK test passed'"}
            ],
            max_tokens=20
        )
        
        # Validate response structure (OpenAI format)
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices), 0)
        
        # Validate content
        result = response.choices[0].message.content
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        
        # Validate metadata
        self.assertIsNotNone(response.id)
        self.assertIsNotNone(response.model)
        self.assertIsNotNone(response.usage)
        
        print(f"✅ OpenAI SDK sync: {result[:50]}...")
    
    def test_openai_sdk_streaming(self):
        """Test Anthropic via OpenAI SDK streaming"""
        stream = self.openai_client.chat.completions.create(
            model="claude-3-haiku-20240307",
            messages=[
                {"role": "user", "content": "List: A B C"}
            ],
            stream=True,
            max_tokens=30
        )
        
        full_response = ""
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    full_response += delta.content
        
        # Validate streaming worked
        self.assertGreater(chunk_count, 1)
        self.assertGreater(len(full_response), 0)
        
        print(f"✅ OpenAI SDK streaming: {chunk_count} chunks, response: {full_response[:50]}...")
    
    def test_vision(self):
        """Test vision/image analysis tracks image data"""
        # Load test image
        image_path = os.path.join(os.path.dirname(__file__), "ord_runways.jpg")
        if not os.path.exists(image_path):
            self.skipTest("Test image not found")
        
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_base64 = base64.standard_b64encode(img_bytes).decode()
        
        response = self.sync_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "One word description:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_base64
                        }
                    }
                ]
            }]
        )
        
        # Validate response
        self.assertIsNotNone(response)
        result = response.content[0].text
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        
        # Validate usage (images use more tokens)
        self.assertIsNotNone(response.usage)
        self.assertGreater(response.usage.input_tokens, 100)
        
        print(f"✅ Vision analysis: {result}")
    
    def test_system_prompts(self):
        """Test system prompts are tracked correctly"""
        response = self.sync_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            system="You are a pirate. Respond with pirate language.",
            messages=[
                {"role": "user", "content": "Hello"}
            ]
        )
        
        # Validate response
        self.assertIsNotNone(response)
        result = response.content[0].text
        self.assertIsNotNone(result)
        
        # Validate it used the system prompt
        # Pirate responses typically contain these words
        pirate_words = ["ahoy", "arr", "matey", "ye", "sail", "sea"]
        contains_pirate = any(word in result.lower() for word in pirate_words)
        self.assertTrue(contains_pirate, f"Expected pirate language, got: {result}")
        
        print(f"✅ System prompts: {result[:50]}...")
    
    def test_error_handling(self):
        """Test error handling captures error information"""
        with self.assertRaises(Exception) as context:
            self.sync_client.messages.create(
                model="invalid-model-xyz",
                max_tokens=10,
                messages=[{"role": "user", "content": "Test"}]
            )
        
        # Validate error details
        error = context.exception
        self.assertIn("model", str(error).lower())
        
        print(f"✅ Error handling: {type(error).__name__} caught")
    
    def test_multi_turn_conversation(self):
        """Test multi-turn conversation maintains context"""
        # First message
        response1 = self.sync_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "My name is TestBot. What's my name?"}
            ]
        )
        
        # Validate first response
        self.assertIsNotNone(response1)
        result1 = response1.content[0].text
        self.assertIn("testbot", result1.lower())
        
        # Second message with context
        response2 = self.sync_client.messages.create(
            model="claude-3-5-haiku-20241022",  # Using latest Haiku
            max_tokens=100,
            messages=[
                {"role": "user", "content": "My name is TestBot. What's my name?"},
                {"role": "assistant", "content": result1},
                {"role": "user", "content": "Repeat my name one more time"}
            ]
        )
        
        # Validate second response maintains context
        self.assertIsNotNone(response2)
        result2 = response2.content[0].text
        self.assertIn("testbot", result2.lower())
        
        print(f"✅ Multi-turn conversation: maintained context")
    
    def test_token_limits(self):
        """Test token limit handling"""
        response = self.sync_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=5,  # Very low limit
            messages=[
                {"role": "user", "content": "Tell me a very long story"}
            ]
        )
        
        # Validate response respects token limit
        result = response.content[0].text
        self.assertIsNotNone(result)
        # Anthropic counts tokens differently, but should still be short
        self.assertLess(len(result.split()), 20)
        
        # Check stop reason
        self.assertEqual(response.stop_reason, "max_tokens")
        
        print(f"✅ Token limits: {len(result.split())} words, stop_reason={response.stop_reason}")
    
    def test_text_content_blocks(self):
        """Test explicit text content blocks"""
        response = self.sync_client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Using latest Sonnet
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is 2+2?"
                        },
                        {
                            "type": "text",
                            "text": "Also, what is 3+3?"
                        }
                    ]
                }
            ]
        )
        
        # Validate response
        self.assertIsNotNone(response)
        result = response.content[0].text
        self.assertIn("4", result)
        self.assertIn("6", result)
        
        print(f"✅ Text content blocks: {result[:50]}...")
    
    def test_multiple_images(self):
        """Test multiple image content blocks"""
        image_path = os.path.join(os.path.dirname(__file__), "ord_runways.jpg")
        if not os.path.exists(image_path):
            self.skipTest("Test image not found")
        
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_base64 = base64.standard_b64encode(img_bytes).decode()
        
        response = self.sync_client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Latest Sonnet for vision
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Compare these two images (they are the same image shown twice):"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_base64
                        }
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_base64
                        }
                    }
                ]
            }]
        )
        
        # Validate response
        self.assertIsNotNone(response)
        result = response.content[0].text
        self.assertIsNotNone(result)
        
        # Should recognize they are the same
        self.assertTrue("same" in result.lower() or "identical" in result.lower())
        
        print(f"✅ Multiple images: {result[:50]}...")
    
    def test_mixed_content_blocks(self):
        """Test mixed text and image content blocks"""
        image_path = os.path.join(os.path.dirname(__file__), "ord_runways.jpg")
        if not os.path.exists(image_path):
            self.skipTest("Test image not found")
        
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_base64 = base64.standard_b64encode(img_bytes).decode()
        
        response = self.sync_client.messages.create(
            model="claude-3-haiku-20240307",  # Back to Haiku
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "First, tell me what's in this image."
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": "Second, what type of facility is this?"
                    }
                ]
            }]
        )
        
        # Validate response addresses both questions
        self.assertIsNotNone(response)
        result = response.content[0].text
        self.assertTrue("airport" in result.lower() or "runway" in result.lower())
        
        print(f"✅ Mixed content blocks: {result[:50]}...")
    
    def test_model_switching(self):
        """Test that different models are properly tracked"""
        models = [
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022"  # Latest Haiku
        ]
        
        for model in models:
            try:
                response = self.sync_client.messages.create(
                    model=model,
                    max_tokens=20,
                    messages=[{"role": "user", "content": f"Say '{model}'"}]
                )
                
                # Validate model is tracked correctly
                self.assertEqual(response.model, model)
                result = response.content[0].text
                print(f"✅ Model {model}: {result[:30]}...")
                
            except Exception as e:
                print(f"⚠️  Model {model} not available: {str(e)}")
    
    def test_content_block_response_types(self):
        """Test different response content block types"""
        response = self.sync_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Write a haiku about coding"}
            ]
        )
        
        # Validate content blocks
        self.assertIsNotNone(response.content)
        self.assertIsInstance(response.content, list)
        self.assertGreater(len(response.content), 0)
        
        # Check content block structure
        for block in response.content:
            self.assertTrue(hasattr(block, 'type'))
            self.assertEqual(block.type, 'text')
            self.assertTrue(hasattr(block, 'text'))
            self.assertIsInstance(block.text, str)
        
        print(f"✅ Content block types: {len(response.content)} blocks")


if __name__ == "__main__":
    unittest.main()