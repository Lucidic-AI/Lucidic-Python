"""Basic OpenAI test with Lucidic initialization to analyze airport runway image"""
import os
import sys
import unittest
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import lucidicai as lai
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class TestOpenAIBasic(unittest.TestCase):
    """Basic test for OpenAI SDK integration with Lucidic"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with Lucidic initialization"""
        if not OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY")
        
        # Initialize Lucidic
        lai.init(
            session_name="Airport Runway Analysis Test",
            providers=["openai"]
        )
        
        # Create test step
        lai.create_step(
            state="Analyzing airport runway image",
            action="Identify airport and count runways",
            goal="Determine airport name and runway count from image"
        )

        print(f"🔍 Step created")
        
        cls.client = OpenAI(api_key=OPENAI_API_KEY)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up Lucidic session"""
        lai.end_step()
        lai.end_session()
    
    def test_airport_runway_analysis(self):
        """Test asking model about airport and runway count in ord_runways.jpg"""
        # First, make a simple text-based call about operating systems
        print("🔍 Making initial OpenAI call about operating systems...")
        
        os_response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "What is the primary difference between macOS and Windows?"}
            ],
            max_tokens=150
        )
        
        # Validate the OS comparison response
        self.assertIsNotNone(os_response)
        os_result = os_response.choices[0].message.content
        self.assertIsNotNone(os_result)
        self.assertIsInstance(os_result, str)
        self.assertGreater(len(os_result), 0)
        
        # Validate usage data
        self.assertIsNotNone(os_response.usage)
        self.assertGreater(os_response.usage.total_tokens, 0)
        
        print(f"✅ OS comparison result:")
        print(f"   {os_result}")
        print(f"   Tokens used: {os_response.usage.total_tokens}")
        
        # Now proceed with the image analysis
        print("\n🖼️ Proceeding with airport runway image analysis...")
        
        # Load test image
        image_path = os.path.join(os.path.dirname(__file__), "../ord_runways.jpg")
        if not os.path.exists(image_path):
            self.skipTest("Test image ord_runways.jpg not found")
        
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        data_uri = f"data:image/jpeg;base64,{base64.standard_b64encode(img_bytes).decode()}"
        
        # Ask about the airport and runway count
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "What airport is shown in this image and how many runways does it have? Please provide the airport name and the exact number of runways you can see."
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": data_uri}
                }
            ]
        }]
        
        response = self.client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for vision capabilities
            messages=messages,
            max_tokens=200
        )
        
        # Validate response
        self.assertIsNotNone(response)
        result = response.choices[0].message.content
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        
        # Validate usage data exists (images use more tokens)
        self.assertIsNotNone(response.usage)
        self.assertGreater(response.usage.prompt_tokens, 100)  # Images use many tokens
        self.assertGreater(response.usage.completion_tokens, 0)
        
        # Validate model info
        self.assertIn("gpt-4o", response.model)
        
        print(f"✅ Airport runway analysis result:")
        print(f"   {result}")
        print(f"   Total tokens used: {response.usage.total_tokens}")
        
        # Basic validation that response mentions airport-related content
        result_lower = result.lower()
        self.assertTrue(
            any(keyword in result_lower for keyword in ["airport", "runway", "runways", "ord", "chicago"]),
            f"Response should mention airport or runway information. Got: {result}"
        )


if __name__ == "__main__":
    unittest.main()
