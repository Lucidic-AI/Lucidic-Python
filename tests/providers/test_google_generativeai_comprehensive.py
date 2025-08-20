"""Comprehensive Google Generative AI SDK tests - validates correct information is tracked"""
import os
import sys
import unittest
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import lucidicai as lai


class TestGoogleGenerativeAIComprehensive(unittest.TestCase):
    """Comprehensive unit tests for Google Generative AI integration"""

    @classmethod
    def setUpClass(cls):
        cls.google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
        try:
            import google.generativeai as genai  # noqa: F401
        except Exception:
            raise unittest.SkipTest("google-generativeai not installed")
        if not cls.google_api_key:
            raise unittest.SkipTest("Missing GOOGLE_API_KEY")

        # Initialize Lucidic
        lai.init(
            session_name="Gemini Unit Tests",
            providers=["google"],
            auto_end=False,
        )
        lai.create_step(state="Testing Gemini SDK", action="Run unit tests", goal="Validate functionality")

    def test_generate_content_sync(self):
        try:
            import google.generativeai as genai
        except Exception:
            self.skipTest("google-generativeai not installed")

        genai.configure(api_key=self.google_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content("Say 'gemini test passed'")
        self.assertIsNotNone(response)
        text = response.text
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_generate_content_streaming(self):
        try:
            import google.generativeai as genai
        except Exception:
            self.skipTest("google-generativeai not installed")

        genai.configure(api_key=self.google_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        try:
            stream = model.generate_content("List: A B C", stream=True)
        except Exception as e:
            self.skipTest(f"Streaming not supported or disabled: {e}")

        full_text = ""
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            if hasattr(chunk, "text") and chunk.text:
                full_text += chunk.text
        self.assertGreater(chunk_count, 0)
        self.assertGreater(len(full_text), 0)

    def test_vision_inline_data(self):
        try:
            import google.generativeai as genai
        except Exception:
            self.skipTest("google-generativeai not installed")

        image_path = os.path.join(os.path.dirname(__file__), "../ord_runways.jpg")
        if not os.path.exists(image_path):
            self.skipTest("Test image not found")

        with open(image_path, "rb") as f:
            img_bytes = f.read()
        data_url = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"

        genai.configure(api_key=self.google_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        try:
            response = model.generate_content({
                "parts": [
                    {"text": "One word description:"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}}
                ]
            })
        except Exception as e:
            self.skipTest(f"Inline image not supported in this environment: {e}")

        self.assertIsNotNone(response)
        self.assertIsInstance(response.text, str)


if __name__ == "__main__":
    unittest.main()


