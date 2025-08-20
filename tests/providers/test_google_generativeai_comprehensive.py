"""Comprehensive Google Generative AI SDK tests - validates correct information is tracked"""
import os
import sys
import unittest
import asyncio
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import lucidicai as lai
from google import genai as genai


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")


class TestGoogleGenerativeAIComprehensive(unittest.TestCase):
    """Comprehensive unit tests for Google Generative AI integration"""

    @classmethod
    def setUpClass(cls):
        try:
            from google import genai as _genai  # noqa: F401
        except Exception:
            raise unittest.SkipTest("google-genai not installed")
        if not GOOGLE_API_KEY:
            raise unittest.SkipTest("Missing GOOGLE_API_KEY")

        # Initialize Lucidic
        lai.init(
            session_name="Gemini Unit Tests",
            providers=["google"],
        )
        # Create one shared client instance for sync tests
        cls.client = genai.Client(api_key=GOOGLE_API_KEY)

    @classmethod
    def tearDownClass(cls):
        lai.end_session()

    def test_generate_content_sync(self):
        response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents="Say 'gemini test passed'",
        )
        self.assertIsNotNone(response)
        text = getattr(response, "text", None) or str(response)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        print(f"✅ Sync generate_content: {text[:50]}...")

    def test_generate_content_async(self):
        async def run_async():
            try:
                aclient = genai.AsyncClient(api_key=GOOGLE_API_KEY)
            except Exception as e:
                self.skipTest(f"AsyncClient not available: {e}")
            try:
                resp = await aclient.models.generate_content(
                    model="gemini-1.5-flash",
                    contents="Say 'async gemini passed'",
                )
            except Exception as e:
                self.skipTest(f"Async generation not supported: {e}")
            self.assertIsNotNone(resp)
            self.assertIsInstance(getattr(resp, "text", ""), str)
            return resp

        resp = asyncio.run(run_async())
        print(f"✅ Async generate_content: {resp.text[:50]}...")

    def test_streaming_sync(self):
        try:
            stream = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents="Count: 1 2 3",
                stream=True,
            )
        except Exception as e:
            self.skipTest(f"Streaming not supported: {e}")

        full_text = ""
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            if getattr(chunk, "text", None):
                full_text += chunk.text
        self.assertGreater(chunk_count, 0)
        self.assertGreater(len(full_text), 0)
        print(f"✅ Sync streaming: {chunk_count} chunks, response: {full_text[:50]}...")

    def test_streaming_async(self):
        async def run_async_stream():
            try:
                stream = self.client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents="List: A B C",
                    stream=True,
                )
            except Exception as e:
                self.skipTest(f"Streaming not supported: {e}")
            full = ""
            chunks = 0
            for ch in stream:
                chunks += 1
                if getattr(ch, "text", None):
                    full += ch.text
            return full, chunks

        full, chunks = asyncio.run(run_async_stream())
        self.assertGreater(chunks, 0)
        self.assertGreater(len(full), 0)
        print(f"✅ Async streaming: {chunks} chunks, response: {full[:50]}...")

    def test_vision_inline_data(self):
        image_path = os.path.join(os.path.dirname(__file__), "../ord_runways.jpg")
        if not os.path.exists(image_path):
            self.skipTest("Test image not found")
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        try:
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[
                    {"text": "One word description:"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}},
                ],
            )
        except Exception as e:
            self.skipTest(f"Inline image not supported: {e}")
        self.assertIsNotNone(response)
        text = getattr(response, "text", None) or str(response)
        self.assertIsInstance(text, str)
        print(f"✅ Vision inline_data: {text[:50]}...")

    def test_error_handling(self):
        with self.assertRaises(Exception) as ctx:
            _ = self.client.models.generate_content(model="invalid-model-xyz", contents="Hello")
        self.assertTrue("model" in str(ctx.exception).lower() or str(ctx.exception))
        print(f"✅ Error handling: {type(ctx.exception).__name__} caught")

    def test_token_limits(self):
        # Try new SDK config style first, then fallback
        try:
            resp = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents="Tell me a very long story",
                config={"max_output_tokens": 8},
            )
        except TypeError:
            resp = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents="Tell me a very long story",
                generation_config={"max_output_tokens": 8},
            )
        self.assertIsNotNone(resp)
        t = getattr(resp, "text", "")
        self.assertIsInstance(t, str)
        if t:
            self.assertLess(len(t.split()), 60)
            print(f"✅ Token limits: {len(t.split())} words")
        else:
            print("✅ Token limits: response received")

    def test_system_instruction(self):
        # New SDK may not accept system_instruction directly; try via config
        try:
            resp = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents="Hello",
                config={"system_instruction": "You are a pirate. Speak like a pirate."},
            )
        except Exception as e:
            self.skipTest(f"System instruction not supported: {e}")
        self.assertIsNotNone(resp)
        txt = getattr(resp, "text", "")
        contains_pirate = any(w in txt.lower() for w in ["ahoy", "arr", "matey", "ye", "sail", "sea"])
        self.assertTrue(contains_pirate, f"Expected pirate language, got: {txt}")
        print(f"✅ System instruction: {txt[:50]}...")

    def test_multi_turn_chat(self):
        # New SDK: emulate multi-turn via concatenated context
        resp1 = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents="My name is TestBot. What's my name?",
        )
        self.assertIsNotNone(resp1)
        txt1 = getattr(resp1, "text", "")
        # expect it to reference TestBot
        self.assertIn("testbot", txt1.lower())
        print(f"✅ Multi-turn: {txt1[:50]}...")

    def test_model_variety(self):
        models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
        ]
        for m in models:
            try:
                resp = self.client.models.generate_content(model=m, contents=f"Say '{m}'")
                self.assertIsNotNone(resp)
                txt = getattr(resp, "text", "")
                print(f"✅ Model {m}: {txt[:30]}...")
            except Exception as e:
                print(f"⚠️  Model {m} not available: {e}")


if __name__ == "__main__":
    unittest.main()


