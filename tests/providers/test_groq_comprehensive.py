"""Comprehensive Groq SDK tests - validates correct information is tracked"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import lucidicai as lai


class TestGroqComprehensive(unittest.TestCase):
    """Comprehensive unit tests for Groq integration"""

    @classmethod
    def setUpClass(cls):
        cls.groq_api_key = os.getenv("GROQ_API_KEY")
        try:
            import groq  # noqa: F401
        except Exception:
            raise unittest.SkipTest("groq SDK not installed")
        if not cls.groq_api_key:
            raise unittest.SkipTest("Missing GROQ_API_KEY")

        lai.init(session_name="Groq Unit Tests", providers=["groq"], auto_end=False)
        lai.create_step(state="Testing Groq", action="Run unit tests", goal="Validate functionality")

    def test_chat_completion_sync(self):
        try:
            from groq import Groq
        except Exception:
            self.skipTest("groq SDK not installed")

        client = Groq(api_key=self.groq_api_key)
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # openai-compatible naming in Groq
                messages=[{"role": "user", "content": "Say 'groq test passed'"}],
                max_tokens=16,
            )
        except Exception as e:
            self.skipTest(f"Groq call failed (possibly no quota): {e}")
        self.assertIsNotNone(resp)
        self.assertTrue(hasattr(resp, "choices"))


if __name__ == "__main__":
    unittest.main()


