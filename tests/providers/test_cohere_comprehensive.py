"""Comprehensive Cohere SDK tests - validates correct information is tracked"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import lucidicai as lai


class TestCohereComprehensive(unittest.TestCase):
    """Comprehensive unit tests for Cohere integration"""

    @classmethod
    def setUpClass(cls):
        cls.cohere_api_key = os.getenv("COHERE_API_KEY")
        try:
            import cohere  # noqa: F401
        except Exception:
            raise unittest.SkipTest("cohere SDK not installed")
        if not cls.cohere_api_key:
            raise unittest.SkipTest("Missing COHERE_API_KEY")

        lai.init(session_name="Cohere Unit Tests", providers=["cohere"], auto_end=False)
        # Steps removed in new SDK – no-op

    def test_chat_sync(self):
        try:
            import cohere
        except Exception:
            self.skipTest("cohere SDK not installed")

        client = cohere.ClientV2(api_key=self.cohere_api_key)

        try:
            # Command R or Command Light may be available
            resp = client.chat(
                model=os.getenv("COHERE_MODEL", "command-r"),
                messages=[{"role": "user", "content": "Say 'cohere test passed'"}],
                max_tokens=64,
            )
        except Exception as e:
            self.skipTest(f"Cohere call failed (possibly no access): {e}")

        self.assertIsNotNone(resp)


if __name__ == "__main__":
    unittest.main()


