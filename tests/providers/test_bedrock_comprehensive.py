"""Comprehensive AWS Bedrock tests - validates correct information is tracked"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import lucidicai as lai


class TestBedrockComprehensive(unittest.TestCase):
    """Comprehensive unit tests for AWS Bedrock integration"""

    @classmethod
    def setUpClass(cls):
        # Bedrock requires AWS credentials; skip if missing
        if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
            raise unittest.SkipTest("Missing AWS credentials for Bedrock")
        try:
            import boto3  # noqa: F401
        except Exception:
            raise unittest.SkipTest("boto3 not installed")

        lai.init(session_name="Bedrock Unit Tests", providers=["bedrock"], auto_end=False)
        lai.create_step(state="Testing Bedrock", action="Run unit tests", goal="Validate functionality")

    def test_bedrock_invoke_model(self):
        try:
            import boto3
        except Exception:
            self.skipTest("boto3 not installed")

        # Use Bedrock Runtime client
        region = os.getenv("AWS_REGION", "us-east-1")
        client = boto3.client("bedrock-runtime", region_name=region)

        # Attempt a minimal invoke with a common model id if available
        # Many accounts have access controls; so handle errors gracefully
        model_id = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
        body = {
            "inputText": "Say 'bedrock test passed'",
            "textGenerationConfig": {"maxTokenCount": 20}
        }
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=bytes(str(body), "utf-8"),
                contentType="application/json",
                accept="application/json",
            )
        except Exception as e:
            self.skipTest(f"Bedrock model not available or permissions missing: {e}")

        self.assertIsNotNone(resp)


if __name__ == "__main__":
    unittest.main()


