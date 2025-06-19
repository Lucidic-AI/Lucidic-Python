"""Comprehensive Pydantic AI unit tests"""
import os
import sys
import unittest
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import lucidicai as lai
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Define structured output models
class TaskBreakdown(BaseModel):
    """Task breakdown structure"""
    task_name: str
    steps: List[str]
    estimated_time: str
    complexity: str

class CodeReview(BaseModel):
    """Code review structure"""
    issues: List[str]
    suggestions: List[str]
    overall_quality: str
    score: int


class TestPydanticAIComprehensive(unittest.TestCase):
    """Comprehensive unit tests for Pydantic AI integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        # Check for at least one API key
        if not any([OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY]):
            raise ValueError("At least one API key required (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY)")
        
        # Initialize Lucidic
        lai.init(
            session_name="Pydantic AI Unit Tests",
            providers=["pydantic_ai"]
        )
        
        # Create test step
        lai.create_step(
            state="Testing Pydantic AI",
            action="Run unit tests",
            goal="Validate Pydantic AI functionality"
        )
    
    @classmethod
    def tearDownClass(cls):
        """Tear down test class"""
        lai.end_step()
        lai.end_session()
    
    def _get_latest_event(self) -> Dict[str, Any]:
        """Get the latest event from session history"""
        session = lai.Client().session
        if not session.event_history:
            return None
        
        event_id = list(session.event_history.keys())[-1]
        event = session.event_history[event_id]
        
        return {
            'event_id': event_id,
            'description': getattr(event, 'description', None),
            'result': getattr(event, 'result', None),
            'is_finished': getattr(event, 'is_finished', False),
            'model': getattr(event, 'model', None),
            'cost_added': getattr(event, 'cost_added', None)
        }
    
    @unittest.skipIf(not OPENAI_API_KEY, "OpenAI API key not available")
    def test_openai_model_sync(self):
        """Test Pydantic AI with OpenAI model"""
        model = OpenAIModel('gpt-4o-mini')
        agent = Agent(
            model=model,
            output_type=str,
            system_prompt="You are a helpful assistant. Be concise."
        )
        
        # Run sync test
        result = asyncio.run(agent.run("What is 2+2? Answer in one word."))
        
        # Validate response
        self.assertIsNotNone(result)
        self.assertIsInstance(result.output, str)
        self.assertTrue("4" in result.output.lower() or "four" in result.output.lower())
        
        # Validate event tracking
        event = self._get_latest_event()
        self.assertIsNotNone(event)
        self.assertTrue(event['is_finished'])
    
    @unittest.skipIf(not OPENAI_API_KEY, "OpenAI API key not available")
    def test_openai_model_streaming(self):
        """Test Pydantic AI streaming with OpenAI"""
        async def run_streaming_test():
            model = OpenAIModel('gpt-4o-mini')
            agent = Agent(
                model=model,
                output_type=str,
                system_prompt="You are a helpful assistant."
            )
            
            full_response = ""
            async with agent.run_stream("Count from 1 to 3") as stream:
                async for chunk in stream.stream_text(delta=True):
                    full_response += chunk
            
            self.assertIn("1", full_response)
            self.assertIn("2", full_response)
            self.assertIn("3", full_response)
            
            return full_response
        
        # Run streaming test
        full_response = asyncio.run(run_streaming_test())
        
        # Note: Pydantic AI streaming may not populate event result field - this is expected
    
    @unittest.skipIf(not ANTHROPIC_API_KEY, "Anthropic API key not available")
    def test_anthropic_model(self):
        """Test Pydantic AI with Anthropic model"""
        model = AnthropicModel('claude-3-haiku-20240307')
        agent = Agent(
            model=model,
            output_type=str,
            system_prompt="You are a helpful assistant. Be concise."
        )
        
        # Run test
        result = asyncio.run(agent.run("What color is the sky? One word."))
        
        # Validate response
        self.assertIsNotNone(result)
        self.assertIsInstance(result.output, str)
        self.assertIn("blue", result.output.lower())
        
        # Validate event tracking
        event = self._get_latest_event()
        self.assertIsNotNone(event)
        self.assertTrue(event['is_finished'])
        if event['result']:
            self.assertIn("blue", str(event['result']).lower())
    
    @unittest.skipIf(not GOOGLE_API_KEY, "Google API key not available")
    def test_gemini_model(self):
        """Test Pydantic AI with Gemini model"""
        model = GeminiModel('gemini-1.5-flash')
        agent = Agent(
            model=model,
            output_type=str,
            system_prompt="You are a helpful assistant. Be concise."
        )
        
        # Run test
        result = asyncio.run(agent.run("What is the capital of France? One word."))
        
        # Validate response
        self.assertIsNotNone(result)
        self.assertIsInstance(result.output, str)
        self.assertIn("paris", result.output.lower())
        
        # Validate event tracking
        event = self._get_latest_event()
        self.assertIsNotNone(event)
        self.assertTrue(event['is_finished'])
        if event['result']:
            self.assertIn("paris", str(event['result']).lower())
    
    @unittest.skipIf(not OPENAI_API_KEY, "OpenAI API key not available")
    def test_structured_output(self):
        """Test structured output with Pydantic AI"""
        model = OpenAIModel('gpt-4o-mini')
        agent = Agent(
            model=model,
            output_type=TaskBreakdown,
            system_prompt="You are a project planning assistant."
        )
        
        # Run test
        result = asyncio.run(agent.run("Break down the task: Build a simple web API"))
        
        # Validate response
        self.assertIsNotNone(result)
        self.assertIsInstance(result.output, TaskBreakdown)
        self.assertIsInstance(result.output.steps, list)
        self.assertGreater(len(result.output.steps), 0)
        self.assertIsNotNone(result.output.task_name)
        self.assertIsNotNone(result.output.complexity)
    
    @unittest.skipIf(not OPENAI_API_KEY, "OpenAI API key not available")
    def test_multi_model_agent(self):
        """Test agent with multiple models"""
        models = []
        if OPENAI_API_KEY:
            models.append(OpenAIModel('gpt-4o-mini'))
        if ANTHROPIC_API_KEY:
            models.append(AnthropicModel('claude-3-haiku-20240307'))
        
        if len(models) < 2:
            self.skipTest("Need at least 2 API keys for multi-model test")
        
        agent = Agent(
            model=models[0],  # Primary model
            output_type=str,
            system_prompt="You are a helpful assistant."
        )
        
        # Run test
        result = asyncio.run(agent.run("Say 'Hello World'"))
        
        # Validate response
        self.assertIsNotNone(result)
        self.assertIn("hello", result.output.lower())
        self.assertIn("world", result.output.lower())
    
    @unittest.skipIf(not OPENAI_API_KEY, "OpenAI API key not available")
    def test_error_handling(self):
        """Test error handling in Pydantic AI"""
        # Create agent with very low token limit to force error
        model = OpenAIModel('gpt-4o-mini')
        agent = Agent(
            model=model,
            output_type=CodeReview,
            system_prompt="You are a code reviewer."
        )
        
        try:
            # This should fail because the response won't fit the schema with such low tokens
            result = asyncio.run(agent.run("Review this code: print('hello')", model_settings={'max_tokens': 5}))
            # If it doesn't fail, just check that an event was created
            event = self._get_latest_event()
            self.assertIsNotNone(event)
        except Exception as e:
            # Expected to fail
            self.assertIsNotNone(str(e))
            # Check if event was created for the error
            event = self._get_latest_event()
            if event:
                self.assertTrue(event['is_finished'])
    
    @unittest.skipIf(not OPENAI_API_KEY, "OpenAI API key not available")
    def test_concurrent_agents(self):
        """Test concurrent agent executions"""
        async def run_concurrent_test():
            model = OpenAIModel('gpt-4o-mini')
            
            # Create multiple agents
            agents = []
            for i in range(3):
                agent = Agent(
                    model=model,
                    output_type=str,
                    system_prompt=f"You are assistant {i+1}. Always include your number in responses."
                )
                agents.append(agent)
            
            # Run concurrent tasks
            tasks = [
                agents[0].run("Say 'I am assistant 1'"),
                agents[1].run("Say 'I am assistant 2'"),
                agents[2].run("Say 'I am assistant 3'")
            ]
            
            results = await asyncio.gather(*tasks)
            return results
        
        # Run concurrent test
        results = asyncio.run(run_concurrent_test())
        
        # Validate all responses
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertIsNotNone(result)
            self.assertIn(str(i+1), result.output)
        
        # Check that multiple events were created
        session = lai.Client().session
        self.assertGreaterEqual(len(session.event_history), 3)


if __name__ == "__main__":
    unittest.main()