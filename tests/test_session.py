import unittest
from unittest.mock import patch, Mock, MagicMock
import json
from datetime import datetime
from lucidicai import (
    Session,
    Step,
    Event,
    Action,
    State,
    APIKeyVerificationError
)

class TestSessionHierarchy(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.agent_id = "test_agent"
        self.session_name = "test_session"
        
        # Mock the requests module for each class
        self.session_patcher = patch('lucidicai.session.requests')
        self.step_patcher = patch('lucidicai.step.requests')
        self.event_patcher = patch('lucidicai.event.requests')
        
        self.mock_session_requests = self.session_patcher.start()
        self.mock_step_requests = self.step_patcher.start()
        self.mock_event_requests = self.event_patcher.start()
        
        # Setup default mock responses
        self.mock_session_response = Mock()
        self.mock_session_response.json.return_value = {"session_id": "test_session_id"}
        self.mock_session_response.status_code = 200
        
        self.mock_step_response = Mock()
        self.mock_step_response.json.return_value = {"step_id": "test_step_id"}
        self.mock_step_response.status_code = 200
        
        self.mock_event_response = Mock()
        self.mock_event_response.json.return_value = {"event_id": "test_event_id"}
        self.mock_event_response.status_code = 200
        
        # Configure mock requests to return our mock responses
        self.mock_session_requests.post.side_effect = self._mock_session_handler
        self.mock_session_requests.put.side_effect = self._mock_session_handler
        
        self.mock_step_requests.post.side_effect = self._mock_step_handler
        self.mock_step_requests.put.side_effect = self._mock_step_handler
        
        self.mock_event_requests.post.side_effect = self._mock_event_handler
        self.mock_event_requests.put.side_effect = self._mock_event_handler

    def tearDown(self):
        self.session_patcher.stop()
        self.step_patcher.stop()
        self.event_patcher.stop()

    def _mock_session_handler(self, url, **kwargs):
        if 'initsession' in url:
            return self.mock_session_response
        return Mock(status_code=200)

    def _mock_step_handler(self, url, **kwargs):
        if 'initstep' in url:
            return self.mock_step_response
        return Mock(status_code=200)

    def _mock_event_handler(self, url, **kwargs):
        if 'initevent' in url:
            return self.mock_event_response
        return Mock(status_code=200)

    def test_session_initialization(self):
        """Test basic session initialization"""
        session = Session(self.api_key, self.agent_id, self.session_name)
        
        self.assertEqual(session.session_id, "test_session_id")
        self.assertEqual(len(session.step_history), 0)
        self.assertIsNone(session.active_step)
        self.assertIsNotNone(session.starttime)

    def test_step_creation_and_active_step(self):
        """Test creating steps and active step management"""
        session = Session(self.api_key, self.agent_id, self.session_name)
        
        # Create first step
        step1 = session.create_step(state="Initial state", action="Initial action", goal="Goal 1")
        self.assertEqual(session.active_step, step1)
        self.assertEqual(str(step1.state), "Initial state")
        self.assertEqual(str(step1.action), "Initial action")
        
        # Cannot create second step while first is active
        with self.assertRaises(ValueError):
            session.create_step(state="State 2", goal="Goal 2")
        
        # Finish first step and create second
        session.finish_step(is_successful=True, state="Final state")
        self.assertIsNone(session.active_step)
        
        step2 = session.create_step(state="State 2", goal="Goal 2")
        self.assertEqual(session.active_step, step2)
        self.assertEqual(len(session.step_history), 2)

    def test_event_management_in_active_step(self):
        """Test event creation and management within active step"""
        session = Session(self.api_key, self.agent_id, self.session_name)
        
        # Cannot create event without active step
        with self.assertRaises(ValueError):
            session.create_event(description="Test description")
        
        # Create step and first event
        step = session.create_step(state="Initial state")
        event1 = session.create_event(description="Description 1", result="Result 1")
        
        # Cannot create second event before finishing first
        with self.assertRaises(ValueError):
            session.create_event(description="Description 2", result="Result 2")
        
        # Finish first event and create second
        session.end_event(is_successful=True)
        event2 = session.create_event(description="Description 2", result="Result 2")
        session.end_event(is_successful=True)
        
        self.assertEqual(len(step.event_history), 2)
        self.assertTrue(step.event_history[0].is_finished)
        self.assertTrue(step.event_history[1].is_finished)
        
        # Cannot create event in finished step
        session.finish_step(is_successful=True)
        with self.assertRaises(ValueError):
            session.create_event(description="Should fail")

    def test_state_and_action_management(self):
        """Test state and action updates within steps"""
        session = Session(self.api_key, self.agent_id, self.session_name)
        step = session.create_step(state="Initial state")
        
        # Test state updates
        step.set_state("New state")
        self.assertEqual(str(step.state), "New state")
        
        # Test action updates
        step.set_action("New action")
        self.assertEqual(str(step.action), "New action")
        
        # Test final state/action in finish_step
        session.finish_step(
            is_successful=True,
            state="Final state",
            action="Final action"
        )
        self.assertEqual(str(step.state), "Final state")
        self.assertEqual(str(step.action), "Final action")

    def test_session_completion(self):
        """Test session completion with active step checks"""
        session = Session(self.api_key, self.agent_id, self.session_name)
        
        # Create and finish first step
        step1 = session.create_step(state="State 1")
        session.finish_step(is_successful=True)
        
        # Create second step
        step2 = session.create_step(state="State 2")
        
        # Cannot finish session with active step
        with self.assertRaises(ValueError):
            session.finish_session(is_successful=True)
        
        # Finish step and then session
        session.finish_step(is_successful=True)
        session.finish_session(is_successful=True)
        
        self.assertTrue(session.is_finished)
        self.assertTrue(session.is_successful)

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test session with invalid API key
        self.mock_session_response.status_code = 401
        self.mock_session_response.json.return_value = {"error": "Invalid API key"}
        
        with self.assertRaises(Exception):
            Session(self.api_key, self.agent_id, self.session_name)
        
        # Reset mock for subsequent tests
        self.mock_session_response.status_code = 200
        self.mock_session_response.json.return_value = {"session_id": "test_session_id"}
        
        session = Session(self.api_key, self.agent_id, self.session_name)
        step = session.create_step(state="Initial state")
        
        # Test ending non-existent event
        with self.assertRaises(ValueError):
            session.end_event(is_successful=True)
        
        # Create and finish an event
        event = session.create_event(description="Test event")
        session.end_event(is_successful=True)
        
        # Test ending already finished event
        with self.assertRaises(ValueError):
            session.end_event(is_successful=True)

    def test_complex_workflow(self):
        """Test a complex workflow with multiple steps and events"""
        session = Session(self.api_key, self.agent_id, self.session_name)
        
        # Step 1: Analysis
        step1 = session.create_step(
            state="Starting analysis",
            action="Initialize analysis",
            goal="Data analysis"
        )
        
        session.create_event(description="Loading data", result="Data loaded")
        session.end_event(is_successful=True)
        
        session.create_event(description="Processing data", result="Processing complete")
        session.end_event(is_successful=True)
        
        session.finish_step(
            is_successful=True,
            state="Analysis complete",
            action="Finalize analysis"
        )
        
        # Step 2: Results
        step2 = session.create_step(
            state="Processing results",
            action="Initialize processing",
            goal="Generate results"
        )
        
        session.create_event(description="Generating report", result="Report generated")
        session.end_event(is_successful=True)
        
        session.finish_step(
            is_successful=True,
            state="Results ready",
            action="Complete processing"
        )
        
        # Verify workflow
        self.assertEqual(len(session.step_history), 2)
        self.assertEqual(len(session.step_history[0].event_history), 2)
        self.assertEqual(len(session.step_history[1].event_history), 1)
        self.assertEqual(str(session.step_history[0].state), "Analysis complete")
        self.assertEqual(str(session.step_history[1].state), "Results ready")

    def test_step_state_transitions(self):
        """Test step state transitions and validation"""
        session = Session(self.api_key, self.agent_id, self.session_name)
        
        # Test default state
        step = session.create_step()
        self.assertEqual(str(step.state), "state not provided")
        
        # Test state transitions with events
        step.set_state("Processing")
        event1 = session.create_event(description="Start processing")
        session.end_event(is_successful=True)
        
        step.set_state("Analyzing")
        event2 = session.create_event(description="Start analysis")
        session.end_event(is_successful=True)
        
        # Verify state history through events
        self.assertEqual(len(step.event_history), 2)
        session.finish_step(is_successful=True, state="Complete")
        self.assertEqual(str(step.state), "Complete")

    def test_nested_step_relationships(self):
        """Test relationships between steps and their events"""
        session = Session(self.api_key, self.agent_id, self.session_name)
        
        # Create parent step
        parent_step = session.create_step(
            state="Parent init",
            action="Parent action",
            goal="Parent goal"
        )
        
        # Add events to parent
        session.create_event(description="Parent event 1")
        session.end_event(is_successful=True)
        
        # Finish parent and create child
        session.finish_step(is_successful=True)
        
        child_step = session.create_step(
            state="Child init",
            action="Child action",
            goal="Child goal"
        )
        
        # Add events to child
        session.create_event(description="Child event 1")
        session.end_event(is_successful=True)
        
        # Verify relationships
        self.assertEqual(len(session.step_history), 2)
        self.assertEqual(session.step_history[0], parent_step)
        self.assertEqual(session.step_history[1], child_step)
        self.assertEqual(len(parent_step.event_history), 1)
        self.assertEqual(len(child_step.event_history), 1)

    def test_concurrent_events_handling(self):
        """Test handling of multiple events within a step"""
        session = Session(self.api_key, self.agent_id, self.session_name)
        step = session.create_step(state="Processing")
        
        # Create multiple events
        events = []
        for i in range(3):
            event = session.create_event(
                description=f"Event {i}",
                result=f"Result {i}"
            )
            events.append(event)
            if i < 2:  # Leave last event unfinished
                session.end_event(is_successful=True)
        
        # Verify event states
        self.assertTrue(events[0].is_finished)
        self.assertTrue(events[1].is_finished)
        self.assertFalse(events[2].is_finished)
        
        # Try to finish step with unfinished event
        with self.assertRaises(ValueError):
            session.finish_step(is_successful=True)
        
        # Finish last event and then step
        session.end_event(is_successful=True)
        session.finish_step(is_successful=True)

    def test_step_cost_tracking(self):
        """Test cost tracking across events and steps"""
        session = Session(self.api_key, self.agent_id, self.session_name)
        step = session.create_step(state="Processing")
        
        # Add events with costs
        session.create_event(description="Event 1")
        session.end_event(is_successful=True, cost_added=0.1, model="gpt-4")
        
        session.create_event(description="Event 2")
        session.end_event(is_successful=True, cost_added=0.2, model="gpt-3.5")
        
        # Verify step costs
        self.assertAlmostEqual(step.cost_added, 0.3, places=6)
        
        # Create another step with costs
        session.finish_step(is_successful=True)
        step2 = session.create_step(state="More processing")
        
        session.create_event(description="Event 3")
        session.end_event(is_successful=True, cost_added=0.4, model="gpt-4")
        
        session.finish_step(is_successful=True)
        
        # Verify total session costs
        total_cost = sum(step.cost_added for step in session.step_history if step.cost_added)
        self.assertAlmostEqual(total_cost, 0.7, places=6)

    def test_invalid_state_transitions(self):
        """Test invalid state transitions and error handling"""
        session = Session(self.api_key, self.agent_id, self.session_name)
        
        # Cannot create event before step
        with self.assertRaises(ValueError):
            session.create_event(description="Invalid event")
        
        # Create and finish a step
        step = session.create_step(state="Initial")
        session.finish_step(is_successful=True)
        
        # Cannot modify finished step
        with self.assertRaises(ValueError):
            step.set_state("New state")
        
        with self.assertRaises(ValueError):
            step.set_action("New action")
        
        with self.assertRaises(ValueError):
            session.create_event(description="Invalid event")
        
        # Cannot finish already finished step
        with self.assertRaises(ValueError):
            session.finish_step(is_successful=True)

    def test_session_rollback(self):
        """Test session state handling during errors"""
        session = Session(self.api_key, self.agent_id, self.session_name)
        
        # Create step and events
        step = session.create_step(state="Processing")
        session.create_event(description="Event 1")
        session.end_event(is_successful=True)
        
        # Simulate API error during event creation
        self.mock_event_response.status_code = 500
        self.mock_event_response.json.return_value = {"error": "Internal server error"}
        
        with self.assertRaises(Exception):
            session.create_event(description="Failed event")
        
        # Verify session state remains valid
        self.assertEqual(len(step.event_history), 1)
        self.assertEqual(session.active_step, step)
        
        # Reset mock
        self.mock_event_response.status_code = 200
        self.mock_event_response.json.return_value = {"event_id": "test_event_id"}
        
        # Can continue with valid operations
        session.create_event(description="Event 2")
        session.end_event(is_successful=True)
        session.finish_step(is_successful=True)
        
        self.assertEqual(len(step.event_history), 2)
        self.assertIsNone(session.active_step)

if __name__ == '__main__':
    unittest.main()
