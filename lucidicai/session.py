from datetime import datetime
from typing import Optional, List
import requests
import json
from .errors import handle_session_response

class Step:
    def __init__(self, session: 'Session', goal: Optional[str] = None, action: Optional[str] = None):
        self.session = session
        self.api_key = session.api_key
        self.session_id = session.session_id
        self.step_id: Optional[str] = None
        self.base_url = "https://analytics.lucidic.ai/api"
        
        self.goal = goal
        self.action = action
        self.is_successful = None
        self.is_finished = None
        self.cost = None
        self.model = None
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        
        self.init_step(goal, action)
        self.session.step_history.append(self)
        
    def init_step(self, goal: Optional[str] = None, action: Optional[str] = None) -> bool:
        request_data = {
            "session_id": self.session_id,
            "current_time": datetime.now().isoformat(),
            "goal": self.goal,
            "action": self.action
        }
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        
        response = requests.post(
            f"{self.base_url}/initstep",
            json=request_data,
            headers=headers
        )
        
        data = handle_session_response(response, required_fields=["step_id"])
        self.step_id = data["step_id"]
        return True

    def update_step(self, goal: Optional[str] = None, action: Optional[str] = None,
                   is_successful: Optional[bool] = None, is_finished: Optional[bool] = None,
                   cost: Optional[float] = None, model: Optional[str] = None) -> bool:
        update_attrs = {k: v for k, v in locals().items() 
                       if k != 'self' and v is not None}
        self.__dict__.update(update_attrs)
        
        request_data = {
            "step_id": self.step_id,
            "current_time": datetime.now().isoformat(),
            "goal": self.goal,
            "action": self.action,
            "is_successful": self.is_successful,
            "is_finished": self.is_finished,
            "cost": self.cost,
            "model": self.model
        }
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        
        response = requests.put(
            f"{self.base_url}/updatestep",
            json=request_data,
            headers=headers
        )
        
        handle_session_response(response)
        return True

    def finish_step(self, is_successful: bool, cost: Optional[float] = None, 
                   model: Optional[str] = None) -> bool:
        self.end_time = datetime.now().isoformat()
        return self.update_step(
            is_finished=True,
            is_successful=is_successful,
            cost=cost,
            model=model
        )

class Session:
    def __init__(self, api_key: str, agent_id: str, session_name: str, 
                 mass_sim_id: Optional[str] = None, task: Optional[str] = None):
        self.api_key = api_key
        self.agent_id = agent_id
        self.session_name = session_name
        self.session_id = None
        self.step_history: List[Step] = []
        self.base_url = "https://analytics.lucidic.ai/api"
        self.starttime = datetime.now().isoformat()
        
        self.mass_sim_id = mass_sim_id
        self.task = task
        self.is_finished = None
        self.is_successful = None
        
        self.init_session()
        
    def init_session(self) -> bool:
        response = requests.post(
            f"{self.base_url}/initsession",
            json={
                "agent_id": self.agent_id,
                "session_name": self.session_name,
                "current_time": datetime.now().isoformat(),
                "mass_sim_id": self.mass_sim_id,
                "task": self.task,
            },
            headers={"Authorization": f"Api-Key {self.api_key}"}
        )
        
        data = handle_session_response(response, required_fields=["session_id"])
        self.session_id = data["session_id"]
        return True
    
    def update_session(self, task: Optional[str] = None, is_finished: Optional[bool] = None,
                      is_successful: Optional[bool] = None) -> bool:
        if not self.session_id:
            raise ValueError("Session ID not set. Call init_session first.")
        
        update_attrs = {k: v for k, v in locals().items() 
                       if k != 'self' and v is not None}
        self.__dict__.update(update_attrs)
            
        response = requests.put(
            f"{self.base_url}/updatesession",
            json={
                "session_id": self.session_id,
                "current_time": datetime.now().isoformat(),
                "task": self.task,
                "is_finished": self.is_finished,
                "is_successful": self.is_successful
            },
            headers={"Authorization": f"Api-Key {self.api_key}"}
        )
        
        handle_session_response(response)
        return True

    def print_step_history(self):
        """Print out the goals and actions of all steps in the session."""
        print("\nSession Step History:")
        print("-" * 50)
        for i, step in enumerate(self.step_history):
            print(f"Step {i}:")
            print(f"Goal: {step.goal}")
            print(f"Action: {step.action}")
            print(f"Status: {'Successful' if step.is_successful else 'Failed' if step.is_successful is False else 'Unknown'}")
            print("-" * 50)

    def finish_session(self, is_successful: bool) -> bool:
        """Finish the session and mark it as successful or failed."""
        self.print_step_history()
        
        self.is_finished = True
        self.is_successful = is_successful
        return self.update_session(is_finished=True, is_successful=is_successful)

    def create_step(self, goal: Optional[str] = None, action: Optional[str] = None) -> Step:
        if not self.session_id:
            raise ValueError("Session ID not set. Call init_session first.")
            
        if isinstance(goal, (list, dict)):
            goal = json.dumps(goal)
                
        return Step(session=self, goal=goal, action=action)