"""Session management for the Lucidic API"""
from datetime import datetime
from typing import Optional, List
from PIL import Image
import requests
import base64
from .errors import handle_session_response
from .action import Action
from .state import State
from .step import Step
from .event import Event
import io


class Session:
    def __init__(self, api_key: str, agent_id: str, session_name: str, 
                 mass_sim_id: Optional[str] = None, task: Optional[str] = None):
        self.api_key = api_key
        self.agent_id = agent_id
        self.session_name = session_name
        self.mass_sim_id = mass_sim_id
        self.task = task
        self.session_id = None
        self.step_history: List[Step] = []
        self._active_step: Optional[Step] = None
        self.base_url = "https://analytics.lucidic.ai/api"
        self.starttime = datetime.now().isoformat()
        
        self.is_finished = False
        self.is_successful = None
        
        self.init_session()

    def init_session(self) -> bool:
        request_data = {
            "agent_id": self.agent_id,
            "session_name": self.session_name,
            "current_time": datetime.now().isoformat(),
            "task": self.task,
            "mass_sim_id": self.mass_sim_id
        }
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        
        response = requests.post(
            f"{self.base_url}/initsession",
            headers=headers,
            json=request_data
        )
        
        data = handle_session_response(response, required_fields=["session_id"])
        self.session_id = data["session_id"]
        return True

    @property   
    def active_step(self) -> Optional[Step]:
        return self._active_step
    
    def update_session(self, is_finished: Optional[bool] = None,
                      is_successful: Optional[bool] = None, task: Optional[str] = None) -> bool:

        update_attrs = {k: v for k, v in locals().items() 
                       if k != 'self' and v is not None}
        self.__dict__.update(update_attrs)
            
        request_data = {
            "session_id": self.session_id,
            "current_time": datetime.now().isoformat(),
            "is_finished": self.is_finished,
            "is_successful": self.is_successful, 
            "task": self.task
        }
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        
        response = requests.put(
            f"{self.base_url}/updatesession",
            headers=headers,
            json=request_data
        )
        
        handle_session_response(response)
        return True

    def create_step(self, state: Optional[str] = None, action: Optional[str] = None, goal: Optional[str] = None) -> Step:
        if not self.session_id:
            raise ValueError("Session ID not set. Call init_session first.")
            
        if self._active_step:
            raise ValueError("Cannot create new step while another step is active. Please finish current step first.")
        
        state = state or "state not provided"
        
        step = Step(
            session_id=self.session_id,
            api_key=self.api_key,
            state=state,
            action=action,
            goal=goal
        )
        self._active_step = step
        self.step_history.append(step)
        return step

    def update_step(self, is_successful: Optional[bool] = None, state: Optional[str] = None, 
                   action: Optional[str] = None, goal: Optional[str] = None, 
                   is_finished: Optional[bool] = None, cost_added: Optional[float] = None,
                   model: Optional[str] = None) -> None:
        if not self._active_step:
            raise ValueError("No active step to update")
        
        self._active_step.update_step(
            is_successful=is_successful,
            state=state,
            action=action,
            goal=goal,
            is_finished=is_finished,
            cost_added=cost_added,
            model=model
        )

    def finish_step(self, is_successful: bool, state: Optional[str] = None, 
                   action: Optional[str] = None, screenshot = None) -> None:
        if not self._active_step:
            raise ValueError("No active step to finish")
        if screenshot is not None:
            presigned_url, bucket_name, object_key = self.get_presigned_url(self._active_step.step_id)
            self.upload_image_to_s3(presigned_url, screenshot)

        self._active_step.finish_step(
            is_successful=is_successful,
            final_state=state,
            final_action=action,
            screenshot=screenshot,
        )
        self._active_step = None

    def create_event(self, description: Optional[str] = None, result: Optional[str] = None) -> Event:
        if not self._active_step:
            raise ValueError("Cannot create event without an active step. Call create_step first.")
        
        return self._active_step.create_event(description=description, result=result)

    def end_event(self, is_successful: bool, cost_added: Optional[float] = None, 
                 model: Optional[str] = None, result: Optional[str] = None) -> None:
        if not self._active_step:
            raise ValueError("No active step to end event in")
        if not self._active_step.event_history:
            raise ValueError("No events exist in the current step")
        latest_event = self._active_step.event_history[-1]
        if latest_event.is_finished:
            raise ValueError("Latest event is already finished")
            
        latest_event.finish_event(
            is_successful=is_successful, 
            cost_added=cost_added, 
            model=model,
            result=result 
        )
        
        self._active_step.cost_added = sum(
            event.cost_added for event in self._active_step.event_history 
            if event.cost_added is not None
        )

    def finish_session(self, is_successful: bool) -> bool:
        if self._active_step:
            raise ValueError("Cannot finish session with active step. Please finish current step first.")
            
        #self.print_step_history()
        self.is_finished = True
        self.is_successful = is_successful
        return self.update_session(is_finished=True, is_successful=is_successful)

    def get_presigned_url(self, step_id):
        request_data = {
            "agent_id": self.agent_id,
            "step_id": step_id
        }
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        
        response = requests.get(
            f"{self.base_url}/getpresigneduploadurl",
            headers=headers,
            params=request_data
        )
        
        handle_session_response(response)
        response = response.json()
        return response['presigned_url'], response['bucket_name'], response['object_key']

    def upload_image_to_s3(self, url, image):
        image_obj = io.BytesIO(base64.b64decode(image))
        image_obj.seek(0)
        upload_response = requests.put(
            url,
            data=image_obj.getvalue(),
        )
        upload_response.raise_for_status()

   #DEBUGGING PRINTS      
    def print_step_history(self):
        print("\nSession Step History:")
        print("=" * 50)
        print()
        
        for i, step in enumerate(self.step_history):
            print(f"Step {i}:")
            print(f"Goal: {step.goal}")
            print(f"Action: {step.action}")
            print(f"State: {step.state}")
            print(f"Status: {'Successful' if step.is_successful else 'Failed' if step.is_successful is False else 'In Progress'}")
            print()
            
            print("Step Event History (Goal: {}):\n{}".format(
                step.goal,
                "-" * 50
            ))
            print()
            
    def print_all(self):
        """Print complete history of session including all steps and their events in a clean format"""
        # Session Header
        print("\n" + "="*80)
        print(f"{'SESSION OVERVIEW':^80}")
        print("="*80)
        print(f"Name: {self.session_name}")
        print(f"Status: {'Successful' if self.is_successful else 'Failed' if self.is_successful is False else 'In Progress'}")
        print(f"Task: {self.task or 'None'}")
        print("-"*80)

        # Steps and Events
        for i, step in enumerate(self.step_history, 1):
            # Step Header
            print(f"\n{'STEP '+str(i):=^80}")
            print(f"Goal: {step.goal}")
            print(f"State: {step.state}")
            print(f"Action: {step.action}")
            print(f"Status: {'Successful' if step.is_successful else 'Failed' if step.is_successful is False else 'In Progress'}")
            print(f"Cost: {step.cost_added}")
            print(f"Duration: {step.start_time} → {step.end_time or 'Ongoing'}")
            
            # Events
            if step.event_history:
                print(f"\n{'EVENTS':->40}")
                for j, event in enumerate(step.event_history, 1):
                    print(f"\nEvent {j}:")
                    print(f"  Description: {event.description}")
                    print(f"  Result: {event.result}")
                    print(f"  Status: {'Successful' if event.is_successful else 'Failed' if event.is_successful is False else 'In Progress'}")
                    print(f"  Cost: {event.cost_added}")
                    print(f"  Model: {event.model or 'None'}")
                    print(f"  Start Time: {event.start_time}")
                    print(f"  End Time: {event.end_time or 'Ongoing'}")
            else:
                print("\nNo events recorded")
            
            print("-"*80)
        
        # Session Footer
        print("\n" + "="*80)
        print(f"Total Steps: {len(self.step_history)}")
        print(f"Total Cost: {sum(step.cost_added or 0 for step in self.step_history)}")
        print("="*80)