from typing import Optional, Tuple
from .session import Session
from .singleton import singleton
from .errors import APIKeyVerificationError

from .providers.base_providers import BaseProvider
import requests

@singleton
class Client:
    def __init__(self,
                 lucidic_api_key: str,
                 agent_id: str,
                 session_name: str,
                 mass_sim_id: Optional[str] = None,
                 task: Optional[str] = None):
        self.base_url = "https://analytics.lucidic.ai/api"
        self._initialized = False
        self._session: Optional[Session] = None
        self.api_key = None
        self.agent_id = None
        self.session_name = None
        self.mass_sim_id = None
        self.task = None
        self._provider = None
        
        self.configure(
            lucidic_api_key=lucidic_api_key,
            agent_id=agent_id,
            session_name=session_name,
            mass_sim_id=mass_sim_id,
            task=task)
    
    def configure(self,
                 lucidic_api_key: Optional[str] = None,
                 agent_id: Optional[str] = None,
                 session_name: Optional[str] = None,
                 mass_sim_id: Optional[str] = None,
                 task: Optional[str] = None) -> None:
        if lucidic_api_key:
            self.verify_api_key(self.base_url, lucidic_api_key)
            self.api_key = lucidic_api_key
            
        self.agent_id = agent_id or self.agent_id
        self.session_name = session_name or self.session_name
        self.mass_sim_id = mass_sim_id or self.mass_sim_id
        self.task = task or self.task
        self._initialized = True
    
    def set_provider(self, provider: BaseProvider):
        """Set the LLM provider to track"""
        if self._provider:
            self._provider.undo_override()
        self._provider = provider
        if self._session:
            self._provider.override()
    
    def init_session(self) -> Session:
        if not self._initialized:
            raise ValueError("Client must be configured before initializing session")
            
        if not all([self.api_key, self.agent_id, self.session_name]):
            raise ValueError("API key, agent ID, and session name are required")
            
        if self._session is None:
            self._session = Session(
                api_key=self.api_key,
                agent_id=self.agent_id,
                session_name=self.session_name,
                mass_sim_id=self.mass_sim_id,
                task=self.task,
            )
            if self._provider:
                self._provider.override()
            
        return self._session
    
    @property
    def session(self) -> Optional[Session]:
        return self._session

    def clear_session(self):
        if self._provider:
            self._provider.undo_override()
        self._session = None
        
    @property
    def is_initialized(self) -> bool:
        return self._initialized
        
    @property
    def has_session(self) -> bool:
        return self._session is not None

    def verify_api_key(self, base_url: str, api_key: str) -> Tuple[str, str]:
        try:
            response = requests.get(
                f"{base_url}/verifyapikey",
                headers={"Authorization": f"Api-Key {api_key}"}
            )
            response.raise_for_status()
            
            data = response.json()
            if not data.get("project") or not data.get("project_id"):
                raise APIKeyVerificationError("Invalid API key: Missing project information")
                
            return data["project"], data["project_id"]
            
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                raise APIKeyVerificationError("Invalid API key: Authentication failed")
            elif e.response.status_code == 403:
                raise APIKeyVerificationError("Invalid API key: Access forbidden")
            else:
                raise APIKeyVerificationError(f"API key verification failed: {str(e)}")
        except Exception as e:
            raise APIKeyVerificationError(f"API key verification failed: {str(e)}")