import os
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import requests

from .errors import APIKeyVerificationError, InvalidOperationError, LucidicNotInitializedError
from .providers.base_providers import BaseProvider
from .session import Session
from .singleton import singleton, clear_singletons


@singleton
class Client:
    def __init__(
        self,
        lucidic_api_key: str,
        agent_id: str,
        session_name: str,
        mass_sim_id: Optional[str] = None,
        task: Optional[str] = None,
        rubrics: Optional[list] = None
    ):
        self.base_url = "https://analytics.lucidic.ai/api" if not (os.getenv("LUCIDIC_DEBUG", 'False') == 'True') else "http://localhost:8000/api"
        self._initialized = False
        self._session = None
        self.api_key = None
        self.agent_id = None
        self.session_name = None
        self.mass_sim_id = None
        self.task = None
        self._provider = None
        self.prompts = dict()
        self.configure(
            lucidic_api_key=lucidic_api_key,
            agent_id=agent_id,
            session_name=session_name,
            mass_sim_id=mass_sim_id,
            task=task,
            rubrics=rubrics
        )
    
    @property
    def session(self) -> Optional[Session]:
        return self._session

    def clear_session(self) -> None:
        if self._provider:
            self._provider.undo_override()
        self._session = None
        
    @property
    def is_initialized(self) -> bool:
        return self._initialized
        
    @property
    def has_session(self) -> bool:
        return self._session is not None

    def configure(
        self,
        lucidic_api_key: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_name: Optional[str] = None,
        mass_sim_id: Optional[str] = None,
        task: Optional[str] = None,
        rubrics: Optional[list] = None
    ) -> None:
        self.api_key = lucidic_api_key
        try:
            self.verify_api_key(self.base_url, lucidic_api_key)
        except APIKeyVerificationError as e:
            raise APIKeyVerificationError(str(e))
        self.agent_id = agent_id
        self.session_name = session_name
        self.mass_sim_id = mass_sim_id
        self.task = task
        self.rubrics = rubrics
        self._initialized = True
    
    def reset(self):
        if self.session:
            self.session.end_session()
            self.clear_session()
        clear_singletons()
        del self

    def verify_api_key(self, base_url: str, api_key: str) -> Tuple[str, str]:
        data = self.make_request('verifyapikey', 'GET', {})
        return data["project"], data["project_id"]

    def set_provider(self, provider: BaseProvider) -> None:
        """Set the LLM provider to track"""
        if self._provider:
            self._provider.undo_override()
        self._provider = provider
        if self._session:
            self._provider.override()
    
    def init_session(self) -> None:
        if not self._initialized:
            raise LucidicNotInitializedError()
            
        if not all([self.agent_id, self.session_name]):
            raise InvalidOperationError("agent ID, and session name are required to initialize a session")
        
        self._session = Session(
            agent_id=self.agent_id,
            session_name=self.session_name,
            mass_sim_id=self.mass_sim_id,
            task=self.task,
            rubrics=self.rubrics
        )
        if self._provider:
            self._provider.override()

    def get_prompt(self, prompt_name, cache_ttl, label) -> str:
        current_time = time.time()
        key = (prompt_name, label)
        if key in self.prompts:
            prompt, expiration_time = self.prompts[key]
            if expiration_time == float('inf') or current_time < expiration_time:
                return prompt
        params={
            "agent_id": self.agent_id,
            "prompt_name": prompt_name,
            "label": label
        }
        prompt = self.make_request('getprompt', 'GET', params)['prompt_content']
        
        if cache_ttl != 0:
            if cache_ttl == -1:
                expiration_time = float('inf')
            else:
                expiration_time = current_time + cache_ttl
            self.prompts[key] = (prompt, expiration_time)
        return prompt

    def make_request(self, endpoint, method, data):
        http_methods = {
            "GET": lambda data: requests.get(f"{self.base_url}/{endpoint}", headers={"Authorization": f"Api-Key {self.api_key}"}, params=data),
            "POST": lambda data: requests.post(f"{self.base_url}/{endpoint}", headers={"Authorization": f"Api-Key {self.api_key}"}, json=data),
            "PUT": lambda data: requests.put(f"{self.base_url}/{endpoint}", headers={"Authorization": f"Api-Key {self.api_key}"}, json=data),
            "DELETE": lambda data: requests.delete(f"{self.base_url}/{endpoint}", headers={"Authorization": f"Api-Key {self.api_key}"}, params=data),
        }
        data['current_time'] = datetime.now().astimezone(timezone.utc).isoformat()
        func = http_methods[method]
        response = func(data)
        if response.status_code == 401:
            raise APIKeyVerificationError("Invalid API key: 401 Unauthorized")
        if response.status_code == 403:
            raise APIKeyVerificationError("Invalid API key: 403 Forbidden")
        response.raise_for_status()
        return response.json()