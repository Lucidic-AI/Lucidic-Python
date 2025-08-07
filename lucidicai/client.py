import os
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import requests
import logging
from requests.adapters import HTTPAdapter, Retry
from urllib3.util import Retry


from .errors import APIKeyVerificationError, InvalidOperationError, LucidicNotInitializedError
from .telemetry.base_provider import BaseProvider 
from .session import Session
from .singleton import singleton, clear_singletons
from .lru import LRUCache

NETWORK_RETRIES = 3


@singleton
class Client:
    def __init__(
        self,
        api_key: str,
        agent_id: str,
    ):
        self.base_url = "https://analytics.lucidic.ai/api" if not (os.getenv("LUCIDIC_DEBUG", 'False') == 'True') else "http://localhost:8000/api"
        self.initialized = False
        self.session = None
        self.previous_sessions = LRUCache(500)  # For LRU cache of previously initialized sessions
        self.custom_session_id_translations = LRUCache(500) # For translations of custom session IDs to real session IDs
        self.providers = []
        self.api_key = api_key
        self.agent_id = agent_id
        self.masking_function = None
        self.auto_end = False  # Default to False until explicitly set during init
        self.request_session = requests.Session()
        retry_cfg = Retry(
            total=3,                     # 3 attempts in total
            backoff_factor=0.5,          # exponential back-off: 0.5s, 1s, 2s …
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retry_cfg, pool_connections=20, pool_maxsize=100)
        self.request_session.mount("https://", adapter)
        self.set_api_key(api_key)
        self.prompts = dict()

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        self.request_session.headers.update({"Authorization": f"Api-Key {self.api_key}", "User-Agent": "lucidic-sdk/1.1"})
        try:
            self.verify_api_key(self.base_url, api_key)
        except APIKeyVerificationError:
            raise APIKeyVerificationError("Invalid API Key")

    def clear(self):
        self.undo_overrides()
        clear_singletons()
        self.initialized = False
        self.session = None
        self.providers = []
        del self

    def verify_api_key(self, base_url: str, api_key: str) -> Tuple[str, str]:
        data = self.make_request('verifyapikey', 'GET', {})  # TODO: Verify against agent ID provided
        return data["project"], data["project_id"]

    def set_provider(self, provider: BaseProvider) -> None:
        """Set the LLM provider to track"""
        self.providers.append(provider)
        provider.override()

    def undo_overrides(self):
        for provider in self.providers:
            provider.undo_override()

    def init_session(
        self,
        session_name: str,
        mass_sim_id: Optional[str] = None,
        task: Optional[str] = None,
        rubrics: Optional[list] = None,
        tags: Optional[list] = None,
        production_monitoring: Optional[bool] = False,
        session_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        if session_id:
            # Check if it's a known session ID, maybe custom and maybe real
            if session_id in self.custom_session_id_translations:
                session_id = self.custom_session_id_translations[session_id]
            # Check if it's the same as the current session
            if self.session and self.session.session_id == session_id:
                return self.session.session_id
            # Check if it's a previous session that we have saved
            if session_id in self.previous_sessions:
                if self.session:
                    self.previous_sessions[self.session.session_id] = self.session
                self.session = self.previous_sessions.pop(session_id)  # Remove from previous sessions because it's now the current session
                return self.session.session_id

        # Either there's no session ID, or we don't know about the old session
        # We need to go to the backend in both cases
        request_data = {
            "agent_id": self.agent_id,
            "session_name": session_name,
            "task": task,
            "mass_sim_id": mass_sim_id,
            "experiment_id": experiment_id,
            "rubrics": rubrics,
            "tags": tags,
            "session_id": session_id
        }
        data = self.make_request('initsession', 'POST', request_data)
        real_session_id = data["session_id"]
        if session_id and session_id != real_session_id:
            self.custom_session_id_translations[session_id] = real_session_id
        
        if self.session:
            self.previous_sessions[self.session.session_id] = self.session

        self.session = Session(
            agent_id=self.agent_id,
            session_id=real_session_id,
            session_name=session_name,
            mass_sim_id=mass_sim_id,
            experiment_id=experiment_id,
            task=task,
            rubrics=rubrics,
            tags=tags,
        )
        self.initialized = True
        return self.session.session_id

    def continue_session(self, session_id: str):
        if session_id in self.custom_session_id_translations:
            session_id = self.custom_session_id_translations[session_id]
        if self.session and self.session.session_id == session_id:
            return self.session.session_id
        if self.session:
            self.previous_sessions[self.session.session_id] = self.session
        data = self.make_request('continuesession', 'POST', {"session_id": session_id})
        real_session_id = data["session_id"]
        if session_id != real_session_id:
            self.custom_session_id_translations[session_id] = real_session_id
        self.session = Session(
            agent_id=self.agent_id,
            session_id=real_session_id
        )
        logger.info(f"Session {data.get('session_name', '')} continuing...")
        return self.session.session_id

    def init_mass_sim(self, **kwargs) -> str:
        kwargs['agent_id'] = self.agent_id
        return self.make_request('initmasssim', 'POST', kwargs)['mass_sim_id']

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
            "GET": lambda data: self.request_session.get(f"{self.base_url}/{endpoint}", params=data),
            "POST": lambda data: self.request_session.post(f"{self.base_url}/{endpoint}", json=data),
            "PUT": lambda data: self.request_session.put(f"{self.base_url}/{endpoint}", json=data),
            "DELETE": lambda data: self.request_session.delete(f"{self.base_url}/{endpoint}", params=data),
        }  # TODO: make into enum
        data['current_time'] = datetime.now().astimezone(timezone.utc).isoformat()
        func = http_methods[method]
        for _ in range(NETWORK_RETRIES):
            try:
                response = func(data)
                break
            except Exception:
                pass
        if response is None:
            raise InvalidOperationError("Cannot reach backend. Check your internet connection.")
        if response.status_code == 401:
            raise APIKeyVerificationError("Invalid API key: 401 Unauthorized")
        if response.status_code == 402:
            raise InvalidOperationError("Invalid operation: 402 Insufficient Credits")
        if response.status_code == 403:
            raise APIKeyVerificationError(f"Invalid API key: 403 Forbidden")
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise InvalidOperationError(f"Request to Lucidic AI Backend failed: {e.response.text}")
        return response.json()

    def mask(self, data):
        if not self.masking_function:
            return data
        if not data:
            return data
        try:
            return self.masking_function(data)
        except Exception as e:
            logger = logging.getLogger('Lucidic')
            logger.error(f"Error in custom masking function: {repr(e)}")
            return "<Error in custom masking function, this is a fully-masked placeholder>"