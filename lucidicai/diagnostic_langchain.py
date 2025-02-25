"""Minimal Langchain integration for Lucidic API - Focus on event closing"""
from typing import Dict, Any, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult, ChatGenerationChunk, GenerationChunk
from langchain_core.documents import Document

class LucidicLangchainHandler(BaseCallbackHandler):
    """Minimal callback handler for Langchain integration with Lucidic"""
    
    def __init__(self, client):
        """Initialize the handler with a Lucidic client."""
        self.client = client
        print("[DEBUG] Handler initialized")
    
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Handle start of chat model calls - ONLY print, no event creation"""
        print(f"[DEBUG] Chat model start for run {run_id}")
    
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Handle LLM errors - ONLY print, no event handling"""
        print(f"[DEBUG] LLM error for run {run_id}: {error}")
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Handle end of LLM calls - ONLY print, no event handling"""
        print(f"[DEBUG] LLM end for run {run_id}")
    
    def attach_to_llms(self, llm_or_chain_or_agent) -> None:
        """Attach this handler to an LLM, chain, or agent"""
        # If it's a direct LLM
        if hasattr(llm_or_chain_or_agent, 'callbacks'):
            callbacks = llm_or_chain_or_agent.callbacks or []
            if self not in callbacks:
                callbacks.append(self)
                llm_or_chain_or_agent.callbacks = callbacks
                print(f"[DEBUG] Attached to {llm_or_chain_or_agent.__class__.__name__}")
                
        # If it's a chain or agent, try to find LLMs recursively
        for attr_name in dir(llm_or_chain_or_agent):
            try:
                if attr_name.startswith('_'):
                    continue
                attr = getattr(llm_or_chain_or_agent, attr_name)
                if hasattr(attr, 'callbacks'):
                    callbacks = attr.callbacks or []
                    if self not in callbacks:
                        callbacks.append(self)
                        attr.callbacks = callbacks
                        print(f"[DEBUG] Attached to {attr.__class__.__name__} in {attr_name}")
            except Exception as e:
                print(f"[DEBUG] Warning: Could not attach to {attr_name}: {e}")