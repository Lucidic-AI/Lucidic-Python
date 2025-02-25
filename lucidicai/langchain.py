"""Fixed Langchain integration for Lucidic API"""
from typing import Dict, Any, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

class LucidicLangchainHandler(BaseCallbackHandler):
    """Callback handler for Langchain integration with Lucidic"""
    
    def __init__(self, client):
        """Initialize the handler with a Lucidic client."""
        self.client = client
        self.event_map = {}  # Maps run_ids to event objects
        print("[Handler] Initialized LucidicLangchainHandler")
    
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Handle start of chat model calls"""
        run_str = str(run_id)
        
        # Get model name
        model = "unknown"
        if "invocation_params" in kwargs and "model" in kwargs["invocation_params"]:
            model = kwargs["invocation_params"]["model"]
        
        # Format messages for description
        message_desc = ""
        if messages and messages[0]:
            for msg in messages[0]:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    message_desc += f"{msg.type}: {msg.content[:50]}...; "
        
        description = f"Chat model call ({model}): {message_desc}"
        
        try:
            # Create a new event and store it directly
            event = self.client.session.create_event(description=description)
            self.event_map[run_str] = {"event": event, "model": model}
            print(f"[Handler] Started event for run {run_str}")
        except Exception as e:
            print(f"[Handler] Error creating event: {e}")
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Handle end of LLM call"""
        run_str = str(run_id)
        
        if run_str not in self.event_map:
            print(f"[Handler] No event found for run {run_str}")
            return
        
        try:
            event_info = self.event_map[run_str]
            event = event_info["event"]
            model = event_info["model"]
            
            # Calculate cost if token usage exists
            cost = None
            if response.llm_output and "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
                cost = self._calculate_cost(model, usage)
            
            # Extract text from response
            result = None
            if response.generations and response.generations[0]:
                result = response.generations[0][0].text[:500]
            
            # Finish the event
            if not event.is_finished:
                event.finish_event(
                    is_successful=True,
                    cost_added=cost,
                    model=model,
                    result=result
                )
                print(f"[Handler] Ended event for run {run_str}")
            
            # Clean up
            del self.event_map[run_str]
            
        except Exception as e:
            print(f"[Handler] Error ending event: {e}")
    
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Handle LLM errors"""
        run_str = str(run_id)
        
        if run_str not in self.event_map:
            print(f"[Handler] No event found for run {run_str}")
            return
        
        try:
            event_info = self.event_map[run_str]
            event = event_info["event"]
            model = event_info["model"]
            
            # Finish the event with error
            if not event.is_finished:
                event.finish_event(
                    is_successful=False,
                    model=model,
                    result=f"Error: {str(error)}"
                )
                print(f"[Handler] Ended event with error for run {run_str}")
            
            # Clean up
            del self.event_map[run_str]
            
        except Exception as e:
            print(f"[Handler] Error ending event: {e}")
    
    def _calculate_cost(self, model: str, token_usage: Dict) -> float:
        """Calculate cost based on model and token usage"""
        if "gpt-4" in model:
            return ((token_usage.get("completion_tokens", 0) * 0.03) + 
                    (token_usage.get("prompt_tokens", 0) * 0.01)) / 1000
        return ((token_usage.get("completion_tokens", 0) * 0.002) + 
                (token_usage.get("prompt_tokens", 0) * 0.001)) / 1000
    
    def attach_to_llms(self, llm_or_chain_or_agent) -> None:
        """Attach this handler to an LLM, chain, or agent"""
        # If it's a direct LLM
        if hasattr(llm_or_chain_or_agent, 'callbacks'):
            callbacks = llm_or_chain_or_agent.callbacks or []
            if self not in callbacks:
                callbacks.append(self)
                llm_or_chain_or_agent.callbacks = callbacks
                print(f"[Handler] Attached to {llm_or_chain_or_agent.__class__.__name__}")
                
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
                        print(f"[Handler] Attached to {attr.__class__.__name__} in {attr_name}")
            except Exception as e:
                print(f"[Handler] Warning: Could not attach to {attr_name}: {e}")