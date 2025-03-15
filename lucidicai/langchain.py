"""Langchain integration for Lucidic API with detailed logging"""
from typing import Dict, Any, List, Optional, Union
from uuid import UUID
import traceback

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult, ChatGenerationChunk, GenerationChunk
from langchain_core.documents import Document

from .client import Client

class LucidicLangchainHandler(BaseCallbackHandler):
    """Callback handler for Langchain integration with Lucidic"""
    
    def __init__(self):
        """Initialize the handler with a Lucidic client.
        
        """
        # TODO: Remove need for client argument, grab singleton
        self.client = Client()
        # Keep track of which run is associated with which model
        self.run_to_model = {}
        print("[Handler] Initialized LucidicLangchainHandler")

    def _get_model_name(self, serialized: Dict, kwargs: Dict) -> str:
        """Extract model name from input parameters"""
        if "invocation_params" in kwargs and "model" in kwargs["invocation_params"]:
            return kwargs["invocation_params"]["model"]
        if serialized and "model_name" in serialized:
            return serialized["model_name"]
        if serialized and "name" in serialized:
            return serialized["name"]
        return "unknown_model"

    def _calculate_cost(self, model: str, token_usage: Dict) -> float:
        """Calculate cost based on model and token usage"""
        if model == "gpt-4o":
            return ((token_usage.get("completion_tokens", 0) * 10) + 
                    (token_usage.get("prompt_tokens", 0) * 2.5)) / 1_000_000
        if model == "gpt-4o-mini":
            return ((token_usage.get("completion_tokens", 0) * 0.6) + 
                    (token_usage.get("prompt_tokens", 0) * 0.15)) / 1_000_000
        if model == "o1":
            return ((token_usage.get("completion_tokens", 0) * 60) + 
                    (token_usage.get("prompt_tokens", 0) * 15)) / 1_000_000
        if model == "o3-mini":
            return ((token_usage.get("completion_tokens", 0) * 4.4) + 
                    (token_usage.get("prompt_tokens", 0) * 1.1)) / 1_000_000
        if model == "o1-mini":
            return ((token_usage.get("completion_tokens", 0) * 4.4) + 
                    (token_usage.get("prompt_tokens", 0) * 1.1)) / 1_000_000

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle start of LLM calls"""
        run_str = str(run_id)
        model = self._get_model_name(serialized, kwargs)
        self.run_to_model[run_str] = model
        
        description = f"LLM call ({model}): {prompts[0]}..." if prompts else "LLM call"
        
        # Make sure we have a valid session and step
        print(f"\n[DEBUG] on_llm_start for run_id {run_str}")
        print(f"[DEBUG] client session: {self.client.session}")
        if not (self.client.session and self.client.session.active_step):
            print(f"[DEBUG] Cannot create event - no active session or step")
            return
        
        print(f"[DEBUG] active_step: {self.client.session.active_step}")
        print(f"[DEBUG] event_history count: {len(self.client.session.active_step.event_history)}")
        
        # Print all events and their status
        for i, event in enumerate(self.client.session.active_step.event_history):
            print(f"[DEBUG] Event {i} id={event.event_id} finished={event.is_finished}")
            
        try:
            # Check if we have any unfinished events
            for i, event in enumerate(self.client.session.active_step.event_history):
                if not event.is_finished:
                    print(f"[DEBUG] Attempting to finish unfinished event {i} id={event.event_id}")
                    try:
                        event.end_event(is_successful=False, result="Force-closed by handler")
                        print(f"[DEBUG] Successfully finished event {i}")
                    except Exception as e:
                        print(f"[DEBUG] Error finishing event {i}: {e}")
                        print(traceback.format_exc())
            
            # Create a new event
            print(f"[DEBUG] Creating new event with description: {description}")
            event = self.client.session.create_event(description=description)
            print(f"[DEBUG] Created new event with id={event.event_id}")
            
            # Check all events again after creation
            print(f"[DEBUG] After creation, event_history count: {len(self.client.session.active_step.event_history)}")
            for i, event in enumerate(self.client.session.active_step.event_history):
                print(f"[DEBUG] Event {i} id={event.event_id} finished={event.is_finished}")
                
        except Exception as e:
            print(f"[DEBUG] Error in event creation: {e}")
            print(traceback.format_exc())

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle start of chat model calls"""
        run_str = str(run_id)
        model = self._get_model_name(serialized, kwargs)
        self.run_to_model[run_str] = model
        
        # Format messages for description
        parsed_messages = []
        if messages and messages[0]:
            for msg in messages[0]:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    parsed_messages.append(f"{msg.type}: {msg.content}...")
        
        message_desc = "; ".join(parsed_messages[:2])
        description = f"Chat model call ({model}): {message_desc}"
        
        # Make sure we have a valid session and step
        if not (self.client.session and self.client.session.active_step):
            print(f"[Handler] Cannot create event - no active session or step")
            return
            
        try:
            # Check if we have any unfinished events
            for event in self.client.session.active_step.event_history:
                if not event.is_finished:
                    print(f"[Handler] Found unfinished event, calling finish_event")
                    event.end_event(is_successful=False)
            
            # Create a new event
            self.client.session.create_event(description=description)
            print(f"[Handler] Started event for run {run_str}")
        except Exception as e:
            print(f"[Handler] Error creating event: {e}")

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        **kwargs: Any
    ) -> None:
        """Handle streaming tokens"""
        # We don't need to track tokens for this implementation
        pass

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle end of LLM call"""
        run_str = str(run_id)
        model = self.run_to_model.get(run_str, "unknown")
        
        # Calculate cost if token usage exists
        cost = None
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            model = response.llm_output.get("model_name", model)
            cost = self._calculate_cost(model, usage)
        
        # Make sure we have a valid session
        print(f"\n[DEBUG] on_llm_end for run_id {run_str}")
        if not (self.client.session and self.client.session.active_step):
            print(f"[DEBUG] Cannot end event - no active session or step")
            return
            
        print(f"[DEBUG] active_step: {self.client.session.active_step}")
        if self.client.session.active_step:
            print(f"[DEBUG] event_history count: {len(self.client.session.active_step.event_history)}")
            
            # Print all events and their status
            for i, event in enumerate(self.client.session.active_step.event_history):
                print(f"[DEBUG] Event {i} id={event.event_id} finished={event.is_finished}")
        
        try:
            # Get the active step
            step = self.client.session.active_step
            if step and step.event_history:
                # Try to get the most recent unfinished event
                unfinished_events = [e for e in step.event_history if not e.is_finished]
                if unfinished_events:
                    last_unfinished = unfinished_events[-1]
                    print(f"[DEBUG] Found unfinished event id={last_unfinished.event_id}")
                    try:
                        result = None
                        if response.generations and response.generations[0]:
                            result = response.generations[0][0].text[:1000]
                            
                        last_unfinished.end_event(
                            is_successful=True, 
                            cost_added=cost, 
                            model=model,
                            result=result
                        )
                        print(f"[DEBUG] Successfully finished event id={last_unfinished.event_id}")
                    except Exception as e:
                        print(f"[DEBUG] Error finishing event: {e}")
                        print(traceback.format_exc())
                else:
                    print(f"[DEBUG] No unfinished events found to end")
            else:
                print(f"[DEBUG] No events found to end")
        except Exception as e:
            print(f"[DEBUG] Error in event ending: {e}")
            print(traceback.format_exc())
            
        # Clean up
        if run_str in self.run_to_model:
            del self.run_to_model[run_str]

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM errors"""
        run_str = str(run_id)
        model = self.run_to_model.get(run_str, "unknown")
        
        # Make sure we have a valid session
        if not (self.client.session and self.client.session.active_step):
            print(f"[Handler] Cannot end event - no active session or step")
            return
            
        try:
            # Make sure we're ending the event in the active step's last event
            step = self.client.session.active_step
            if step and step.event_history:
                last_event = step.event_history[-1]
                if not last_event.is_finished:
                    last_event.end_event(is_successful=False, model=model)
                    print(f"[Handler] Ended event with error for run {run_str}")
                else:
                    print(f"[Handler] Event already finished for run {run_str}")
            else:
                print(f"[Handler] No events found to end for run {run_str}")
        except Exception as e:
            print(f"[Handler] Error ending event: {e}")
            
        # Clean up
        if run_str in self.run_to_model:
            del self.run_to_model[run_str]

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Handle start of chain execution"""
        run_str = str(run_id)
        chain_type = serialized.get("name", "Unknown Chain")
        input_desc = str(next(iter(inputs.values())))[:50] if inputs else "No inputs"
        
        description = f"Chain execution ({chain_type}): {input_desc}..."
        
        # Make sure we have a valid session and step
        if not (self.client.session and self.client.session.active_step):
            print(f"[Handler] Cannot create event - no active session or step")
            return
            
        try:
            # Check if we have any unfinished events
            for event in self.client.session.active_step.event_history:
                if not event.is_finished:
                    print(f"[Handler] Found unfinished event, calling finish_event")
                    event.end_event(is_successful=False)
            
            # Create a new event
            self.client.session.create_event(description=description)
            print(f"[Handler] Started chain event for run {run_str}")
        except Exception as e:
            print(f"[Handler] Error creating chain event: {e}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Handle end of chain execution"""
        run_id = str(kwargs.get("run_id", "unknown"))
        
        # Make sure we have a valid session
        if not (self.client.session and self.client.session.active_step):
            print(f"[Handler] Cannot end event - no active session or step")
            return
            
        try:
            # Make sure we're ending the event in the active step's last event
            step = self.client.session.active_step
            if step and step.event_history:
                last_event = step.event_history[-1]
                if not last_event.is_finished:
                    last_event.end_event(is_successful=True)
                    print(f"[Handler] Ended chain event for run {run_id}")
                else:
                    print(f"[Handler] Chain event already finished for run {run_id}")
            else:
                print(f"[Handler] No chain events found to end for run {run_id}")
        except Exception as e:
            print(f"[Handler] Error ending chain event: {e}")

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Handle chain errors"""
        run_id = str(kwargs.get("run_id", "unknown"))
        
        # Make sure we have a valid session
        if not (self.client.session and self.client.session.active_step):
            print(f"[Handler] Cannot end event - no active session or step")
            return
            
        try:
            # Make sure we're ending the event in the active step's last event
            step = self.client.session.active_step
            if step and step.event_history:
                last_event = step.event_history[-1]
                if not last_event.is_finished:
                    last_event.end_event(is_successful=False)
                    print(f"[Handler] Ended chain event with error for run {run_id}")
                else:
                    print(f"[Handler] Chain event already finished for run {run_id}")
            else:
                print(f"[Handler] No chain events found to end for run {run_id}")
        except Exception as e:
            print(f"[Handler] Error ending chain event: {e}")

    # Simple implementations for remaining methods:
    def on_tool_start(self, serialized, input_str, **kwargs):
        run_id = str(kwargs.get("run_id", "unknown"))
        tool_name = serialized.get("name", "Unknown Tool")
        description = f"Tool Call ({tool_name}): {input_str[:100]}..."
        
        # Make sure we have a valid session and step
        if not (self.client.session and self.client.session.active_step):
            return
            
        try:
            # Check for unfinished events
            for event in self.client.session.active_step.event_history:
                if not event.is_finished:
                    event.end_event(is_successful=False)
            
            # Create event
            self.client.session.create_event(description=description)
            print(f"[Handler] Started tool event for run {run_id}")
        except Exception as e:
            print(f"[Handler] Error creating tool event: {e}")

    def on_tool_end(self, output, **kwargs):
        run_id = str(kwargs.get("run_id", "unknown"))
        
        # Make sure we have a valid session and step
        if not (self.client.session and self.client.session.active_step):
            return
            
        try:
            # Make sure we're ending the event in the active step's last event
            step = self.client.session.active_step
            if step and step.event_history:
                last_event = step.event_history[-1]
                if not last_event.is_finished:
                    last_event.end_event(is_successful=True)
                    print(f"[Handler] Ended tool event for run {run_id}")
                else:
                    print(f"[Handler] Tool event already finished for run {run_id}")
            else:
                print(f"[Handler] No tool events found to end for run {run_id}")
        except Exception as e:
            print(f"[Handler] Error ending tool event: {e}")

    def on_tool_error(self, error, **kwargs):
        run_id = str(kwargs.get("run_id", "unknown"))
        
        # Make sure we have a valid session and step
        if not (self.client.session and self.client.session.active_step):
            return
            
        try:
            # Make sure we're ending the event in the active step's last event
            step = self.client.session.active_step
            if step and step.event_history:
                last_event = step.event_history[-1]
                if not last_event.is_finished:
                    last_event.end_event(is_successful=False)
                    print(f"[Handler] Ended tool event with error for run {run_id}")
                else:
                    print(f"[Handler] Tool event already finished for run {run_id}")
            else:
                print(f"[Handler] No tool events found to end for run {run_id}")
        except Exception as e:
            print(f"[Handler] Error ending tool event: {e}")

    def on_retriever_start(self, serialized, query, **kwargs):
        run_id = str(kwargs.get("run_id", "unknown"))
        retriever_type = serialized.get("name", "Unknown Retriever")
        description = f"Retriever ({retriever_type}): {query[:100]}..."
        
        # Make sure we have a valid session and step
        if not (self.client.session and self.client.session.active_step):
            return
            
        try:
            # Check for unfinished events
            for event in self.client.session.active_step.event_history:
                if not event.is_finished:
                    event.end_event(is_successful=False)
            
            # Create event
            self.client.session.create_event(description=description)
            print(f"[Handler] Started retriever event for run {run_id}")
        except Exception as e:
            print(f"[Handler] Error creating retriever event: {e}")

    def on_retriever_end(self, documents, **kwargs):
        run_id = str(kwargs.get("run_id", "unknown"))
        
        # Make sure we have a valid session and step
        if not (self.client.session and self.client.session.active_step):
            return
            
        try:
            # Make sure we're ending the event in the active step's last event
            step = self.client.session.active_step
            if step and step.event_history:
                last_event = step.event_history[-1]
                if not last_event.is_finished:
                    last_event.end_event(is_successful=True)
                    print(f"[Handler] Ended retriever event for run {run_id}")
                else:
                    print(f"[Handler] Retriever event already finished for run {run_id}")
            else:
                print(f"[Handler] No retriever events found to end for run {run_id}")
        except Exception as e:
            print(f"[Handler] Error ending retriever event: {e}")

    def on_retriever_error(self, error, **kwargs):
        run_id = str(kwargs.get("run_id", "unknown"))
        
        # Make sure we have a valid session and step
        if not (self.client.session and self.client.session.active_step):
            return
            
        try:
            # Make sure we're ending the event in the active step's last event
            step = self.client.session.active_step
            if step and step.event_history:
                last_event = step.event_history[-1]
                if not last_event.is_finished:
                    last_event.end_event(is_successful=False)
                    print(f"[Handler] Ended retriever event with error for run {run_id}")
                else:
                    print(f"[Handler] Retriever event already finished for run {run_id}")
            else:
                print(f"[Handler] No retriever events found to end for run {run_id}")
        except Exception as e:
            print(f"[Handler] Error ending retriever event: {e}")

    def on_agent_action(self, action, **kwargs):
        run_id = str(kwargs.get("run_id", "unknown"))
        tool = getattr(action, 'tool', 'unknown_tool')
        description = f"Agent Action: {tool}"
        
        # Make sure we have a valid session and step
        if not (self.client.session and self.client.session.active_step):
            return
            
        try:
            # Check for unfinished events
            for event in self.client.session.active_step.event_history:
                if not event.is_finished:
                    event.end_event(is_successful=False)
            
            # Create and immediately end event
            self.client.session.create_event(description=description)
            self.client.session.end_event(is_successful=True)
            print(f"[Handler] Processed agent action for run {run_id}")
        except Exception as e:
            print(f"[Handler] Error processing agent action: {e}")

    def on_agent_finish(self, finish, **kwargs):
        run_id = str(kwargs.get("run_id", "unknown"))
        description = "Agent Finish"
        
        # Make sure we have a valid session and step
        if not (self.client.session and self.client.session.active_step):
            return
            
        try:
            # Check for unfinished events
            for event in self.client.session.active_step.event_history:
                if not event.is_finished:
                    event.end_event(is_successful=False)
            
            # Create and immediately end event
            self.client.session.create_event(description=description)
            self.client.session.end_event(is_successful=True)
            print(f"[Handler] Processed agent finish for run {run_id}")
        except Exception as e:
            print(f"[Handler] Error processing agent finish: {e}")
    
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