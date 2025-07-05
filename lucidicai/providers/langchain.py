"""Langchain integration for Lucidic API with detailed logging"""
import base64
import logging
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult

from lucidicai.client import Client
from lucidicai.model_pricing import calculate_cost
from langchain_core.load.dump import dumps
import json
import traceback

logger = logging.getLogger("Lucidic")


class LucidicLangchainHandler(BaseCallbackHandler):
    
    def __init__(self):
        """Initialize the handler with a Lucidic client.
        
        """
        # Keep track of which run is associated with which model
        self.run_to_model = {}
        # Keep track of which run is associated with which event
        self.run_to_event = {}
        logger.debug("Initialized LucidicLangchainHandler")

    def _get_model_name(self, serialized: Dict, kwargs: Dict) -> str:
        """Extract model name from input parameters"""
        if "invocation_params" in kwargs and "model" in kwargs["invocation_params"]:
            return kwargs["invocation_params"]["model"]
        if serialized and "model_name" in serialized:
            return serialized["model_name"]
        if serialized and "name" in serialized:
            return serialized["name"]
        return "unknown_model"

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
        logger.debug("Starting LLM call in Langchain Handler, creating event...")
        run_str = str(run_id)
        model = self._get_model_name(serialized, kwargs)
        self.run_to_model[run_str] = model
        
        text = []
        images = []
        for prompt in prompts:
            if isinstance(prompt, str):
                text.append(prompt)
            elif isinstance(prompt, dict) and 'image' in prompt:
                images.append(prompt['image'])
            
        try:
            # Create a new event
            event_id = Client().session.create_event(description=text, screenshots=images)
            self.run_to_event[run_str] = event_id
        except Exception as e:
            logger.error(f"Error creating event: {e}")
            logger.error(traceback.format_exc())

#TODO: Don't really know when this is used probably need to check documentation. 
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
        logger.debug("Starting LLM call in Langchain Handler, creating event...")
        run_str = str(run_id)
        model = self._get_model_name(serialized, kwargs)
        self.run_to_model[run_str] = model
        
        text = []
        images_b64 = []

        if messages and messages[0]:
            for msg in messages[0]:
                content = msg.content
                if isinstance(content, str):
                    text.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text.append(block.get("text", ""))
                            elif block.get("type") == "image_url":
                                image_url = block.get("image_url", "")
                                image_str = image_url.get('url', "")
                                images_b64.append(image_str[image_str.find(',') + 1:])
            
        try:
            # Create a new event
            event_id = Client().session.create_event(description=text, screenshots=images_b64)
            self.run_to_event[run_str] = event_id
        except Exception as e:
            logger.error(f"Error creating event: {e}")

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
        logger.debug("Ending LLM call in Langchain Handler, ending event...")
        run_str = str(run_id)
        model = self.run_to_model.get(run_str, "unknown")
        
        # Calculate cost if token usage exists
        cost = None
        if response.generations and response.generations[0]:
            message = response.generations[0][0].message
            usage = message.usage_metadata
            cost = calculate_cost(model, usage)
            
        try:
            if run_str in self.run_to_event:
                event_id = self.run_to_event[run_str]
                
                if not Client().session.event_history[event_id].is_finished:
                    result = None
                    if message:
                        result = message.pretty_repr()
                        
                    Client().session.update_event(
                        event_id=event_id,
                        is_finished=True, 
                        is_successful=True, 
                        cost_added=cost, 
                        model=model,
                        result=result
                    )
                else:
                    logger.debug("Event already finished")
                
                del self.run_to_event[run_str]
            else:
                logger.warning("No event found")
        except Exception as e:
            logger.error(f"Error in event ending: {e}")
            logger.error(traceback.format_exc())
            
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
        logger.debug("Handling LLM error in Langchain Handler, ending event...")
        run_str = str(run_id)
        model = self.run_to_model.get(run_str, "unknown")
            
        try:
            if run_str in self.run_to_event:
                event_id = self.run_to_event[run_str]
                if not Client().session.event_history[event_id].is_finished:
                    Client().session.update_event(
                        event_id=event_id,
                        is_finished=True, 
                        is_successful=False, 
                        model=model,
                        result=str(error)
                    )
                    logger.debug("Ended event with error")
                del self.run_to_event[run_str]
            else:
                logger.warning("No event found")
        except Exception as e:
            logger.error(f"Error ending event: {e}")
            
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
        logger.debug("Starting chain execution in Langchain Handler, creating event...")
        run_str = str(run_id)
        
        text = []
        images_b64 = []

        if inputs and inputs[0]:
            for msg in inputs[0]:
                content = msg.content
                if isinstance(content, str):
                    text.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text.append(block.get("text", ""))
                            elif block.get("type") == "image_url":
                                image_url = block.get("image_url", "")
                                image_str = image_url.get('url', "")
                                images_b64.append(image_str[image_str.find(',') + 1:])
            
        try:
            # Create a new event
            event_id = Client().session.active_step.create_event(description=text, screenshots=images_b64)
            self.run_to_event[run_str] = event_id
        except Exception as e:
            logger.error(f"Error creating chain event: {e}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Handle end of chain execution"""
        logger.debug("Ending chain execution in Langchain Handler, ending event...")
        run_id = str(kwargs.get("run_id", "unknown"))
        
        # Extract result from outputs
        result = None
        if outputs:
            # Try to get the first output value
            first_key = next(iter(outputs))
            output_value = outputs[first_key]
            
            # Convert to string if needed and truncate
            if output_value is not None:
                result = str(dumps(output_value, pretty=True)) if output_value else None
            
        try:
            if run_id in self.run_to_event:
                event_id = self.run_to_event[run_id]
                if not Client().session.event_history[event_id].is_finished:
                    Client().session.update_event(
                        event_id=event_id,
                        is_finished=True, 
                        is_successful=True, 
                        result=result
                    )
                del self.run_to_event[run_id]
            else:
                logger.warning("No event found")
        except Exception as e:
            logger.error(f"Error ending chain event: {e}")

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Handle chain errors"""
        logger.debug("Handling chain error in Langchain Handler, ending event...")
        run_id = str(kwargs.get("run_id", "unknown"))
            
        try:
            if run_id in self.run_to_event:
                event_id = self.run_to_event[run_id]
                if not Client().session.event_history[event_id].is_finished:
                    Client().session.update_event(
                        event_id=event_id,
                        is_finished=True, 
                        is_successful=False, 
                        result=str(error)
                    )
                del self.run_to_event[run_id]
            else:
                logger.warning("No event found")
        except Exception as e:
            logger.error(f"Error ending chain event: {e}")

    # Simple implementations for remaining methods:
    def on_tool_start(self, serialized, input_str, **kwargs):
        """
        Handle start of tool execution
        """
        logger.debug("Starting tool execution in Langchain Handler, creating event...")
        run_id = str(kwargs.get("run_id", "unknown"))
        tool_name = serialized.get("name", "Unknown Tool")
        description = f"Tool Call ({tool_name}): {input_str[:100]}..."
            
        try:
            # Create event
            event_id = Client().session.active_step.create_event(description=description)
            self.run_to_event[run_id] = event_id
        except Exception as e:
            logger.error(f"Error creating tool event: {e}")

    def on_tool_end(self, output, **kwargs):
        """
        Handle end of tool execution
        """
        logger.debug("Ending tool execution in Langchain Handler, ending event...")
        run_id = str(kwargs.get("run_id", "unknown"))
        
        # Get result from output
        result = None
        if output is not None:
            result = str(output)[:1000]
            
        try:
            if run_id in self.run_to_event:
                event_id = self.run_to_event[run_id]
                if not Client().session.event_history[event_id].is_finished:
                    Client().session.update_event(
                        event_id=event_id,
                        is_finished=True, 
                        is_successful=True, 
                        result=result
                    )
                del self.run_to_event[run_id]
            else:
                logger.warning("No event found")
        except Exception as e:
            logger.error(f"Error ending tool event: {e}")

    def on_tool_error(self, error, **kwargs):
        """
        Handle tool errors
        """
        logger.debug("Handling tool error in Langchain Handler, ending event...")
        run_id = str(kwargs.get("run_id", "unknown"))
            
        try:
            if run_id in self.run_to_event:
                event_id = self.run_to_event[run_id]
                if not Client().session.event_history[event_id].is_finished:
                    Client().session.update_event(
                        event_id=event_id,
                        is_finished=True, 
                        is_successful=False, 
                        result=str(error)
                    )
                del self.run_to_event[run_id]
            else:
                logger.warning("No event found")
        except Exception as e:
            logger.error(f"Error ending tool event: {e}")

    def on_retriever_start(self, serialized, query, **kwargs):
        """
        Handle start of retriever execution
        """
        logger.debug("Starting retriever execution in Langchain Handler, creating event...")
        run_id = str(kwargs.get("run_id", "unknown"))
        retriever_type = serialized.get("name", "Unknown Retriever")
        description = f"Retriever ({retriever_type}): {query[:100]}..."
            
        try:
            # Create event
            event_id = Client().session.active_step.create_event(description=description)
            self.run_to_event[run_id] = event_id
        except Exception as e:
            logger.error(f"Error creating retriever event: {e}")

    def on_retriever_end(self, documents, **kwargs):
        """
        Handle end of retriever execution
        """ 
        logger.debug("Ending retriever execution in Langchain Handler, ending event...")
        run_id = str(kwargs.get("run_id", "unknown"))
        
        # Extract result from documents
        result = None
        if documents:
            # Try to get a meaningful summary of retrieved documents
            try:
                doc_count = len(documents)
                sample = str(documents[0].page_content)[:200] if hasattr(documents[0], 'page_content') else str(documents[0])[:200]
                result = f"Retrieved {doc_count} documents. Sample: {sample}..."
            except (IndexError, AttributeError):
                # Fallback to simple string representation
                result = f"Retrieved {len(documents)} documents"
            
        try:
            if run_id in self.run_to_event:
                event_id = self.run_to_event[run_id]
                if not Client().session.event_history[event_id].is_finished:
                    Client().session.update_event(
                        event_id=event_id,
                        is_finished=True, 
                        is_successful=True, 
                        result=result
                    )
                del self.run_to_event[run_id]
            else:
                logger.warning("No event found")
        except Exception as e:
            logger.error(f"Error ending retriever event: {e}")

    def on_retriever_error(self, error, **kwargs):
        """
        Handle retriever errors
        """
        logger.debug("Handling retriever error in Langchain Handler, ending event...")
        run_id = str(kwargs.get("run_id", "unknown"))
            
        try:
            if run_id in self.run_to_event:
                event_id = self.run_to_event[run_id]
                if not Client().session.event_history[event_id].is_finished:
                    Client().session.update_event(
                        event_id=event_id,
                        is_finished=True, 
                        is_successful=False, 
                        result=str(error)
                    )
                del self.run_to_event[run_id]
            else:
                logger.warning("No event found")
        except Exception as e:
            logger.error(f"Error ending retriever event: {e}")

    def on_agent_action(self, action, **kwargs):
        """
        Handle agent actions
        """
        logger.debug("Starting agent action in Langchain Handler, creating event...")
        run_id = str(kwargs.get("run_id", "unknown"))
        tool = getattr(action, 'tool', 'unknown_tool')
        description = f"Agent Action: {tool}"
        
        # Extract useful information from the action
        result = None
        try:
            tool_input = getattr(action, 'tool_input', None)
            if tool_input:
                if isinstance(tool_input, dict):
                    # Format dictionary nicely
                    input_str = ", ".join(f"{k}: {v}" for k, v in tool_input.items())
                    result = f"Using tool '{tool}' with inputs: {input_str}"
                else:
                    result = f"Using tool '{tool}' with input: {str(tool_input)}"
            else:
                result = f"Using tool '{tool}'"
        except Exception:
            result = f"Using tool '{tool}'"
            
        try:
            # Create event
            event_id = Client().session.active_step.create_event(description=description)
            self.run_to_event[run_id] = event_id
            
            # Note: Agent actions are immediately ended in the original code
            # This seems intentional so we'll keep the behavior but use our event
            if not Client().session.event_history[event_id].is_finished:
                Client().session.update_event(
                    event_id=event_id,
                    is_finished=True, 
                    is_successful=True, 
                    result=result
                )
            del self.run_to_event[run_id]
            
            logger.debug("Processed agent action")
        except Exception as e:
            logger.error(f"Error processing agent action: {e}")

    def on_agent_finish(self, finish, **kwargs):
        """
        Handle agent finish events
        """
        logger.debug("Handling agent finish in Langchain Handler, ending event...")
        run_id = str(kwargs.get("run_id", "unknown"))
        
        # Extract result from finish
        result = None
        try:
            if hasattr(finish, 'return_values'):
                if isinstance(finish.return_values, dict) and 'output' in finish.return_values:
                    result = str(finish.return_values['output'])[:1000]
                else:
                    result = str(finish.return_values)[:1000]
            elif hasattr(finish, 'output'):
                result = str(finish.output)[:1000]
        except Exception:
            pass
            
        try:
            # Create event
            if not Client().session._active_event.is_finished:
                Client().session.update_event(
                    is_finished=True, 
                    is_successful=True, 
                    result=result
                )
        
            
            logger.debug("Processed agent finish")
        except Exception as e:
            logger.error(f"Error processing agent finish: {e}")
    
    def attach_to_llms(self, llm_or_chain_or_agent) -> None:
        """Attach this handler to an LLM, chain, or agent"""
        # If it's a direct LLM
        logger.debug(f"Attempting to attach to {llm_or_chain_or_agent.__class__.__name__}")
        if hasattr(llm_or_chain_or_agent, 'callbacks'):
            callbacks = llm_or_chain_or_agent.callbacks or []
            if not any(isinstance(callback, LucidicLangchainHandler) for callback in callbacks):
                callbacks.append(self)
                llm_or_chain_or_agent.callbacks = callbacks
                logger.debug(f"Successfully attached to {llm_or_chain_or_agent.__class__.__name__}")
            else:
                logger.debug(f"Already attached to {llm_or_chain_or_agent.__class__.__name__}")
        # If it's a chain or agent, try to find LLMs recursively
        for attr_name in dir(llm_or_chain_or_agent):
            try:
                if attr_name.startswith('_'):
                    continue
                attr = getattr(llm_or_chain_or_agent, attr_name)
                if hasattr(attr, 'callbacks'):
                    callbacks = attr.callbacks or []
                    if not any(isinstance(callback, LucidicLangchainHandler) for callback in callbacks):
                        callbacks.append(self)
                        attr.callbacks = callbacks
                        logger.debug(f"Successfully attached to {attr.__class__.__name__} in {attr_name}")
                    else:
                        logger.debug(f"Already attached to {attr.__class__.__name__} in {attr_name}")
            except Exception as e:
                logger.warning(f"Could not attach to {attr_name}: {e}")