import logging
from typing import Optional, Tuple, List

from .base_providers import BaseProvider
from lucidicai.client import Client
from lucidicai.singleton import singleton
from lucidicai.model_pricing import calculate_cost

from anthropic import Anthropic, AsyncAnthropic, Stream, AsyncStream

logger = logging.getLogger("Lucidic")


@singleton
class AnthropicHandler(BaseProvider):
    def __init__(self):
        super().__init__()
        self._provider_name = "Anthropic"
        self.original_create = None
        self.original_create_async = None
        logger.debug("AnthropicHandler initialized")

    def _format_messages(self, messages) -> Tuple[str, List[str]]:
        """
        Extract plain-text description and list of image URLs from Anthropic-formatted messages.
        """
        logger.debug("formatting messages")
        descriptions: List[str] = []
        screenshots: List[str] = []
        if not messages:
            return "", []

        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for piece in content:
                    if isinstance(piece, dict):
                        if piece.get("type") == "text":
                            descriptions.append(piece.get("text", ""))
                        elif piece.get("type") == "image":
                            # Fix: Anthropic uses "source" instead of "image"
                            source = piece.get("source", {})
                            img = source.get("data")
                            if img:
                                screenshots.append(img)
                    else:  # Assume ContentBlock
                        descriptions.append(str(piece))
            elif isinstance(content, str):
                descriptions.append(content)

        return " ".join(descriptions), screenshots

    def handle_response(self, response, kwargs):

        # for synchronous streaming responses
        if isinstance(response, Stream):
            return self._handle_stream_response(response, kwargs)
        
        # for async streaming responses -- added new
        if isinstance(response, AsyncStream):
            return self._handle_async_stream_response(response, kwargs)
        
        # for non streaming responses
        return self._handle_regular_response(response, kwargs)
    
    def _handle_stream_response(self, response: Stream, kwargs):

        accumulated_reponse = ""
        input_tokens = 0
        output_tokens = 0

        def generate():

            nonlocal accumulated_reponse, input_tokens, output_tokens

            try:
                for chunk in response:
                    if chunk.type == "content_block_start" and chunk.content_block.type == "text":
                        accumulated_reponse += chunk.content_block.text
                    elif chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                        accumulated_reponse += chunk.delta.text
                    elif chunk.type == "message_start" and hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                        input_tokens = chunk.message.usage.input_tokens
                    elif chunk.type == "message_delta" and hasattr(chunk, "delta") and hasattr(chunk.delta, "usage"):
                        output_tokens = chunk.delta.usage.output_tokens
                    elif chunk.type == "message_stop":
                        # Calculate cost on final chunk
                        cost = None
                        if input_tokens > 0 or output_tokens > 0:
                            model = kwargs.get("model")
                            cost = calculate_cost(model, {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens
                            })

                        Client().session.update_event(
                            is_finished=True,
                            is_successful=True,
                            cost_added=cost,
                            model=kwargs.get("model"),
                            result=accumulated_reponse
                        )
                    
                    yield chunk

            except Exception as e:
                Client().session.update_event(
                    is_finished=True,
                    result=f"anthropic Error: {str(e)}",
                    is_successful=False,
                    cost_added=None,
                    model=kwargs.get("model"),
                )
                
                raise

        return generate()
    
    def _handle_async_stream_response(self, response: AsyncStream, kwargs):

        accumulated_reponse = ""
        input_tokens = 0
        output_tokens = 0

        async def agenerate():

            nonlocal accumulated_reponse, input_tokens, output_tokens

            try:
                async for chunk in response:
                    if chunk.type == "content_block_start" and chunk.content_block.type == "text":
                        accumulated_reponse += chunk.content_block.text
                    elif chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                        accumulated_reponse += chunk.delta.text
                    elif chunk.type == "message_start" and hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                        input_tokens = chunk.message.usage.input_tokens
                    elif chunk.type == "message_delta" and hasattr(chunk, "delta") and hasattr(chunk.delta, "usage"):
                        output_tokens = chunk.delta.usage.output_tokens
                    elif chunk.type == "message_stop":
                        # Calculate cost on final chunk
                        cost = None
                        if input_tokens > 0 or output_tokens > 0:
                            model = kwargs.get("model")
                            cost = calculate_cost(model, {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens
                            })

                        Client().session.update_event(
                            is_finished=True,
                            is_successful=True,
                            cost_added=cost,
                            model=kwargs.get("model"),
                            result=accumulated_reponse
                        )
                    
                    yield chunk

            except Exception as e:

                Client().session.update_event(
                    is_finished=True,
                    result=f"anthropic Error: {str(e)}",
                    is_successful=False,
                    cost_added=None,
                    model=kwargs.get("model"),
                )

                raise
            
        return agenerate()
    

    def _handle_regular_response(self, response, kwargs):

        try:
            # extract text
            if hasattr(response, "content") and response.content:
                if hasattr(response.content[0], "text"):
                    response_text = response.content[0].text
                else:
                    response_text = str(response.content[0])
            else:
                response_text = str(response)

            cost = None

            if hasattr(response, "usage"):
                model = getattr(response, "model", kwargs.get("model"))
                cost = calculate_cost(model, {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                })

            Client().session.update_event(
                result=response_text,
                is_finished=True,
                is_successful=True,
                cost_added=cost,
                model=getattr(response, "model", kwargs.get("model")),
            )

        except Exception as e:
            Client().session.update_event(
                is_finished=True,
                is_successful=False,
                cost_added=None,
                model=kwargs.get("model"),
                result=f"Error processing response: {e}"
            )

            raise

        return response

    def override(self):
        from anthropic.resources.messages import messages
        from anthropic.resources.messages.messages import AsyncMessages
        
        self.original_create = messages.Messages.create
        self.original_create_async = AsyncMessages.create
        
        def patched_create(*args, **kwargs):            
            description, images = self._format_messages(kwargs.get("messages", []))
            
            event_id = Client().session.create_event(
                description=description,
                result="Waiting for response...",
                screenshots=images
            )
            
            result = self.original_create(*args, **kwargs)
            return self.handle_response(result, kwargs)
        
        async def patched_create_async(*args, **kwargs):
            description, images = self._format_messages(kwargs.get("messages", []))
            
            event_id = Client().session.create_event(
                description=description,
                result="Waiting for response...",
                screenshots=images
            )
            
            result = await self.original_create_async(*args, **kwargs)
            return self.handle_response(result, kwargs)
        
        messages.Messages.create = patched_create
        AsyncMessages.create = patched_create_async


    def undo_override(self):
        if self.original_create:
            from anthropic.resources.messages import messages
            messages.Messages.create = self.original_create
            self.original_create = None
            
        if self.original_create_async:
            from anthropic.resources.messages.messages import AsyncMessages
            AsyncMessages.create = self.original_create_async
            self.original_create_async = None
