from typing import Optional, Tuple, List

from .base_providers import BaseProvider
from lucidicai.singleton import singleton

from anthropic import Anthropic, AsyncAnthropic, Stream, AsyncStream


@singleton
class AnthropicHandler(BaseProvider):
    def __init__(self, client):
        super().__init__(client)
        self._provider_name = "Anthropic"
        self.original_create = None
        self.original_create_async = None

    def _format_messages(self, messages) -> Tuple[str, List[str]]:
        """
        Extract plain-text description and list of image URLs from Anthropic-formatted messages.
        """
        descriptions: List[str] = []
        screenshots: List[str] = []
        if not messages:
            return "", []

        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for piece in content:
                    if piece.get("type") == "text":
                        descriptions.append(piece.get("text", ""))
                    elif piece.get("type") == "image":
                        img = piece.get("image", {}).get("data")
                        if img:
                            screenshots.append(img)
            elif isinstance(content, str):
                descriptions.append(content)

        return " ".join(descriptions), screenshots

    def handle_response(self, response, kwargs, event = None):

        # for synchronous streaming responses
        if isinstance(response, Stream):
            return self._handle_stream_response(response, kwargs, event)
        
        # for async streaming responses -- added new
        if isinstance(response, AsyncStream):
            return self._handle_async_stream_response(response, kwargs, event)
        
        # for non streaming responses
        return self._handle_regular_response(response, kwargs, event)
    
    def _handle_stream_response(self, response: Stream, kwargs, event):

        accumulated_reponse = ""

        def generate():

            nonlocal accumulated_reponse

            try:
                for chunk in response:
                    if chunk.type == "content_block_start" and chunk.content_block.type == "text":
                        accumulated_reponse += chunk.content_block.text
                    elif chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                        accumulated_reponse += chunk.delta.text
                    yield chunk

                event.update_event(
                    is_finished=True,
                    is_successful=True,
                    cost_added=None,
                    model=kwargs.get("model"),
                    result=accumulated_reponse
                )

            except Exception as e:
                event.update_event(
                    is_finished=True,
                    result=f"anthropic Error: {str(e)}",
                    is_successful=False,
                    cost_added=None,
                    model=kwargs.get("model"),
                )
                
                raise

        return generate()
    
    def _handle_async_stream_response(self, response: AsyncStream, kwargs, event):

        accumulated_reponse = ""

        async def agenerate():

            nonlocal accumulated_reponse

            try:
                async for chunk in response:
                    if chunk.type == "content_block_start" and chunk.content_block.type == "text":
                        accumulated_reponse += chunk.content_block.text
                    elif chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                        accumulated_reponse += chunk.delta.text
                    yield chunk

                event.update_event(
                    is_finished=True,
                    is_successful=True,
                    cost_added=None,
                    model=kwargs.get("model"),
                    result=accumulated_reponse
                )

            except Exception as e:

                event.update_event(
                    is_finished=True,
                    result=f"anthropic Error: {str(e)}",
                    is_successful=False,
                    cost_added=None,
                    model=kwargs.get("model"),
                )

                raise
            
        return agenerate()
    

    def _handle_regular_response(self, response, kwargs, event):

        try:
            # extract text
            if hasattr(response, "content") and response.content:
                response_text = response.content[0].text
            else:
                response_text = str(response)

            # calculate token-based cost
            cost = None

            if hasattr(response, "usage"):
                cost = response.usage.input_tokens + response.usage.output_tokens

            event.update_event(result=response_text)
            event.finish_event(
                is_successful=True,
                cost_added=cost,
                model=getattr(response, "model", kwargs.get("model"))
            )

        except Exception as e:
            event.update_event(
                is_finished=True,
                is_successful=False,
                cost_added=None,
                model=kwargs.get("model"),
                result=f"Error processing response: {e}"
            )

            raise

        return response

    def override(self):

        # sync
        self.original_create = Anthropic().messages.create
        def patched_create(*args, **kwargs):

            step = kwargs.pop("step", getattr(self.client.session, "active_step", None))
            description, images = self._format_messages(kwargs.get("messages", []))
            event = None

            if step:
                event = step.create_event(
                    description=description,
                    result="Waiting for response...",
                    screenshots=images
                )

            result = self.original_create(*args, **kwargs)
            return self.handle_response(result, kwargs, event)
        
        Anthropic().messages.create = patched_create

        self.original_create_async = AsyncAnthropic().messages.create

        # async
        self.original_create_async = AsyncAnthropic().messages.create
        async def patched_create_async(*args, **kwargs):

            step = kwargs.pop("step", getattr(self.client.session, "active_step", None))
            description, images = self._format_messages(kwargs.get("messages", []))
            event = None

            if step:
                event = step.create_event(
                    description=description,
                    result="Waiting for response...",
                    screenshots=images
                )

            result = await self.original_create_async(*args, **kwargs)
            return self.handle_response(result, kwargs, event)
        
        AsyncAnthropic.messages.create = patched_create_async


    def undo_override(self):

        if self.original_create:
            Anthropic().messages.create = self.original_create
            self.original_create = None
            
        if self.original_create_async:
            AsyncAnthropic().messages.create = self.original_create_async
            self.original_create_async = None
