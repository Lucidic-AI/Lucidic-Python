"""OpenAI responses.parse instrumentation patch.

This module provides instrumentation for OpenAI's responses.parse API
which is not covered by the standard opentelemetry-instrumentation-openai package.
"""
import functools
import logging
import time
from typing import Any, Callable, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanKind

from ..sdk.context import current_session_id, current_parent_event_id
from ..utils.logger import debug, verbose, warning

logger = logging.getLogger("Lucidic")


class OpenAIResponsesPatcher:
    """Patches OpenAI client to instrument responses.parse method."""

    def __init__(self, tracer_provider=None):
        """Initialize the patcher.

        Args:
            tracer_provider: OpenTelemetry TracerProvider to use
        """
        self._tracer_provider = tracer_provider or trace.get_tracer_provider()
        self._tracer = self._tracer_provider.get_tracer(__name__)
        self._is_patched = False
        self._original_parse = None
        self._client_refs = []  # Keep track of patched clients for cleanup

    def patch(self):
        """Apply the patch to OpenAI client initialization."""
        if self._is_patched:
            debug("[OpenAI Patch] responses.parse already patched")
            return

        try:
            import openai
            from openai import OpenAI

            # Store the original __init__
            original_init = OpenAI.__init__

            @functools.wraps(original_init)
            def patched_init(client_self, *args, **kwargs):
                # Call original initialization
                original_init(client_self, *args, **kwargs)

                # Patch the responses.parse method on this specific instance
                if hasattr(client_self, 'resources') and hasattr(client_self.resources, 'responses'):
                    responses = client_self.resources.responses
                    if hasattr(responses, 'parse'):
                        # Store original and apply wrapper
                        original_parse = responses.parse
                        responses.parse = self._create_parse_wrapper(original_parse)

                        # Track this client for cleanup
                        self._client_refs.append((responses, original_parse))

                        verbose("[OpenAI Patch] Patched responses.parse on client instance")

                # Also patch the direct access if available
                if hasattr(client_self, 'responses') and hasattr(client_self.responses, 'parse'):
                    original_parse = client_self.responses.parse
                    client_self.responses.parse = self._create_parse_wrapper(original_parse)
                    self._client_refs.append((client_self.responses, original_parse))
                    verbose("[OpenAI Patch] Patched client.responses.parse")

            # Replace the __init__ method
            OpenAI.__init__ = patched_init
            self._original_init = original_init
            self._is_patched = True

            logger.info("[OpenAI Patch] Successfully patched OpenAI client for responses.parse")

        except ImportError:
            logger.warning("[OpenAI Patch] OpenAI library not installed, skipping patch")
        except Exception as e:
            logger.error(f"[OpenAI Patch] Failed to patch responses.parse: {e}")

    def _create_parse_wrapper(self, original_method: Callable) -> Callable:
        """Create a wrapper for the responses.parse method.

        Args:
            original_method: The original parse method to wrap

        Returns:
            Wrapped method with instrumentation
        """
        @functools.wraps(original_method)
        def wrapper(**kwargs):
            # Create span for tracing
            with self._tracer.start_as_current_span(
                "openai.responses.parse",
                kind=SpanKind.CLIENT
            ) as span:
                start_time = time.time()

                try:
                    # Extract request parameters
                    model = kwargs.get('model', 'unknown')
                    temperature = kwargs.get('temperature', 1.0)
                    input_param = kwargs.get('input', [])
                    text_format = kwargs.get('text_format')
                    instructions = kwargs.get('instructions')

                    # Convert input to messages format if needed
                    if isinstance(input_param, str):
                        messages = [{"role": "user", "content": input_param}]
                    elif isinstance(input_param, list):
                        messages = input_param
                    else:
                        messages = []

                    # Set span attributes
                    span.set_attribute("gen_ai.system", "openai")
                    span.set_attribute("gen_ai.request.model", model)
                    span.set_attribute("gen_ai.request.temperature", temperature)
                    span.set_attribute("gen_ai.operation.name", "responses.parse")

                    # Add a unique marker for our instrumentation
                    span.set_attribute("lucidic.instrumented", "responses.parse")
                    span.set_attribute("lucidic.patch.version", "1.0")

                    if text_format and hasattr(text_format, '__name__'):
                        span.set_attribute("gen_ai.request.response_format", text_format.__name__)

                    if instructions:
                        span.set_attribute("gen_ai.request.instructions", str(instructions)[:500])

                    # Always set message attributes for proper event creation
                    # Content capture setting only affects how much detail we include
                    capture_full_content = self._should_capture_content()
                    for i, msg in enumerate(messages[:5]):  # Limit to first 5 messages
                        if isinstance(msg, dict):
                            role = msg.get('role', 'user')
                            content = msg.get('content', '')
                            span.set_attribute(f"gen_ai.prompt.{i}.role", role)
                            # Include full content if capture enabled, otherwise just a snippet
                            if capture_full_content:
                                span.set_attribute(f"gen_ai.prompt.{i}.content", str(content)[:2048])
                            else:
                                span.set_attribute(f"gen_ai.prompt.{i}.content", str(content)[:100] + ("..." if len(str(content)) > 100 else ""))

                    # Call the original method
                    result = original_method(**kwargs)

                    # Process the response and set attributes on span
                    # DO NOT create events here - let the exporter handle it
                    self._set_response_attributes(span, result, model, messages, start_time, text_format)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    # Record error in span
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)

                    # The exporter will handle creating error events from the span
                    # Re-raise the exception
                    raise

        return wrapper

    def _set_response_attributes(self, span, result, model: str, messages: list, start_time: float, text_format):
        """Set response attributes on the span for the exporter to use.

        Args:
            span: OpenTelemetry span
            result: Response from OpenAI
            model: Model name
            messages: Input messages
            start_time: Request start time
            text_format: Response format (Pydantic model)
        """
        duration = time.time() - start_time

        # Extract output
        output_text = None

        # Handle structured output response
        if hasattr(result, 'output_parsed'):
            output_text = str(result.output_parsed)

            # Always set completion attributes so the exporter can extract them
            span.set_attribute("gen_ai.completion.0.role", "assistant")
            span.set_attribute("gen_ai.completion.0.content", output_text[:2048])

        # Handle usage data
        if hasattr(result, 'usage'):
            usage = result.usage
            # Set usage attributes on span
            span.set_attribute("gen_ai.usage.prompt_tokens", getattr(usage, 'prompt_tokens', 0))
            span.set_attribute("gen_ai.usage.completion_tokens", getattr(usage, 'completion_tokens', 0))
            span.set_attribute("gen_ai.usage.total_tokens", getattr(usage, 'total_tokens', 0))

        # Set additional metadata for the exporter
        if text_format and hasattr(text_format, '__name__'):
            span.set_attribute("lucidic.response_format", text_format.__name__)

        # Set duration as attribute
        span.set_attribute("lucidic.duration_seconds", duration)

        # DO NOT create events here - the exporter will handle it when processing the span


    def _should_capture_content(self) -> bool:
        """Check if message content should be captured.

        Returns:
            True if content capture is enabled
        """
        import os
        # check OTEL standard env var
        otel_capture = os.getenv('OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT', 'false')
        # check Lucidic-specific env var
        lucidic_capture = os.getenv('LUCIDIC_CAPTURE_CONTENT', 'false')

        return otel_capture.lower() == 'true' or lucidic_capture.lower() == 'true'

    def unpatch(self):
        """Remove the patch and restore original behavior."""
        if not self._is_patched:
            return

        try:
            # restore original __init__ if we have it
            if hasattr(self, '_original_init'):
                import openai
                from openai import OpenAI
                OpenAI.__init__ = self._original_init

            # restore original parse methods on tracked clients
            for responses_obj, original_parse in self._client_refs:
                try:
                    responses_obj.parse = original_parse
                except:
                    pass  # Client might have been garbage collected

            self._client_refs.clear()
            self._is_patched = False

            logger.info("[OpenAI Patch] Successfully removed responses.parse patch")

        except Exception as e:
            logger.error(f"[OpenAI Patch] Failed to unpatch: {e}")


# Global singleton instance
_patcher_instance: Optional[OpenAIResponsesPatcher] = None


def get_responses_patcher(tracer_provider=None) -> OpenAIResponsesPatcher:
    """Get or create the global patcher instance.

    Args:
        tracer_provider: OpenTelemetry TracerProvider

    Returns:
        The singleton patcher instance
    """
    global _patcher_instance
    if _patcher_instance is None:
        _patcher_instance = OpenAIResponsesPatcher(tracer_provider)
    return _patcher_instance