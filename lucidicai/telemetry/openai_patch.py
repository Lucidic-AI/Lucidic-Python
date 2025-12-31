"""OpenAI responses API instrumentation patch.

This module provides instrumentation for OpenAI's responses.parse and responses.create APIs
which are not covered by the standard opentelemetry-instrumentation-openai package.
"""
import functools
import logging
import time
from typing import Any, Callable, Optional, Dict

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanKind

from ..sdk.context import current_session_id, current_parent_event_id
from ..utils.logger import debug, verbose, warning

logger = logging.getLogger("Lucidic")


class OpenAIResponsesPatcher:
    """Patches OpenAI client to instrument responses API methods."""

    def __init__(self, tracer_provider=None):
        """Initialize the patcher.

        Args:
            tracer_provider: OpenTelemetry TracerProvider to use
        """
        self._tracer_provider = tracer_provider or trace.get_tracer_provider()
        self._tracer = self._tracer_provider.get_tracer(__name__)
        self._is_patched = False
        self._original_init = None
        self._client_refs = []  # Keep track of patched clients for cleanup

    def patch(self):
        """Apply the patch to OpenAI client initialization."""
        if self._is_patched:
            debug("[OpenAI Patch] responses API already patched")
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

                # Patch responses API methods
                self._patch_responses_api(client_self)

                # Also patch beta.chat.completions.parse if it exists
                self._patch_beta_api(client_self)

            # Replace the __init__ method
            OpenAI.__init__ = patched_init
            self._original_init = original_init
            self._is_patched = True

            logger.info("[OpenAI Patch] Successfully patched OpenAI client for responses API")

        except ImportError:
            logger.warning("[OpenAI Patch] OpenAI library not installed, skipping patch")
        except Exception as e:
            logger.error(f"[OpenAI Patch] Failed to patch responses API: {e}")

    def _patch_responses_api(self, client):
        """Patch the responses API methods on the client."""
        # Check for client.resources.responses (newer structure)
        if hasattr(client, 'resources') and hasattr(client.resources, 'responses'):
            responses = client.resources.responses
            self._patch_responses_object(responses, "client.resources.responses")

        # Check for client.responses (direct access)
        if hasattr(client, 'responses'):
            responses = client.responses
            self._patch_responses_object(responses, "client.responses")

    def _patch_responses_object(self, responses, location: str):
        """Patch methods on a responses object.

        Args:
            responses: The responses object to patch
            location: String describing where this object is (for logging)
        """
        methods_to_patch = {
            'parse': 'openai.responses.parse',
            'create': 'openai.responses.create'
        }

        for method_name, span_name in methods_to_patch.items():
            if hasattr(responses, method_name):
                original_method = getattr(responses, method_name)
                wrapped_method = self._create_method_wrapper(original_method, span_name)
                setattr(responses, method_name, wrapped_method)

                # Track for cleanup
                self._client_refs.append((responses, method_name, original_method))

                verbose(f"[OpenAI Patch] Patched {location}.{method_name}")

    def _patch_beta_api(self, client):
        """Patch beta.chat.completions.parse if it exists."""
        try:
            if (hasattr(client, 'beta') and
                hasattr(client.beta, 'chat') and
                hasattr(client.beta.chat, 'completions') and
                hasattr(client.beta.chat.completions, 'parse')):

                completions = client.beta.chat.completions
                original_parse = completions.parse

                # Wrap with a slightly different span name for clarity
                wrapped_parse = self._create_method_wrapper(
                    original_parse,
                    'openai.beta.chat.completions.parse'
                )
                completions.parse = wrapped_parse

                # Track for cleanup
                self._client_refs.append((completions, 'parse', original_parse))

                verbose("[OpenAI Patch] Patched beta.chat.completions.parse")

        except Exception as e:
            debug(f"[OpenAI Patch] Could not patch beta API: {e}")

    def _create_method_wrapper(self, original_method: Callable, span_name: str) -> Callable:
        """Create a wrapper for an OpenAI API method.

        Args:
            original_method: The original method to wrap
            span_name: Name for the OpenTelemetry span

        Returns:
            Wrapped method with instrumentation
        """
        @functools.wraps(original_method)
        def wrapper(*args, **kwargs):
            # Create span for tracing
            with self._tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT
            ) as span:
                start_time = time.time()

                try:
                    # Debug log for responses.create to understand the parameters
                    if 'responses.create' in span_name:
                        debug(f"[OpenAI Patch] responses.create called with kwargs keys: {list(kwargs.keys())}")

                    # Extract and process request parameters
                    request_attrs = self._extract_request_attributes(span_name, args, kwargs)

                    # Set span attributes
                    span.set_attribute("gen_ai.system", "openai")
                    span.set_attribute("gen_ai.operation.name", span_name)

                    # Add our instrumentation marker
                    span.set_attribute("lucidic.instrumented", span_name)
                    span.set_attribute("lucidic.patch.version", "2.0")

                    # Set request attributes on span
                    for key, value in request_attrs.items():
                        if value is not None:
                            span.set_attribute(key, value)
                            if 'responses.create' in span_name and ('prompt' in key or 'completion' in key):
                                debug(f"[OpenAI Patch] Set attribute {key}: {str(value)[:100]}")

                    # Call the original method
                    result = original_method(*args, **kwargs)

                    # Process the response
                    self._set_response_attributes(span, result, span_name, start_time)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    # Record error in span
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)

                    # The exporter will handle creating error events from the span
                    raise

        return wrapper

    def _extract_request_attributes(self, span_name: str, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract request attributes based on the API method being called.

        Args:
            span_name: Name of the span/API method
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Dictionary of span attributes to set
        """
        attrs = {}

        # Common attributes
        model = kwargs.get('model', 'unknown')
        attrs['gen_ai.request.model'] = model

        temperature = kwargs.get('temperature')
        if temperature is not None:
            attrs['gen_ai.request.temperature'] = temperature

        # Method-specific handling
        if 'responses.parse' in span_name:
            # Handle responses.parse format
            input_param = kwargs.get('input', [])
            text_format = kwargs.get('text_format')
            instructions = kwargs.get('instructions')

            # Convert input to messages format
            if isinstance(input_param, str):
                messages = [{"role": "user", "content": input_param}]
            elif isinstance(input_param, list):
                messages = input_param
            else:
                messages = []

            if text_format and hasattr(text_format, '__name__'):
                attrs['gen_ai.request.response_format'] = text_format.__name__

            if instructions:
                # Never truncate - large payloads are handled via blob storage
                attrs['gen_ai.request.instructions'] = str(instructions)

        elif 'responses.create' in span_name:
            # Handle responses.create format - it uses 'input' not 'messages'
            input_param = kwargs.get('input', [])

            # Convert input to messages format
            if isinstance(input_param, str):
                messages = [{"role": "user", "content": input_param}]
            elif isinstance(input_param, list):
                messages = input_param
            else:
                messages = []

            # Handle text parameter for structured outputs
            text_format = kwargs.get('text')
            if text_format and hasattr(text_format, '__name__'):
                attrs['gen_ai.request.response_format'] = text_format.__name__

        elif 'completions.parse' in span_name:
            # Handle standard chat completion format
            messages = kwargs.get('messages', [])

            # Handle response_format for structured outputs
            response_format = kwargs.get('response_format')
            if response_format:
                if hasattr(response_format, '__name__'):
                    attrs['gen_ai.request.response_format'] = response_format.__name__
                elif isinstance(response_format, dict):
                    attrs['gen_ai.request.response_format'] = str(response_format)

        else:
            # Fallback: try to get messages from kwargs
            messages = kwargs.get('messages', kwargs.get('input', []))
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]

        # Always set message attributes for proper event creation
        # Large payloads are handled via blob storage
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                attrs[f'gen_ai.prompt.{i}.role'] = role
                # Always include full content - large payloads use blob storage
                attrs[f'gen_ai.prompt.{i}.content'] = str(content)

        return attrs

    def _set_response_attributes(self, span, result, span_name: str, start_time: float):
        """Set response attributes on the span for the exporter to use.

        Args:
            span: OpenTelemetry span
            result: Response from OpenAI
            span_name: Name of the API method
            start_time: Request start time
        """
        duration = time.time() - start_time
        span.set_attribute("lucidic.duration_seconds", duration)

        # Extract output based on response structure
        output_text = None

        # Handle different response formats
        if 'responses.parse' in span_name:
            # responses.parse format
            if hasattr(result, 'output_parsed'):
                output_text = str(result.output_parsed)
            elif hasattr(result, 'parsed'):
                output_text = str(result.parsed)

        elif 'responses.create' in span_name:
            # responses.create returns a Response object with output_text
            if hasattr(result, 'output_text'):
                output_text = result.output_text
            elif hasattr(result, 'output'):
                output_text = result.output
            else:
                # Log what we actually got for debugging
                debug(f"[OpenAI Patch] responses.create result type: {type(result)}")
                debug(f"[OpenAI Patch] responses.create result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")

        elif 'completions.parse' in span_name:
            # Standard chat completion format
            if hasattr(result, 'choices') and result.choices:
                choice = result.choices[0]
                if hasattr(choice, 'message'):
                    msg = choice.message
                    if hasattr(msg, 'parsed'):
                        # Structured output
                        output_text = str(msg.parsed)
                    elif hasattr(msg, 'content'):
                        # Regular content
                        output_text = msg.content
                elif hasattr(choice, 'text'):
                    # Completion format
                    output_text = choice.text

        # Set completion attributes if we have output
        if output_text:
            # Never truncate - large payloads are handled via blob storage
            span.set_attribute("gen_ai.completion.0.role", "assistant")
            span.set_attribute("gen_ai.completion.0.content", output_text)
            debug(f"[OpenAI Patch] Set completion: {output_text[:100]}")
        else:
            debug(f"[OpenAI Patch] No output_text found for {span_name}")

        # Handle usage data
        if hasattr(result, 'usage'):
            usage = result.usage

            # Debug logging
            debug(f"[OpenAI Patch] Usage object type: {type(usage)}")
            debug(f"[OpenAI Patch] Usage attributes: {[attr for attr in dir(usage) if not attr.startswith('_')]}")

            # Extract tokens with proper handling
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

            # Try different ways to access token data
            if hasattr(usage, 'prompt_tokens'):
                prompt_tokens = usage.prompt_tokens
            elif hasattr(usage, 'input_tokens'):
                prompt_tokens = usage.input_tokens

            if hasattr(usage, 'completion_tokens'):
                completion_tokens = usage.completion_tokens
            elif hasattr(usage, 'output_tokens'):
                completion_tokens = usage.output_tokens

            if hasattr(usage, 'total_tokens'):
                total_tokens = usage.total_tokens
            elif prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens

            debug(f"[OpenAI Patch] Extracted tokens - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")

            # Set usage attributes on span
            if prompt_tokens is not None:
                span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
            if completion_tokens is not None:
                span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
            if total_tokens is not None:
                span.set_attribute("gen_ai.usage.total_tokens", total_tokens)

    def unpatch(self):
        """Remove the patch and restore original behavior."""
        if not self._is_patched:
            return

        try:
            # Restore original __init__ if we have it
            if self._original_init:
                import openai
                from openai import OpenAI
                OpenAI.__init__ = self._original_init

            # Restore original methods on tracked clients
            for obj, method_name, original_method in self._client_refs:
                try:
                    setattr(obj, method_name, original_method)
                except (AttributeError, ReferenceError) as e:
                    # Client might have been garbage collected
                    logger.debug(f"[OpenAI Patch] Could not restore {method_name}: {e}")

            self._client_refs.clear()
            self._is_patched = False

            logger.info("[OpenAI Patch] Successfully removed responses API patch")

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