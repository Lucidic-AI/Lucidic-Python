"""Utility to uninstrument specific OpenAI methods to prevent duplicates.

This module helps prevent the standard OpenTelemetry instrumentation
from creating duplicate spans for methods we're handling ourselves.
"""
import logging

logger = logging.getLogger("Lucidic")


def uninstrument_responses(openai_module):
    """Remove any incorrect instrumentation from responses module.

    The standard OpenTelemetry instrumentation might try to instrument
    responses.create (which doesn't exist) or other responses methods.
    This function removes any such instrumentation.

    Args:
        openai_module: The OpenAI module
    """
    try:
        # Check if responses module exists
        if not hasattr(openai_module, 'resources'):
            return

        resources = openai_module.resources
        if not hasattr(resources, 'responses'):
            return

        responses = resources.responses

        # Check for incorrectly wrapped methods
        methods_to_check = ['create', 'parse']

        for method_name in methods_to_check:
            if hasattr(responses, method_name):
                method = getattr(responses, method_name)

                # Check if it's wrapped (wrapped methods usually have __wrapped__ attribute)
                if hasattr(method, '__wrapped__'):
                    # Restore original
                    original = method.__wrapped__
                    setattr(responses, method_name, original)
                    logger.debug(f"[OpenAI Uninstrument] Removed wrapper from responses.{method_name}")

                # Also check for _original_* attributes (another wrapping pattern)
                original_attr = f'_original_{method_name}'
                if hasattr(responses, original_attr):
                    original = getattr(responses, original_attr)
                    setattr(responses, method_name, original)
                    delattr(responses, original_attr)
                    logger.debug(f"[OpenAI Uninstrument] Restored original responses.{method_name}")

        # Also check the Responses class itself
        if hasattr(responses, 'Responses'):
            Responses = responses.Responses
            for method_name in methods_to_check:
                if hasattr(Responses, method_name):
                    method = getattr(Responses, method_name)
                    if hasattr(method, '__wrapped__'):
                        original = method.__wrapped__
                        setattr(Responses, method_name, original)
                        logger.debug(f"[OpenAI Uninstrument] Removed wrapper from Responses.{method_name}")

    except Exception as e:
        logger.debug(f"[OpenAI Uninstrument] Error while checking responses instrumentation: {e}")


def clean_openai_instrumentation():
    """Clean up any problematic OpenAI instrumentation.

    This should be called after standard instrumentation but before our patches.
    """
    try:
        import openai
        uninstrument_responses(openai)

        # Also check if client instances need cleaning
        if hasattr(openai, 'OpenAI'):
            # The OpenAI class might have wrapped __init__ that creates bad instrumentation
            # We don't want to break it, just ensure responses aren't double-instrumented
            pass

    except ImportError:
        pass  # OpenAI not installed
    except Exception as e:
        logger.debug(f"[OpenAI Uninstrument] Error during cleanup: {e}")