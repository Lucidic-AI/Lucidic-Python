"""Centralized logging configuration for Lucidic SDK"""

import logging
import os

# Create a single logger instance for the SDK
logger = logging.getLogger("Lucidic")

# Configure logging only if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[Lucidic] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Set log level based on environment
    if os.getenv("LUCIDIC_DEBUG", "False").lower() == "true":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

# Convenience functions
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)