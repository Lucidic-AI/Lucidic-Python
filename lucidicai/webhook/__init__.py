"""Lucidic eval-run webhook signatures (``X-Lucidic-Signature``).

``lucidicai.load(body, signature)`` verifies and parses a delivery; use
``sign``/``verify`` directly only when building your own receiver or tests.
"""
from .signing import SIGNATURE_HEADER, sign, verify

SECRET_ENV_VAR = "LUCIDIC_WEBHOOK_SECRET"

__all__ = ["sign", "verify", "SIGNATURE_HEADER", "SECRET_ENV_VAR"]
