"""Webhook signature v1: HMAC-SHA256 over the raw body, keyed by the shared
secret, sent as ``X-Lucidic-Signature: sha256=<lowercase hex>``.
"""
import hashlib
import hmac
from typing import Optional

SIGNATURE_HEADER = "X-Lucidic-Signature"
_PREFIX = "sha256="


def sign(secret: str, body: bytes) -> str:
    """Return the header value for ``body`` under ``secret``."""
    return _PREFIX + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def verify(secret: str, body: bytes, header: Optional[str]) -> bool:
    """True iff ``header`` is the exact signature of ``body``. Constant-time compare.

    Raises:
        ValueError: if ``secret`` is empty (a configuration bug, not a bad request).
    """
    if not secret:
        raise ValueError("webhook secret is empty")
    if not header or not header.startswith(_PREFIX):
        return False
    # bytes compare so a non-ASCII header is a 401, not a TypeError
    return hmac.compare_digest(sign(secret, body).encode("ascii"), header.encode("utf-8"))
