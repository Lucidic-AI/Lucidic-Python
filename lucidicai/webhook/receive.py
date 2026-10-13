"""Framework-neutral receive step: raw body + headers in, verified case out."""
import json
import os
from typing import Mapping, Optional

from .case import WebhookCase, WebhookPayloadError, WebhookSignatureError
from .signing import SIGNATURE_HEADER, verify

SECRET_ENV_VAR = "LUCIDIC_WEBHOOK_SECRET"


def _header(headers: Mapping[str, str], name: str) -> Optional[str]:
    # framework header objects are case-insensitive; a plain dict is not.
    value = headers.get(name)
    if value is not None:
        return value
    wanted = name.lower()
    for key, candidate in headers.items():
        if key.lower() == wanted:
            return candidate
    return None


def parse(body: bytes, headers: Mapping[str, str], *, secret: Optional[str] = None) -> WebhookCase:
    """Verify ``X-Lucidic-Signature`` over the raw body, then return the typed case.

    Pass the exact request bytes (FastAPI/Starlette ``await request.body()``,
    Flask ``request.get_data()``, Django ``request.body``), never re-serialised
    JSON. The signature is checked before the body is decoded.

    Args:
        body: Raw request body bytes.
        headers: Request headers; lookup is case-insensitive.
        secret: Shared webhook secret. Defaults to ``$LUCIDIC_WEBHOOK_SECRET``.

    Raises:
        ValueError: no secret configured (server misconfiguration; reply 500).
        WebhookSignatureError: reply ``.status_code`` (401).
        WebhookPayloadError: reply ``.status_code`` (422).
    """
    key = secret if secret is not None else os.environ.get(SECRET_ENV_VAR, "")
    if not key:
        raise ValueError(f"no webhook secret: pass secret= or set {SECRET_ENV_VAR}")
    if not verify(key, body, _header(headers, SIGNATURE_HEADER)):
        raise WebhookSignatureError(f"missing or invalid {SIGNATURE_HEADER}")
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError) as e:
        raise WebhookPayloadError(f"body is not JSON: {e}") from None
    return WebhookCase.from_payload(payload)
