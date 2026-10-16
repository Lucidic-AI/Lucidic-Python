"""Webhook signature v2: ``X-Lucidic-Signature: t=<unix>,v1=<hex>[,v1=<hex>]``.

``v1 = HMAC-SHA256(secret, t + "." + body)`` over the raw body, one per signing
secret (two while a secret rotates). A ``t`` more than ``tolerance`` seconds from
the receiver's clock is rejected, so a captured delivery can't be replayed later.
"""
import hashlib
import hmac
import re
import time
from typing import List, Optional, Sequence, Tuple, Union

SIGNATURE_HEADER = "X-Lucidic-Signature"
DEFAULT_TOLERANCE = 300
MAX_SIGNATURES = 5
_T_TEXT = re.compile(r"[0-9]{1,12}")

Secret = Union[str, Sequence[str]]


def secret_keys(secret: Optional[Secret]) -> List[str]:
    """The stripped, non-blank keys in ``secret``. A str is one key, never split into characters."""
    if secret is None:
        return []
    items = [secret] if isinstance(secret, str) else list(secret)
    if not all(isinstance(key, str) for key in items):
        raise TypeError("webhook secret must be a str or a sequence of str")
    return [key.strip() for key in items if key.strip()]


def _v1(key: str, t_text: str, body: bytes) -> str:
    return hmac.new(key.encode("utf-8"), t_text.encode("ascii") + b"." + body, hashlib.sha256).hexdigest()


def sign(secret: str, body: bytes, *, t: Optional[int] = None) -> str:
    """Return the header value for ``body`` under ``secret`` at unix time ``t`` (default: now)."""
    key = secret.strip()
    if not key:
        raise ValueError("webhook secret is empty")
    t_text = str(int(time.time()) if t is None else t)
    if not _T_TEXT.fullmatch(t_text):
        raise ValueError(f"t must be a unix time of 1 to 12 digits, got {t_text}")
    return f"t={t_text},v1={_v1(key, t_text, body)}"


def _parse(header: str) -> Optional[Tuple[str, List[str]]]:
    """``(t_text, [v1, ...])``, or None when the header is malformed."""
    t_text, signatures = None, []
    for part in header.split(","):
        key, sep, value = part.partition("=")
        if not sep:
            return None
        if key == "t":
            if t_text is not None or not _T_TEXT.fullmatch(value):
                return None
            t_text = value
        elif key == "v1":
            signatures.append(value)
            if len(signatures) > MAX_SIGNATURES:
                return None
    if t_text is None or not signatures:
        return None
    return t_text, signatures


def verify(
    secret: Secret,
    body: bytes,
    header: Optional[str],
    *,
    tolerance: int = DEFAULT_TOLERANCE,
    now: Optional[float] = None,
) -> bool:
    """True iff ``header`` carries a fresh ``v1`` of ``body`` under any key in ``secret``.

    Raises:
        ValueError: if ``secret`` holds no key (a configuration bug, not a bad request).
    """
    keys = secret_keys(secret)
    if not keys:
        raise ValueError("webhook secret is empty")
    parsed = _parse(header) if header else None
    if parsed is None:
        return False
    t_text, signatures = parsed
    if abs((time.time() if now is None else now) - int(t_text)) > tolerance:
        return False
    # bytes compare so a non-ASCII header is a 401, not a TypeError
    expected = [_v1(key, t_text, body).encode("ascii") for key in keys]
    return any(hmac.compare_digest(e, s.encode("utf-8")) for e in expected for s in signatures)
