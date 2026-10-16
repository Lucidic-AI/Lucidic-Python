"""Presigned-URL download fetch-through (LUC-904).

Backend read paths — event raw blobs (``GET .../events/{id}?raw=true``) and
training-module export downloads (``GET .../download-url``) — return
short-lived presigned S3 GET URLs (5-20 min). These helpers download the bytes
directly from S3 with **no** Lucidic auth header, mirroring the presigned
PUT-upload path in ``sdk/event.py``.

Never cache a presigned URL — it expires. Fetch the presigned URL fresh from
the API each time and download immediately.
"""
from typing import Optional

import httpx

# Blob downloads can be large; give them a more generous default than the
# 30s API timeout, but still bounded.
DEFAULT_TIMEOUT = 60.0


def fetch_presigned(url: str, *, timeout: Optional[float] = DEFAULT_TIMEOUT) -> bytes:
    """Download and return the bytes at a presigned URL (sync).

    Raises ``httpx.HTTPStatusError`` on a non-2xx from S3 (e.g. an expired
    URL → 403). The caller re-requests a fresh presigned URL and retries.
    """
    resp = httpx.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


async def afetch_presigned(url: str, *, timeout: Optional[float] = DEFAULT_TIMEOUT) -> bytes:
    """Async sibling of ``fetch_presigned``."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content
