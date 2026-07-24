"""Cursor-pagination iteration for the SDK read surface (LUC-902).

v2 list endpoints return ``{"results": [...], "next": <url|null>,
"previous": <url|null>}`` with ``?page_size`` (default 50, max 200),
``?cursor``, and an allow-listed ``?ordering``. ``paginate`` / ``apaginate``
return a lazy iterator that transparently follows ``next``, yielding typed
items so callers can write ``for agent in client.agents.list(...)``.

Following ``next`` re-issues the page request through the authed
``HttpClient`` (the ``fetch`` callback extracts and forwards the ``cursor``)
rather than hitting the raw ``next`` URL directly, so auth headers, retry, and
base-URL handling are all preserved. For single-page / manual control, call
the endpoint once and wrap the body with ``CursorPage.from_body``.
"""
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterator, Optional, Type

from .models.base import APIModel, CursorPage, _extract_cursor

__all__ = ["paginate", "apaginate", "CursorPage"]

# fetch(cursor) -> raw page body. cursor is None for the first page.
_SyncFetch = Callable[[Optional[str]], Dict[str, Any]]
_AsyncFetch = Callable[[Optional[str]], Awaitable[Dict[str, Any]]]


def _convert(item: Any, model: Optional[Type[APIModel]]) -> Any:
    return model.from_dict(item) if model is not None else item


def paginate(fetch: _SyncFetch, *, model: Optional[Type[APIModel]] = None) -> Iterator[Any]:
    """Lazily yield every item across all pages.

    ``fetch(cursor)`` performs one page request (``cursor=None`` first) and
    returns the raw ``{results, next, previous}`` body. Each item is converted
    via ``model.from_dict`` when a ``model`` is given, else yielded raw.
    """
    cursor: Optional[str] = None
    while True:
        body = fetch(cursor)
        for item in (body.get("results", []) if isinstance(body, dict) else []):
            yield _convert(item, model)
        cursor = _extract_cursor(body.get("next") if isinstance(body, dict) else None)
        if cursor is None:
            return


async def apaginate(fetch: _AsyncFetch, *, model: Optional[Type[APIModel]] = None) -> AsyncIterator[Any]:
    """Async sibling of ``paginate`` — ``async for item in apaginate(...)``."""
    cursor: Optional[str] = None
    while True:
        body = await fetch(cursor)
        for item in (body.get("results", []) if isinstance(body, dict) else []):
            yield _convert(item, model)
        cursor = _extract_cursor(body.get("next") if isinstance(body, dict) else None)
        if cursor is None:
            return
