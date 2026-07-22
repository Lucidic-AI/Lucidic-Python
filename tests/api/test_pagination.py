"""LUC-902 — cursor-pagination lazy iterator."""
from dataclasses import dataclass
from typing import Optional

import pytest

from lucidicai.api.models.base import APIModel, CursorPage, _extract_cursor
from lucidicai.api.pagination import apaginate, paginate


@dataclass
class _Item(APIModel):
    id: str
    name: Optional[str] = None


def _page(results, next_cursor):
    nxt = f"https://stub.lucidic.test/things?cursor={next_cursor}" if next_cursor else None
    return {"results": results, "next": nxt, "previous": None}


class TestExtractCursor:
    def test_pulls_cursor_param(self):
        assert _extract_cursor("https://x/things?cursor=abc123&page_size=50") == "abc123"

    def test_none_when_no_next(self):
        assert _extract_cursor(None) is None

    def test_none_when_no_cursor_param(self):
        assert _extract_cursor("https://x/things?page_size=50") is None


class TestPaginate:
    def test_follows_next_across_pages(self):
        pages = {
            None: _page([{"id": "1"}, {"id": "2"}], "c1"),
            "c1": _page([{"id": "3"}], "c2"),
            "c2": _page([{"id": "4"}], None),
        }
        seen_cursors = []

        def fetch(cursor):
            seen_cursors.append(cursor)
            return pages[cursor]

        ids = [item.id for item in paginate(fetch, model=_Item)]
        assert ids == ["1", "2", "3", "4"]
        assert seen_cursors == [None, "c1", "c2"]

    def test_single_page_stops(self):
        calls = []

        def fetch(cursor):
            calls.append(cursor)
            return _page([{"id": "only"}], None)

        items = list(paginate(fetch, model=_Item))
        assert len(items) == 1 and items[0].id == "only"
        assert calls == [None]  # no second request

    def test_raw_dicts_when_no_model(self):
        def fetch(cursor):
            return _page([{"id": "1"}], None)

        items = list(paginate(fetch))
        assert items == [{"id": "1"}]

    def test_empty_results(self):
        def fetch(cursor):
            return {"results": [], "next": None}

        assert list(paginate(fetch, model=_Item)) == []


class TestCursorPage:
    def test_from_body_typed(self):
        page = CursorPage.from_body(_page([{"id": "1"}], "next-cur"), model=_Item)
        assert isinstance(page.results[0], _Item)
        assert page.next_cursor == "next-cur"
        assert page.previous_cursor is None
        assert page.has_next is True

    def test_from_body_last_page(self):
        page = CursorPage.from_body(_page([{"id": "1"}], None), model=_Item)
        assert page.has_next is False


class TestAsyncPaginate:
    @pytest.mark.asyncio
    async def test_apaginate_follows_next(self):
        pages = {
            None: _page([{"id": "1"}], "c1"),
            "c1": _page([{"id": "2"}], None),
        }

        async def fetch(cursor):
            return pages[cursor]

        got = []
        async for item in apaginate(fetch, model=_Item):
            got.append(item.id)
        assert got == ["1", "2"]
