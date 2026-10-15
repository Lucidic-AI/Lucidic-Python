"""LUC-1141 — client.resources.download(session_id, dir) (documents into the workdir)."""
import hashlib
import os

import httpx
import pytest
import respx

from lucidicai.api.documents import DownloadSummary
from lucidicai.api.resources.resources import ResourcesResource
from lucidicai.core.errors import APIError, NotFoundError

_BASE = "https://stub.lucidic.test"
_S3 = "https://bucket.s3.amazonaws.com"


@pytest.fixture
def resources(http):
    return ResourcesResource(http)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _doc(path, data, n=1):
    return {
        "path": path, "sha256": _sha(data), "bytes": len(data),
        "content_type": "text/plain; charset=utf-8", "url": f"{_S3}/{n}?sig=x",
    }


_MSA = b"master services agreement\n"
_NOTES = b"notes\n"


def _listing(*docs):
    return httpx.Response(200, json={"expires_in": 1200, "documents": list(docs)})


def _mock_listing(session_id="s1", *responses):
    route = respx.get(f"{_BASE}/sdk/v2/sessions/{session_id}/documents")
    if len(responses) == 1:
        return route.mock(return_value=responses[0])
    return route.mock(side_effect=list(responses))


def _two_docs():
    return _listing(_doc("contracts/msa.md", _MSA, 1), _doc("notes.txt", _NOTES, 2))


class TestDownload:
    @respx.mock
    def test_writes_every_doc_and_summarises(self, resources, tmp_path):
        route = respx.get(f"{_BASE}/sdk/v2/sessions/a%2Fb/documents").mock(return_value=_two_docs())
        s1 = respx.get(f"{_S3}/1").mock(return_value=httpx.Response(200, content=_MSA))
        respx.get(f"{_S3}/2").mock(return_value=httpx.Response(200, content=_NOTES))
        out = tmp_path / "work"
        got = resources.download("a/b", out)
        assert got == DownloadSummary(
            written=["contracts/msa.md", "notes.txt"], skipped=[], bytes=len(_MSA) + len(_NOTES))
        assert (out / "contracts" / "msa.md").read_bytes() == _MSA
        assert (out / "notes.txt").read_bytes() == _NOTES
        assert route.call_count == 1
        assert "authorization" not in s1.calls.last.request.headers

    @respx.mock
    def test_skips_file_with_matching_sha(self, resources, tmp_path):
        _mock_listing("s1", _two_docs())
        (tmp_path / "notes.txt").write_bytes(_NOTES)
        respx.get(f"{_S3}/1").mock(return_value=httpx.Response(200, content=_MSA))
        s2 = respx.get(f"{_S3}/2")
        got = resources.download("s1", tmp_path)
        assert got.written == ["contracts/msa.md"] and got.skipped == ["notes.txt"]
        assert got.bytes == len(_MSA)
        assert not s2.called

    @respx.mock
    def test_overwrites_stale_file(self, resources, tmp_path):
        _mock_listing("s1", _listing(_doc("notes.txt", _NOTES, 2)))
        (tmp_path / "notes.txt").write_bytes(b"old")
        respx.get(f"{_S3}/2").mock(return_value=httpx.Response(200, content=_NOTES))
        got = resources.download("s1", tmp_path)
        assert got.written == ["notes.txt"]
        assert (tmp_path / "notes.txt").read_bytes() == _NOTES

    @respx.mock
    def test_sha_mismatch_raises_and_writes_nothing(self, resources, tmp_path):
        route = _mock_listing("s1", _listing(_doc("notes.txt", _NOTES, 2)))
        respx.get(f"{_S3}/2").mock(return_value=httpx.Response(200, content=b"tampered"))
        with pytest.raises(APIError, match="sha256 mismatch"):
            resources.download("s1", tmp_path)
        assert os.listdir(tmp_path) == []
        assert route.call_count == 1

    @pytest.mark.parametrize("status,code", [(403, "AccessDenied"), (400, "ExpiredToken")])
    @respx.mock
    def test_expired_url_relists_once(self, resources, tmp_path, status, code):
        route = _mock_listing("s1", _two_docs(), _two_docs())
        s1 = respx.get(f"{_S3}/1").mock(return_value=httpx.Response(200, content=_MSA))
        respx.get(f"{_S3}/2").mock(side_effect=[
            httpx.Response(status, text=f"<Error><Code>{code}</Code></Error>"),
            httpx.Response(200, content=_NOTES),
        ])
        got = resources.download("s1", tmp_path)
        assert got.written == ["contracts/msa.md", "notes.txt"]
        assert route.call_count == 2
        assert s1.call_count == 1

    @respx.mock
    def test_expired_twice_raises(self, resources, tmp_path):
        route = _mock_listing("s1", _two_docs(), _two_docs())
        respx.get(f"{_S3}/1").mock(return_value=httpx.Response(403))
        with pytest.raises(httpx.HTTPStatusError):
            resources.download("s1", tmp_path)
        assert route.call_count == 2

    @respx.mock
    def test_other_s3_error_does_not_relist(self, resources, tmp_path):
        route = _mock_listing("s1", _two_docs())
        respx.get(f"{_S3}/1").mock(return_value=httpx.Response(500))
        with pytest.raises(httpx.HTTPStatusError):
            resources.download("s1", tmp_path)
        assert route.call_count == 1

    @pytest.mark.parametrize("path", ["../x", "/etc/passwd", "a/../../b", "a\\..\\b", "", "."])
    @respx.mock
    def test_unsafe_path_rejected_before_any_fetch(self, resources, tmp_path, path):
        _mock_listing("s1", _listing(_doc("notes.txt", _NOTES, 2), _doc(path, _MSA, 1)))
        s3 = respx.get(url__startswith=_S3)
        with pytest.raises(APIError, match="unsafe document path"):
            resources.download("s1", tmp_path)
        assert not s3.called
        assert os.listdir(tmp_path) == []

    @respx.mock
    def test_symlink_escaping_dir_rejected(self, resources, tmp_path):
        out, outside = tmp_path / "work", tmp_path / "outside"
        out.mkdir()
        outside.mkdir()
        (out / "link").symlink_to(outside, target_is_directory=True)
        _mock_listing("s1", _listing(_doc("link/x.txt", _NOTES, 2)))
        with pytest.raises(APIError, match="unsafe document path"):
            resources.download("s1", out)
        assert os.listdir(outside) == []

    @respx.mock
    def test_duplicate_path_rejected(self, resources, tmp_path):
        _mock_listing("s1", _listing(_doc("notes.txt", _NOTES, 1), _doc("notes.txt", _NOTES, 2)))
        with pytest.raises(APIError, match="duplicate document path"):
            resources.download("s1", tmp_path)

    @respx.mock
    def test_route_404_propagates(self, resources, tmp_path):
        _mock_listing("s1", httpx.Response(404, json={"error": "Session's document corpus is not hydrated"}))
        with pytest.raises(NotFoundError):
            resources.download("s1", tmp_path)


class TestAdownload:
    @pytest.mark.asyncio
    @respx.mock
    async def test_writes_and_summarises(self, resources, tmp_path):
        _mock_listing("s1", _two_docs())
        respx.get(f"{_S3}/1").mock(return_value=httpx.Response(200, content=_MSA))
        respx.get(f"{_S3}/2").mock(return_value=httpx.Response(200, content=_NOTES))
        got = await resources.adownload("s1", tmp_path)
        assert got.written == ["contracts/msa.md", "notes.txt"]
        assert (tmp_path / "contracts" / "msa.md").read_bytes() == _MSA

    @pytest.mark.asyncio
    @respx.mock
    async def test_relists_once_on_403(self, resources, tmp_path):
        route = _mock_listing("s1", _two_docs(), _two_docs())
        respx.get(f"{_S3}/1").mock(side_effect=[httpx.Response(403), httpx.Response(200, content=_MSA)])
        respx.get(f"{_S3}/2").mock(return_value=httpx.Response(200, content=_NOTES))
        got = await resources.adownload("s1", tmp_path)
        assert got.written == ["contracts/msa.md", "notes.txt"]
        assert route.call_count == 2
