"""LUC-1141 — client.resources.download(session_id, dir) (documents into the workdir)."""
import hashlib
import logging
import os

import httpx
import pytest
import respx

from lucidicai.api.documents import DownloadSummary
from lucidicai.api.resources.resources import ResourcesResource
from lucidicai.core.errors import APIError, InsufficientScopeError, NotFoundError

_BASE = "https://stub.lucidic.test"
_S3 = "https://bucket.s3.amazonaws.com"
_MSA = b"master services agreement\n"
_NOTES = b"notes\n"


@pytest.fixture
def resources(http):
    return ResourcesResource(http)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _doc(path, data, n):
    return {
        "path": path, "sha256": _sha(data), "bytes": len(data),
        "content_type": "text/plain; charset=utf-8", "url": f"{_S3}/{n}?sig=x",
    }


def _listing(*docs):
    return httpx.Response(200, json={"expires_in": 1200, "documents": list(docs)})


def _mock_listing(*responses):
    route = respx.get(f"{_BASE}/sdk/v2/sessions/s1/documents")
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
        _mock_listing(_two_docs())
        (tmp_path / "notes.txt").write_bytes(_NOTES)
        respx.get(f"{_S3}/1").mock(return_value=httpx.Response(200, content=_MSA))
        s2 = respx.get(f"{_S3}/2")
        got = resources.download("s1", tmp_path)
        assert got.written == ["contracts/msa.md"] and got.skipped == ["notes.txt"]
        assert got.bytes == len(_MSA)
        assert not s2.called

    @respx.mock
    def test_overwrites_stale_file(self, resources, tmp_path):
        _mock_listing(_listing(_doc("notes.txt", _NOTES, 2)))
        (tmp_path / "notes.txt").write_bytes(b"old")
        respx.get(f"{_S3}/2").mock(return_value=httpx.Response(200, content=_NOTES))
        got = resources.download("s1", tmp_path)
        assert got.written == ["notes.txt"]
        assert (tmp_path / "notes.txt").read_bytes() == _NOTES

    @respx.mock
    def test_sha_mismatch_raises_and_writes_nothing(self, resources, tmp_path):
        route = _mock_listing(_listing(_doc("notes.txt", _NOTES, 2)))
        respx.get(f"{_S3}/2").mock(return_value=httpx.Response(200, content=b"tampered"))
        with pytest.raises(APIError, match="sha256 mismatch"):
            resources.download("s1", tmp_path)
        assert os.listdir(tmp_path) == []
        assert route.call_count == 1

    @pytest.mark.parametrize("status,code", [(403, "AccessDenied"), (400, "ExpiredToken")])
    @respx.mock
    def test_expired_url_relists_once(self, resources, tmp_path, status, code):
        fresh = _listing(_doc("contracts/msa.md", _MSA, 3), _doc("notes.txt", _NOTES, 4))
        route = _mock_listing(_two_docs(), fresh)
        s1 = respx.get(f"{_S3}/1").mock(return_value=httpx.Response(200, content=_MSA))
        s2 = respx.get(f"{_S3}/2").mock(
            return_value=httpx.Response(status, text=f"<Error><Code>{code}</Code></Error>"))
        s3 = respx.get(f"{_S3}/3")
        s4 = respx.get(f"{_S3}/4").mock(return_value=httpx.Response(200, content=_NOTES))
        got = resources.download("s1", tmp_path)
        assert got.written == ["contracts/msa.md", "notes.txt"]
        assert route.call_count == 2
        assert (s1.call_count, s2.call_count, s3.call_count, s4.call_count) == (1, 1, 0, 1)

    @respx.mock
    def test_expired_twice_raises(self, resources, tmp_path):
        route = _mock_listing(_two_docs(), _two_docs())
        s1 = respx.get(f"{_S3}/1").mock(return_value=httpx.Response(403))
        s2 = respx.get(f"{_S3}/2")
        with pytest.raises(APIError) as err:
            resources.download("s1", tmp_path)
        assert err.value.status_code == 403
        assert _S3 not in str(err.value)
        assert (route.call_count, s1.call_count, s2.call_count) == (2, 2, 0)

    @respx.mock
    def test_other_s3_error_does_not_relist(self, resources, tmp_path):
        route = _mock_listing(_two_docs())
        respx.get(f"{_S3}/1").mock(return_value=httpx.Response(500))
        with pytest.raises(httpx.HTTPStatusError):
            resources.download("s1", tmp_path)
        assert route.call_count == 1

    @pytest.mark.parametrize("path", ["../x", "/etc/passwd", "a/../../b", "a\\..\\b", "", ".", "a\x00b"])
    @respx.mock
    def test_unsafe_path_rejected_before_any_fetch(self, resources, tmp_path, path):
        _mock_listing(_listing(_doc("notes.txt", _NOTES, 2), _doc(path, _MSA, 1)))
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
        _mock_listing(_listing(_doc("link/x.txt", _NOTES, 2)))
        with pytest.raises(APIError, match="unsafe document path"):
            resources.download("s1", out)
        assert os.listdir(outside) == []

    @pytest.mark.parametrize("other", ["notes.txt", "./notes.txt", ".//notes.txt"])
    @respx.mock
    def test_duplicate_path_rejected(self, resources, tmp_path, other):
        _mock_listing(_listing(_doc("notes.txt", _NOTES, 1), _doc(other, _NOTES, 2)))
        with pytest.raises(APIError, match="duplicate document path"):
            resources.download("s1", tmp_path)

    @respx.mock
    def test_file_and_dir_clash_rejected_before_any_fetch(self, resources, tmp_path):
        _mock_listing(_listing(_doc("a/b", _NOTES, 1), _doc("a", _MSA, 2)))
        s3 = respx.get(url__startswith=_S3)
        with pytest.raises(APIError, match="'a/b' is under another document"):
            resources.download("s1", tmp_path)
        assert not s3.called

    @pytest.mark.parametrize("body", [
        {"expires_in": 1200},
        {"documents": {"path": "x"}},
        {"documents": ["notes.txt"]},
        {"documents": [{"path": "notes.txt", "sha256": _sha(_NOTES)}]},
        {"documents": [{"path": "notes.txt", "sha256": None, "url": f"{_S3}/2"}]},
    ])
    @respx.mock
    def test_malformed_listing_rejected(self, resources, tmp_path, body):
        _mock_listing(httpx.Response(200, json=body))
        s3 = respx.get(url__startswith=_S3)
        with pytest.raises(APIError, match="malformed documents listing"):
            resources.download("s1", tmp_path)
        assert not s3.called
        assert os.listdir(tmp_path) == []

    @respx.mock
    def test_uppercase_sha_still_skips(self, resources, tmp_path):
        doc = _doc("notes.txt", _NOTES, 2)
        _mock_listing(_listing({**doc, "sha256": doc["sha256"].upper()}))
        (tmp_path / "notes.txt").write_bytes(_NOTES)
        s2 = respx.get(f"{_S3}/2")
        assert resources.download("s1", tmp_path).skipped == ["notes.txt"]
        assert not s2.called

    @respx.mock
    def test_part_suffixed_path_is_its_own_file(self, resources, tmp_path):
        _mock_listing(_listing(_doc("x.part", _MSA, 1), _doc("x", _NOTES, 2)))
        respx.get(f"{_S3}/1").mock(return_value=httpx.Response(200, content=_MSA))
        respx.get(f"{_S3}/2").mock(return_value=httpx.Response(200, content=_NOTES))
        assert resources.download("s1", tmp_path).written == ["x.part", "x"]
        assert sorted(os.listdir(tmp_path)) == ["x", "x.part"]
        assert (tmp_path / "x.part").read_bytes() == _MSA

    @pytest.mark.parametrize("message", [
        "Session has no dataset item, so it has no documents",
        "Session's dataset has no document corpus",
        "Session's document corpus is not hydrated",
    ])
    @respx.mock
    def test_no_corpus_404_is_an_empty_summary(self, resources, tmp_path, message):
        _mock_listing(httpx.Response(404, json={"error": message}))
        assert resources.download("s1", tmp_path) == DownloadSummary(written=[], skipped=[], bytes=0)
        assert os.listdir(tmp_path) == []

    @respx.mock
    def test_unknown_session_404_raises(self, resources, tmp_path):
        _mock_listing(httpx.Response(404, json={"error": "Specified session_id not found"}))
        with pytest.raises(NotFoundError):
            resources.download("s1", tmp_path)

    @respx.mock
    def test_other_route_error_raises(self, resources, tmp_path):
        _mock_listing(httpx.Response(403, json={"error": "missing scope"}))
        with pytest.raises(InsufficientScopeError):
            resources.download("s1", tmp_path)

    @pytest.mark.parametrize("umask,mode", [(0o022, 0o644), (0o027, 0o640)])
    @respx.mock
    def test_landed_file_mode_follows_umask(self, resources, tmp_path, umask, mode):
        _mock_listing(_listing(_doc("notes.txt", _NOTES, 2)))
        respx.get(f"{_S3}/2").mock(return_value=httpx.Response(200, content=_NOTES))
        old = os.umask(umask)
        try:
            resources.download("s1", tmp_path)
        finally:
            os.umask(old)
        assert (tmp_path / "notes.txt").stat().st_mode & 0o777 == mode
        assert os.listdir(tmp_path) == ["notes.txt"]

    @respx.mock
    def test_debug_log_redacts_presigned_query(self, resources, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("LUCIDIC_DEBUG", "true")
        caplog.set_level(logging.DEBUG, logger="Lucidic")
        _mock_listing(_two_docs())
        respx.get(f"{_S3}/1").mock(return_value=httpx.Response(200, content=_MSA))
        respx.get(f"{_S3}/2").mock(return_value=httpx.Response(200, content=_NOTES))
        resources.download("s1", tmp_path)
        assert f"{_S3}/1" in caplog.text
        assert "sig=" not in caplog.text


class TestAdownload:
    @pytest.mark.asyncio
    @respx.mock
    async def test_writes_and_summarises(self, resources, tmp_path):
        _mock_listing(_two_docs())
        respx.get(f"{_S3}/1").mock(return_value=httpx.Response(200, content=_MSA))
        respx.get(f"{_S3}/2").mock(return_value=httpx.Response(200, content=_NOTES))
        got = await resources.adownload("s1", tmp_path)
        assert got.written == ["contracts/msa.md", "notes.txt"]
        assert (tmp_path / "contracts" / "msa.md").read_bytes() == _MSA

    @pytest.mark.parametrize("status", [403, 400])
    @pytest.mark.asyncio
    @respx.mock
    async def test_relists_once_on_expiry(self, resources, tmp_path, status):
        route = _mock_listing(_two_docs(), _two_docs())
        respx.get(f"{_S3}/1").mock(side_effect=[httpx.Response(status), httpx.Response(200, content=_MSA)])
        respx.get(f"{_S3}/2").mock(return_value=httpx.Response(200, content=_NOTES))
        got = await resources.adownload("s1", tmp_path)
        assert got.written == ["contracts/msa.md", "notes.txt"]
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_expired_twice_raises(self, resources, tmp_path):
        route = _mock_listing(_two_docs(), _two_docs())
        s1 = respx.get(f"{_S3}/1").mock(return_value=httpx.Response(400))
        with pytest.raises(APIError) as err:
            await resources.adownload("s1", tmp_path)
        assert err.value.status_code == 400 and _S3 not in str(err.value)
        assert (route.call_count, s1.call_count) == (2, 2)

    @pytest.mark.asyncio
    @respx.mock
    async def test_route_404_is_an_empty_summary(self, resources, tmp_path):
        _mock_listing(httpx.Response(404, json={"error": "Session's document corpus is not hydrated"}))
        assert await resources.adownload("s1", tmp_path) == DownloadSummary(written=[], skipped=[], bytes=0)
