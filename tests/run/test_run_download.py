"""run.download(dir): the case documents via client.resources.download for the run's session."""
import hashlib

import httpx
import pytest
import respx

import lucidicai as lucidic

from .conftest import BASE_URL, SESSION_ID, full_payload

_S3 = "https://bucket.s3.amazonaws.com/1"
_NOTES = b"notes\n"


@pytest.fixture
def run(client):
    return lucidic.load(client=client, **{k: v for k, v in full_payload().items() if k != "input"})


def _mock_documents():
    doc = {"path": "notes.txt", "sha256": hashlib.sha256(_NOTES).hexdigest(), "url": f"{_S3}?sig=x"}
    respx.get(f"{BASE_URL}/sdk/v2/sessions/{SESSION_ID}/documents").mock(
        return_value=httpx.Response(200, json={"expires_in": 1200, "documents": [doc]}))
    respx.get(_S3).mock(return_value=httpx.Response(200, content=_NOTES))


@respx.mock
def test_download_lands_the_case_documents(run, tmp_path):
    _mock_documents()
    assert run.download(tmp_path).written == ["notes.txt"]
    assert (tmp_path / "notes.txt").read_bytes() == _NOTES


@pytest.mark.asyncio
@respx.mock
async def test_adownload_lands_the_case_documents(run, tmp_path):
    _mock_documents()
    assert (await run.adownload(tmp_path)).written == ["notes.txt"]
    assert (tmp_path / "notes.txt").read_bytes() == _NOTES


@respx.mock
def test_download_without_a_corpus_is_empty(run, tmp_path):
    respx.get(f"{BASE_URL}/sdk/v2/sessions/{SESSION_ID}/documents").mock(
        return_value=httpx.Response(404, json={"error": "Session's dataset has no document corpus"}))
    assert run.download(tmp_path).written == []
