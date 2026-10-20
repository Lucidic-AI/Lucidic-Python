"""LUC-904 — presigned-URL GET fetch-through."""
import httpx
import pytest
import respx

from lucidicai.api.downloads import afetch_presigned, fetch_presigned

_PRESIGNED = "https://s3.example.com/bucket/key?X-Amz-Signature=deadbeef"


class TestFetchPresigned:
    @respx.mock
    def test_returns_bytes(self):
        respx.get(_PRESIGNED).mock(return_value=httpx.Response(200, content=b"blob-bytes"))
        assert fetch_presigned(_PRESIGNED) == b"blob-bytes"

    @respx.mock
    def test_no_auth_header_sent(self):
        # Presigned download bypasses the authed HttpClient — S3 needs no
        # Lucidic Authorization header.
        route = respx.get(_PRESIGNED).mock(return_value=httpx.Response(200, content=b"x"))
        fetch_presigned(_PRESIGNED)
        assert "Authorization" not in route.calls.last.request.headers

    @respx.mock
    def test_expired_url_raises(self):
        respx.get(_PRESIGNED).mock(return_value=httpx.Response(403, text="expired"))
        with pytest.raises(httpx.HTTPStatusError):
            fetch_presigned(_PRESIGNED)

    @respx.mock
    @pytest.mark.asyncio
    async def test_afetch_returns_bytes(self):
        respx.get(_PRESIGNED).mock(return_value=httpx.Response(200, content=b"async-blob"))
        assert await afetch_presigned(_PRESIGNED) == b"async-blob"
