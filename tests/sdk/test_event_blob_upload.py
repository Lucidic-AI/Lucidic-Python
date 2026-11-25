"""Payloads over the blob threshold upload their full body to the presigned URL."""
import asyncio
import gzip
import json
from types import SimpleNamespace

import httpx
import pytest

from lucidicai.sdk import event

BLOB_URL = "https://blob.test/put"
LARGE = "x" * 70_000


class FakeEvents:
    def __init__(self):
        self.bodies = []

    def create_event(self, body):
        self.bodies.append(body)
        return {"blob_url": BLOB_URL}

    async def acreate_event(self, body):
        return self.create_event(body)


@pytest.fixture
def blob_env(monkeypatch):
    events = FakeEvents()
    uploads = []

    async def slow_upload(url, data):
        await asyncio.sleep(0.05)
        uploads.append((url, json.loads(gzip.decompress(data))))

    monkeypatch.setattr(event, "_get_event_resource", lambda: events)
    monkeypatch.setattr(event, "get_config", lambda: SimpleNamespace(blob_threshold=65536))
    monkeypatch.setattr(event, "_upload_blob_async", slow_upload)
    return SimpleNamespace(events=events, uploads=uploads)


def test_emit_event_uploads_full_payload(blob_env):
    event.emit_event(type="generic", session_id="sess-1", details=LARGE)
    event.flush(timeout=5)

    assert len(blob_env.uploads) == 1
    url, uploaded = blob_env.uploads[0]
    assert url == BLOB_URL
    assert uploaded["details"] == LARGE
    [body] = blob_env.events.bodies
    assert body["needs_blob"] is True
    assert len(body["payload"]["details"]) <= 200


def test_acreate_event_returns_after_upload(blob_env):
    asyncio.run(event.acreate_event(type="generic", session_id="sess-1", details=LARGE))

    assert len(blob_env.uploads) == 1
    assert blob_env.uploads[0][1]["details"] == LARGE


def test_small_payload_is_sent_inline(blob_env):
    asyncio.run(event.acreate_event(type="generic", session_id="sess-1", details="small"))

    assert blob_env.uploads == []
    [body] = blob_env.events.bodies
    assert body["needs_blob"] is False
    assert body["payload"]["details"] == "small"


def test_upload_falls_back_to_sync_after_interpreter_shutdown(blob_env, monkeypatch):
    sync_uploads = []

    async def shutdown_upload(url, data):
        raise RuntimeError("cannot schedule new futures after interpreter shutdown")

    monkeypatch.setattr(event, "_upload_blob_async", shutdown_upload)
    monkeypatch.setattr(event, "_upload_blob_sync", lambda url, data: sync_uploads.append((url, data)))

    asyncio.run(event.acreate_event(type="generic", session_id="sess-1", details=LARGE))

    assert len(sync_uploads) == 1
    url, data = sync_uploads[0]
    assert url == BLOB_URL
    assert json.loads(gzip.decompress(data))["details"] == LARGE


def test_failed_upload_does_not_raise(blob_env, monkeypatch):
    async def failing_upload(url, data):
        request = httpx.Request("PUT", url)
        raise httpx.HTTPStatusError("forbidden", request=request, response=httpx.Response(403, request=request))

    monkeypatch.setattr(event, "_upload_blob_async", failing_upload)

    event_id = asyncio.run(
        event.acreate_event(type="generic", event_id="evt-1", session_id="sess-1", details=LARGE)
    )

    assert event_id == "evt-1"
