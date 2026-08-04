"""EventBuilder.build strict-format regression (LUC-941).

A caller-supplied `payload` dict (strict format) must still map the identity fields to
the backend wire names (client_event_id / client_parent_event_id). Previously the strict
path returned params verbatim, so event_id/parent_event_id never became client_*,
dropping parent-linking and idempotency dedup.
"""
from lucidicai.sdk.event_builder import EventBuilder


def _public_params(**over):
    p = {
        "type": "generic",
        "event_id": "ev-1",
        "parent_event_id": "ev-parent",
        "session_id": "sess-1",
        "occurred_at": "2026-01-01T00:00:00+00:00",
        "payload": {"details": "hi"},
    }
    p.update(over)
    return p


def test_strict_format_maps_identity_fields_to_wire_names():
    out = EventBuilder.build(_public_params())
    # the whole point: event_id/parent_event_id -> client_* (what the backend reads)
    assert out["client_event_id"] == "ev-1"
    assert out["client_parent_event_id"] == "ev-parent"
    # the raw public names must not leak through untranslated
    assert "event_id" not in out
    assert "parent_event_id" not in out
    # payload passed through verbatim; base fields preserved
    assert out["payload"] == {"details": "hi"}
    assert out["type"] == "generic"
    assert out["session_id"] == "sess-1"
    assert out["occurred_at"] == "2026-01-01T00:00:00+00:00"


def test_strict_and_non_strict_produce_the_same_identity_mapping():
    # The divergence between the two paths WAS the bug — they must now agree.
    strict = EventBuilder.build(_public_params(payload={"details": "hi"}))
    non_strict = EventBuilder.build({
        "type": "generic", "event_id": "ev-1", "parent_event_id": "ev-parent",
        "session_id": "sess-1", "occurred_at": "2026-01-01T00:00:00+00:00", "details": "hi",
    })
    for k in ("client_event_id", "client_parent_event_id", "session_id", "type"):
        assert strict[k] == non_strict[k]


def test_strict_format_without_parent_is_fine():
    out = EventBuilder.build(_public_params(parent_event_id=None))
    assert out["client_event_id"] == "ev-1"
    assert out.get("client_parent_event_id") is None  # no-parent, mapped as None


def test_strict_format_empty_payload_still_maps_ids():
    # payload={} is the exact LUC-941 / SE-E4 trigger — still strict (isinstance({}, dict)),
    # ids still mapped.
    out = EventBuilder.build(_public_params(payload={}))
    assert out["client_event_id"] == "ev-1"
    assert out["client_parent_event_id"] == "ev-parent"
    assert out["payload"] == {}


def test_strict_format_passes_through_duration_tags_metadata():
    # these base fields must survive the strict path (most likely to silently regress
    # if _extract_base_params is edited later).
    out = EventBuilder.build(_public_params(duration=1.5, tags=["a"], metadata={"k": "v"}))
    assert out["duration"] == 1.5
    assert out["tags"] == ["a"]
    assert out["metadata"] == {"k": "v"}
