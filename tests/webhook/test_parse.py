"""lucidicai.webhook: verify X-Lucidic-Signature, then return a typed case.

Scheme and payload contract: mvp-build/specs/webhook-signing.md (v1).
"""
import json

import httpx
import pytest
import respx

from lucidicai import LucidicAI, webhook

_SECRET = "test-signing-secret"
_SESSION_ID = "6f1c2b9e-3d4a-4e5f-8a7b-0c1d2e3f4a5b"
_ITEM_ID = "0a0b0c0d-1111-2222-3333-444455556666"
_EXPERIMENT_ID = "e0e0e0e0-aaaa-bbbb-cccc-dddddddddddd"
_EVAL_RUN_ID = "beefbeef-0000-1111-2222-333344445555"
_CHECKPOINT_ID = "c0c0c0c0-9999-8888-7777-666655554444"
_BASE_URL = "https://stub.lucidic.test"


def _wire(payload) -> bytes:
    # the backend's serialisation (webhook_egress.py::sign_body)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True,
                      ensure_ascii=False).encode("utf-8")


def _signed(payload, secret=_SECRET):
    body = _wire(payload) if not isinstance(payload, bytes) else payload
    return body, {webhook.SIGNATURE_HEADER: webhook.sign(secret, body)}


def _full_payload(**overrides):
    payload = {
        "session_id": _SESSION_ID,
        "datasetitem_id": _ITEM_ID,
        "input": {"harvey_task": "corporate-ma/x", "documents": ["a.pdf"]},
        "experiment_id": _EXPERIMENT_ID,
        "eval_run_id": _EVAL_RUN_ID,
        "checkpoint_id": _CHECKPOINT_ID,
    }
    payload.update(overrides)
    return payload


# ---- signing -----------------------------------------------------------------

def test_spec_test_vector():
    body = b'{"session_id":"abc"}'
    expected = "sha256=0e400f505aa8fee24c91b988b3b7a8664934bd161a4389b7492af0523cd60ffe"
    assert webhook.sign("test-signing-secret", body) == expected
    assert webhook.verify("test-signing-secret", body, expected) is True


def test_verify_rejects_a_different_body_or_secret():
    body = b'{"session_id":"abc"}'
    header = webhook.sign(_SECRET, body)
    assert webhook.verify(_SECRET, b'{"session_id":"abd"}', header) is False
    assert webhook.verify("other-secret", body, header) is False


def test_verify_refuses_an_empty_secret():
    # an empty key would make every sender able to sign; that is a config bug.
    with pytest.raises(ValueError):
        webhook.verify("", b"{}", webhook.sign("x", b"{}"))


# ---- parse: signature --------------------------------------------------------

def test_parse_returns_the_typed_case():
    body, headers = _signed(_full_payload())
    case = webhook.parse(body, headers, secret=_SECRET)

    assert case.session_id == _SESSION_ID
    assert case.datasetitem_id == _ITEM_ID
    assert case.input == {"harvey_task": "corporate-ma/x", "documents": ["a.pdf"]}
    assert case.experiment_id == _EXPERIMENT_ID
    assert case.eval_run_id == _EVAL_RUN_ID
    assert case.checkpoint_id == _CHECKPOINT_ID


@pytest.mark.parametrize("header", [
    None,
    "",
    "0e400f505aa8fee24c91b988b3b7a8664934bd161a4389b7492af0523cd60ffe",  # no prefix
    "sha1=0e400f505aa8fee24c91b988b3b7a8664934bd161a4389b7492af0523cd60ffe",
    "sha256=deadbeef",
])
def test_parse_rejects_a_missing_or_malformed_signature(header):
    body = _wire(_full_payload())
    headers = {} if header is None else {webhook.SIGNATURE_HEADER: header}
    with pytest.raises(webhook.WebhookSignatureError) as exc:
        webhook.parse(body, headers, secret=_SECRET)
    assert exc.value.status_code == 401


def test_parse_rejects_uppercase_hex():
    # the scheme is lowercase hex compared exactly; no normalisation.
    body = _wire(_full_payload())
    sig = webhook.sign(_SECRET, body)
    with pytest.raises(webhook.WebhookSignatureError):
        webhook.parse(body, {webhook.SIGNATURE_HEADER: "sha256=" + sig[7:].upper()},
                      secret=_SECRET)


def test_parse_rejects_a_tampered_body():
    body, headers = _signed(_full_payload())
    tampered = body.replace(_ITEM_ID.encode(), _ITEM_ID[::-1].encode())
    with pytest.raises(webhook.WebhookSignatureError):
        webhook.parse(tampered, headers, secret=_SECRET)


def test_parse_rejects_a_body_signed_with_another_secret():
    body, headers = _signed(_full_payload(), secret="someone-else")
    with pytest.raises(webhook.WebhookSignatureError):
        webhook.parse(body, headers, secret=_SECRET)


def test_parse_finds_the_header_case_insensitively():
    body, headers = _signed(_full_payload())
    lowered = {k.lower(): v for k, v in headers.items()}
    assert webhook.parse(body, lowered, secret=_SECRET).session_id == _SESSION_ID


def test_parse_verifies_the_raw_bytes_not_a_reserialisation():
    # spaced, unsorted, non-ASCII: re-serialising the parsed JSON would change
    # the bytes and break the signature.
    raw = ('{ "session_id": "%s", "input": {"q": "café"}, "datasetitem_id": null }'
           % _SESSION_ID).encode("utf-8")
    body, headers = _signed(raw)
    case = webhook.parse(body, headers, secret=_SECRET)
    assert case.input == {"q": "café"}


def test_parse_checks_the_signature_before_reading_json():
    # garbage with a bad signature is a 401, not a 422.
    with pytest.raises(webhook.WebhookSignatureError):
        webhook.parse(b"not json", {webhook.SIGNATURE_HEADER: "sha256=00"},
                      secret=_SECRET)


def test_parse_reads_the_secret_from_the_environment(monkeypatch):
    monkeypatch.setenv("LUCIDIC_WEBHOOK_SECRET", _SECRET)
    body, headers = _signed(_full_payload())
    assert webhook.parse(body, headers).session_id == _SESSION_ID


def test_parse_without_any_secret_is_a_config_error(monkeypatch):
    monkeypatch.delenv("LUCIDIC_WEBHOOK_SECRET", raising=False)
    body, headers = _signed(_full_payload(), secret="")
    with pytest.raises(ValueError):
        webhook.parse(body, headers)


# ---- parse: payload ----------------------------------------------------------

def test_optional_ids_default_to_none_and_checkpoint_may_be_absent():
    payload = {"session_id": _SESSION_ID, "datasetitem_id": None, "input": None,
               "experiment_id": None, "eval_run_id": None}
    body, headers = _signed(payload)
    case = webhook.parse(body, headers, secret=_SECRET)

    assert case.session_id == _SESSION_ID
    assert case.datasetitem_id is None
    assert case.input is None
    assert case.experiment_id is None
    assert case.eval_run_id is None
    assert case.checkpoint_id is None


def test_ids_are_normalised_to_canonical_uuid_strings():
    body, headers = _signed(_full_payload(session_id=_SESSION_ID.upper()))
    assert webhook.parse(body, headers, secret=_SECRET).session_id == _SESSION_ID


def test_unknown_keys_are_kept_read_only_in_extra():
    body, headers = _signed(_full_payload(style="formal", n_turns=3))
    case = webhook.parse(body, headers, secret=_SECRET)

    assert dict(case.extra) == {"style": "formal", "n_turns": 3}
    with pytest.raises(TypeError):
        case.extra["style"] = "casual"


@pytest.mark.parametrize("body", [
    b"not json",
    b"[1, 2]",
    _wire({"datasetitem_id": _ITEM_ID}),                    # no session_id
    _wire({"session_id": "abc"}),                           # not a uuid
    _wire({"session_id": 123}),                             # not a string
    _wire(_full_payload(datasetitem_id="not-a-uuid")),
    _wire(_full_payload(checkpoint_id=None)),               # present means a uuid
])
def test_parse_rejects_a_malformed_case(body):
    _, headers = _signed(body)
    with pytest.raises(webhook.WebhookPayloadError) as exc:
        webhook.parse(body, headers, secret=_SECRET)
    assert exc.value.status_code == 422


# ---- the case feeds sessions.create ------------------------------------------

def test_session_kwargs_link_the_session_to_the_case():
    case = webhook.parse(*_signed(_full_payload()), secret=_SECRET)
    assert case.session_kwargs() == {
        "session_id": _SESSION_ID,
        "datasetitem_id": _ITEM_ID,
        "experiment_id": _EXPERIMENT_ID,
        "checkpoint_id": _CHECKPOINT_ID,
        "tags": [f"eval_run:{_EVAL_RUN_ID}"],
    }


def test_session_kwargs_omit_absent_ids():
    payload = {"session_id": _SESSION_ID, "datasetitem_id": None, "input": None,
               "experiment_id": None, "eval_run_id": None}
    case = webhook.parse(*_signed(payload), secret=_SECRET)
    assert case.session_kwargs() == {"session_id": _SESSION_ID}


@respx.mock
def test_session_kwargs_reach_initsession(monkeypatch):
    monkeypatch.setenv("LUCIDIC_DEBUG", "false")
    monkeypatch.setenv("LUCIDIC_BASE_URL", _BASE_URL)
    monkeypatch.delenv("LUCIDIC_REGION", raising=False)
    route = respx.post(f"{_BASE_URL}/initsession").mock(
        return_value=httpx.Response(201, json={"session_id": _SESSION_ID})
    )
    respx.post(f"{_BASE_URL}/sdk/session-init-fixtures").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.put(f"{_BASE_URL}/updatesession").mock(
        return_value=httpx.Response(200, json={})
    )
    client = LucidicAI(api_key="k", agent_id="aaaa1111-0000-0000-0000-000000000000",
                       production=True)
    case = webhook.parse(*_signed(_full_payload()), secret=_SECRET)
    try:
        client.sessions.create(session_name="case", **case.session_kwargs())
    finally:
        client.close()

    sent = json.loads(route.calls.last.request.content)
    assert sent["session_id"] == _SESSION_ID
    assert sent["datasetitem_id"] == _ITEM_ID
    assert sent["experiment_id"] == _EXPERIMENT_ID
    assert sent["checkpoint_id"] == _CHECKPOINT_ID
    assert sent["tags"] == [f"eval_run:{_EVAL_RUN_ID}"]
