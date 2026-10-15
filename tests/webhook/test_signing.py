"""lucidicai.webhook sign/verify: scheme v1 of mvp-build/specs/webhook-signing.md."""
import pytest

from lucidicai import webhook

_SECRET = "test-signing-secret"
_BODY = b'{"session_id":"abc"}'
_VECTOR = "sha256=0e400f505aa8fee24c91b988b3b7a8664934bd161a4389b7492af0523cd60ffe"


def test_spec_test_vector():
    assert webhook.sign(_SECRET, _BODY) == _VECTOR
    assert webhook.verify(_SECRET, _BODY, _VECTOR) is True


def test_verify_rejects_a_different_body_or_secret():
    assert webhook.verify(_SECRET, b'{"session_id":"abd"}', _VECTOR) is False
    assert webhook.verify("other-secret", _BODY, _VECTOR) is False


def test_verify_refuses_an_empty_secret():
    # an empty key would let every sender sign; that is a config bug.
    with pytest.raises(ValueError):
        webhook.verify("", _BODY, _VECTOR)


@pytest.mark.parametrize("header", [
    None,
    "",
    _VECTOR[len("sha256="):],                 # no prefix
    "sha1=" + _VECTOR[len("sha256="):],
    "sha256=deadbeef",
])
def test_verify_rejects_a_missing_or_malformed_header(header):
    assert webhook.verify(_SECRET, _BODY, header) is False


def test_verify_treats_a_non_ascii_header_as_a_mismatch():
    # starlette decodes headers as latin-1; this must not escape as a TypeError (500).
    assert webhook.verify(_SECRET, _BODY, "sha256=é") is False


def test_verify_rejects_uppercase_hex():
    # the scheme is lowercase hex compared exactly; no normalisation.
    assert webhook.verify(_SECRET, _BODY, "sha256=" + _VECTOR[7:].upper()) is False
