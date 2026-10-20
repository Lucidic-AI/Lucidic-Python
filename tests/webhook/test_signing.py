"""lucidicai.webhook sign/verify: signature scheme v2 (``t=<unix>,v1=<hex>``)."""
import pytest

from lucidicai import webhook

_SECRET = "test-signing-secret"
_BODY = b'{"session_id":"abc"}'
_T = 1760000000
# shared with the backend and Harvey tests; computed independently with
# printf '1760000000.{"session_id":"abc"}' | openssl dgst -sha256 -hmac test-signing-secret
_V1 = "943cdd9487def092eab8fa18fa1ea3f97411fbf567bb8d59cea5759a0b28c246"
_VECTOR = f"t={_T},v1={_V1}"


def _verify(header, secret=_SECRET, body=_BODY, now=_T, **kwargs):
    return webhook.verify(secret, body, header, now=now, **kwargs)


def test_spec_test_vector():
    assert webhook.sign(_SECRET, _BODY, t=_T) == _VECTOR
    assert _verify(_VECTOR) is True


def test_sign_defaults_to_now():
    header = webhook.sign(_SECRET, _BODY)
    assert webhook.verify(_SECRET, _BODY, header) is True


def test_verify_rejects_a_different_body_or_secret():
    assert _verify(_VECTOR, body=b'{"session_id":"abd"}') is False
    assert _verify(_VECTOR, secret="other-secret") is False


def test_t_is_bound_into_the_mac():
    assert _verify(f"t={_T + 1},v1={_V1}", now=_T + 1) is False


@pytest.mark.parametrize("skew, ok", [(300, True), (-300, True), (301, False), (-301, False)])
def test_window_is_300_seconds_either_way(skew, ok):
    # a negative skew is a t from the receiver's future
    assert _verify(_VECTOR, now=_T + skew) is ok


def test_tolerance_is_configurable():
    assert _verify(_VECTOR, now=_T + 10, tolerance=5) is False
    assert _verify(_VECTOR, now=_T + 10, tolerance=10) is True


@pytest.mark.parametrize("secret", ["", [], [""], ["", ""], None])
def test_verify_refuses_an_empty_secret(secret):
    # an empty key would let every sender sign; that is a config bug.
    with pytest.raises(ValueError):
        _verify(_VECTOR, secret=secret)


def test_a_str_secret_is_one_key_not_characters():
    # a one-character key list would be trivially forgeable
    forged = webhook.sign("t", _BODY, t=_T)
    assert _verify(forged, secret="test") is False
    assert webhook.secret_keys("a,b") == ["a,b"]
    assert webhook.secret_keys([" a ", "  ", ""]) == ["a"]


def test_any_key_in_a_list_matches():
    assert _verify(_VECTOR, secret=["new-secret", _SECRET]) is True
    assert _verify(_VECTOR, secret=("new-secret", "older")) is False


def test_any_v1_in_the_header_matches():
    other = webhook.sign("other-secret", _BODY, t=_T).split(",")[1]
    assert _verify(f"t={_T},{other},v1={_V1}") is True
    assert _verify(f"v1={_V1},t={_T}") is True   # any order


def test_unknown_keys_are_ignored():
    assert _verify(f"t={_T},v0=abc,v1={_V1}") is True


def test_five_v1_values_are_read_six_are_rejected():
    pad = ",".join(["v1=" + "0" * 64] * 4)
    assert _verify(f"t={_T},{pad},v1={_V1}") is True
    assert _verify(f"t={_T},{pad},v1=00,v1={_V1}") is False


@pytest.mark.parametrize("t_text", ["+1760000000", " 1760000000", "1_760_000_000", "1760000000.0",
                                    "1760000000000", "", "-1", "１７６０００００００", "1760000000\n"])
def test_t_must_be_1_to_12_ascii_digits(t_text):
    assert _verify(f"t={t_text},v1={_V1}") is False


@pytest.mark.parametrize("header", [
    None,
    "",
    _V1,                                        # no t
    f"t={_T}",                                  # no v1
    f"t={_T},t={_T},v1={_V1}",                  # two t
    f"t={_T},v1={_V1},junk",                    # a part without =
    f"t={_T}, v1={_V1}",                        # whitespace
    "sha256=" + _V1,                            # the old v1 scheme
    f"t={_T},v1={_V1.upper()}",                 # lowercase hex, compared exactly
])
def test_verify_rejects_a_missing_or_malformed_header(header):
    assert _verify(header) is False


def test_verify_treats_a_non_ascii_header_as_a_mismatch():
    # starlette decodes headers as latin-1; this must not escape as a TypeError (500).
    assert _verify(f"t={_T},v1=é") is False


def test_a_padded_secret_round_trips():
    header = webhook.sign(f"  {_SECRET}\n", _BODY, t=_T)
    assert header == _VECTOR
    assert _verify(header, secret=f" {_SECRET} ") is True


@pytest.mark.parametrize("secret", ["", "  "])
def test_sign_refuses_an_empty_secret(secret):
    with pytest.raises(ValueError):
        webhook.sign(secret, _BODY, t=_T)


@pytest.mark.parametrize("t", [-1, 10**12])
def test_sign_refuses_a_t_no_verifier_accepts(t):
    with pytest.raises(ValueError):
        webhook.sign(_SECRET, _BODY, t=t)


@pytest.mark.parametrize("secret", [b"test-signing-secret", [b"a"], ["a", 1]])
def test_a_non_str_secret_is_a_type_error(secret):
    with pytest.raises(TypeError):
        _verify(_VECTOR, secret=secret)
