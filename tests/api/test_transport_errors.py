"""LUC-900 — central error-envelope decode + typed exception hierarchy.

Backend is mocked via respx at the HTTP boundary so we exercise the real
``HttpClient._handle_response`` → ``exception_from_response`` path and assert
each envelope/status maps to the right typed ``LucidicError`` subclass.
"""
import httpx
import pytest
import respx

from lucidicai.core.errors import (
    APIError,
    APIKeyVerificationError,
    AuthError,
    ConflictError,
    exception_from_response,
    InsufficientScopeError,
    LucidicError,
    LucidicUnknownToolError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    ValidationError,
)

_URL = "https://stub.lucidic.test/agents"


def _mock(status, json=None, text=None, headers=None):
    if json is not None:
        return httpx.Response(status, json=json, headers=headers or {})
    return httpx.Response(status, text=text or "", headers=headers or {})


class TestTypedDecode:
    @respx.mock
    def test_400_error_string_is_validation_error(self, http):
        respx.get(_URL).mock(return_value=_mock(400, {"error": "bad input"}))
        with pytest.raises(ValidationError) as ei:
            http.get("agents")
        assert str(ei.value) == "bad input"
        assert ei.value.status_code == 400

    @respx.mock
    def test_400_validation_envelope_carries_details(self, http):
        respx.get(_URL).mock(return_value=_mock(
            400, {"error": "Validation failed", "details": {"name": ["required"]}}))
        with pytest.raises(ValidationError) as ei:
            http.get("agents")
        assert ei.value.details == {"name": ["required"]}

    @respx.mock
    def test_422_errors_dict_is_validation_error_with_details(self, http):
        respx.get(_URL).mock(return_value=_mock(
            422, {"errors": {"tools": ["bad source_hash"]}}))
        with pytest.raises(ValidationError) as ei:
            http.get("agents")
        assert ei.value.details == {"tools": ["bad source_hash"]}

    @respx.mock
    def test_401_is_api_key_verification_error(self, http):
        respx.get(_URL).mock(return_value=_mock(401, {"detail": "no key"}))
        with pytest.raises(APIKeyVerificationError) as ei:
            http.get("agents")
        # Still an AuthError so `except AuthError` works.
        assert isinstance(ei.value, AuthError)

    @respx.mock
    def test_403_is_insufficient_scope_and_names_scope(self, http):
        respx.get(_URL).mock(return_value=_mock(
            403, {"error": "missing scope", "required_scope": "agent:read"}))
        with pytest.raises(InsufficientScopeError) as ei:
            http.get("agents")
        assert ei.value.required_scope == "agent:read"

    @respx.mock
    def test_404_is_not_found(self, http):
        respx.get(_URL).mock(return_value=_mock(404, {"error": "gone"}))
        with pytest.raises(NotFoundError):
            http.get("agents")

    @respx.mock
    def test_409_is_conflict(self, http):
        respx.get(_URL).mock(return_value=_mock(409, {"error": "duplicate"}))
        with pytest.raises(ConflictError):
            http.get("agents")

    @respx.mock
    def test_500_json_is_generic_api_error(self, http):
        respx.get(_URL).mock(return_value=_mock(500, {"detail": "boom"}))
        with pytest.raises(APIError) as ei:
            http.get("agents")
        assert ei.value.status_code == 500
        assert str(ei.value) == "boom"

    @respx.mock
    def test_non_json_body_is_api_error_with_text(self, http):
        respx.get(_URL).mock(return_value=_mock(500, text="<html>500</html>"))
        with pytest.raises(APIError) as ei:
            http.get("agents")
        assert ei.value.response_text == "<html>500</html>"

    @respx.mock
    def test_mock_call_code_family_still_produced_centrally(self, http):
        # A `{"error": {"code"}}` body maps to the mock-call typed family even
        # from a generic endpoint — the family is keyed by shape, not path.
        respx.get(_URL).mock(return_value=_mock(
            404, {"error": {"code": "unknown_tool", "detail": "no tool"}}))
        with pytest.raises(LucidicUnknownToolError):
            http.get("agents")

    @respx.mock
    def test_every_typed_error_is_a_lucidic_error(self, http):
        respx.get(_URL).mock(return_value=_mock(404, {"error": "x"}))
        with pytest.raises(LucidicError):
            http.get("agents")


class TestDecoderUnit:
    """Direct unit coverage of exception_from_response for the statuses the
    transport retries (429/503), which the HttpClient path would retry first."""

    def test_429_maps_to_rate_limit(self):
        exc = exception_from_response(429, {"error": "slow down"}, retry_after=1.5)
        assert isinstance(exc, RateLimitError)
        assert exc.retry_after == 1.5

    def test_503_maps_to_service_unavailable(self):
        exc = exception_from_response(503, {"error": "degraded"}, retry_after=2.0)
        assert isinstance(exc, ServiceUnavailableError)
        assert exc.retry_after == 2.0

    def test_unmapped_status_is_api_error(self):
        exc = exception_from_response(418, {"error": "teapot"})
        assert type(exc) is APIError
        assert exc.status_code == 418


class TestAsyncDecode:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_path_raises_typed(self, http):
        respx.get(_URL).mock(return_value=_mock(404, {"error": "gone"}))
        with pytest.raises(NotFoundError):
            await http.aget("agents")
