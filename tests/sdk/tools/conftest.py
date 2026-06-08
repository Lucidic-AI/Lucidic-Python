"""Shared fixtures for SDK tools tests.

Builds a real ``HttpClient`` against a deterministic stub base URL.
We construct ``SDKConfig`` and ``NetworkConfig`` directly rather than
going through ``from_env`` — the latter calls ``load_dotenv()`` and
honors ``LUCIDIC_DEBUG`` which would silently redirect requests at
``http://localhost:8000/api`` if the developer has that env var set
(common when running ``sst dev`` for the backend locally). Tests must
be hermetic.
"""
from types import SimpleNamespace

import pytest

from lucidicai.api.client import HttpClient
from lucidicai.api.resources.mock_call import MockCallResource
from lucidicai.core.config import NetworkConfig, SDKConfig
from lucidicai.sdk.context import current_session_id
from lucidicai.sdk.tools.context import current_mock_context


_STUB_BASE_URL = "https://stub.lucidic.test"


@pytest.fixture(autouse=True)
def _isolate_context_vars():
    """Clear cross-cutting contextvars between every test in tests/sdk/tools.

    Without this, a test that binds ``current_session_id`` (via a
    real session creation) leaks the value into the next test —
    creating order-dependent failures that are hard to track.
    """
    current_session_id.set(None)
    current_mock_context.set(None)
    yield
    current_session_id.set(None)
    current_mock_context.set(None)


@pytest.fixture
def http_client() -> HttpClient:
    """Real HttpClient against a stub URL that respx intercepts."""
    network = NetworkConfig(base_url=_STUB_BASE_URL)
    config = SDKConfig(
        api_key="test-key",
        agent_id="00000000-0000-0000-0000-000000000000",
        network=network,
    )
    return HttpClient(config=config)


@pytest.fixture
def mock_call_resource(http_client) -> MockCallResource:
    return MockCallResource(http=http_client, production=False)


@pytest.fixture
def stub_client(mock_call_resource):
    """Stand-in for ``LucidicAI`` carrying just ``_resources["mock_calls"]``.

    Avoids the real ``LucidicAI.__init__`` (which does API-key
    verification, telemetry setup, etc.) — none of which transport
    touches. SimpleNamespace gives us a dot-accessible attribute bag.
    """
    return SimpleNamespace(_resources={"mock_calls": mock_call_resource})
