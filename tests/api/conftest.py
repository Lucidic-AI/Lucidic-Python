"""Shared fixtures for transport-layer tests (tests/api/).

Build a hermetic ``HttpClient`` against a stub URL with configs constructed
directly (never ``from_env`` — that honors ``LUCIDIC_DEBUG`` and would
silently redirect to localhost). ``backoff_factor`` is tiny and retry sleeps
are patched out in the retry tests, so nothing here actually waits.
"""
import pytest

from lucidicai.api.client import HttpClient
from lucidicai.core.config import NetworkConfig, SDKConfig

BASE_URL = "https://stub.lucidic.test"


@pytest.fixture
def base_url() -> str:
    return BASE_URL


@pytest.fixture
def http() -> HttpClient:
    config = SDKConfig(
        api_key="test-key",
        agent_id="00000000-0000-0000-0000-000000000000",
        network=NetworkConfig(base_url=BASE_URL, max_retries=3, backoff_factor=0.01),
    )
    return HttpClient(config=config)
