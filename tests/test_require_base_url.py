"""require_base_url / LUCIDIC_REQUIRE_BASE_URL: never fall back to Lucidic SaaS.

In a self-hosted (BYOC) deployment an agent with no base URL would otherwise send
prompts and outputs to https://backend.lucidic.ai/api. With the flag on, a missing
base URL or any region raises from ``NetworkConfig.from_env``, which the client,
``get_config()`` and the legacy modules all go through. The raise must survive
production mode and error suppression, before any ``HttpClient`` is built. Flag
off, nothing changes. No test here touches the network: the one request is
mocked with respx.
"""
import logging

import httpx
import pytest
import respx

import lucidicai.client as client_module
from lucidicai import LucidicAI
from lucidicai.api.client import HttpClient
from lucidicai.core.config import (
    DEBUG_URL,
    REGION_URLS,
    NetworkConfig,
    Region,
    SDKConfig,
    get_config,
    reset_config,
)
from lucidicai.sdk.shutdown_manager import ShutdownManager

_BYOC_URL = "https://lucidic.byoc.test/api"
_SAAS_US = "https://backend.lucidic.ai/api"
_SAAS_INDIA = "https://in.backend.lucidic.ai/api"
_ENV = (
    "LUCIDIC_REQUIRE_BASE_URL", "LUCIDIC_BASE_URL", "LUCIDIC_REGION", "LUCIDIC_DEBUG",
    "LUCIDIC_PRODUCTION", "LUCIDIC_SUPPRESS_ERRORS", "LUCIDIC_API_KEY", "LUCIDIC_AGENT_ID",
)
_NO_URL = "no base URL was given"


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    # SDKConfig.from_env loads a .env from the cwd; keep a developer's out of these tests.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    reset_config()
    yield
    reset_config()


@pytest.fixture
def built(monkeypatch):
    """Every HttpClient that LucidicAI builds (still the real class)."""
    calls = []

    class _Recording(HttpClient):
        def __init__(self, config=None):
            calls.append(config)
            super().__init__(config)

    monkeypatch.setattr(client_module, "HttpClient", _Recording)
    return calls


@pytest.fixture
def flag(monkeypatch):
    monkeypatch.setenv("LUCIDIC_REQUIRE_BASE_URL", "true")


@pytest.fixture
def production(monkeypatch):
    monkeypatch.setenv("LUCIDIC_PRODUCTION", "true")
    monkeypatch.setenv("LUCIDIC_SUPPRESS_ERRORS", "true")


def _client(**kwargs):
    kwargs.setdefault("api_key", "test-key")
    return LucidicAI(**kwargs)


def _registered_clients():
    return len(ShutdownManager()._clients)


class TestFlagSetNoURL:
    """Ticket case 1: flag set and no URL, init raises (production too)."""

    def test_env_flag_raises(self, flag, built):
        before = _registered_clients()
        with pytest.raises(ValueError, match=_NO_URL):
            _client()
        assert built == []
        assert _registered_clients() == before

    def test_raises_with_production_env_and_suppression(self, flag, production, built):
        before = _registered_clients()
        with pytest.raises(ValueError, match=_NO_URL):
            _client()
        assert built == []
        assert _registered_clients() == before

    def test_raises_with_production_arg(self, flag, built):
        with pytest.raises(ValueError, match=_NO_URL):
            _client(production=True)
        assert built == []

    def test_kwarg_alone_raises(self, built):
        with pytest.raises(ValueError, match=_NO_URL):
            _client(require_base_url=True)
        assert built == []

    def test_kwarg_alone_raises_in_production(self, production, built):
        with pytest.raises(ValueError, match=_NO_URL):
            _client(require_base_url=True, production=True)
        assert built == []

    def test_kwarg_false_does_not_turn_env_flag_off(self, flag, built):
        with pytest.raises(ValueError, match=_NO_URL):
            _client(require_base_url=False)
        assert built == []

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_url_counts_as_missing(self, flag, monkeypatch, built, blank):
        monkeypatch.setenv("LUCIDIC_BASE_URL", blank)
        with pytest.raises(ValueError, match=_NO_URL):
            _client(base_url=blank or None)
        assert built == []

    def test_message_names_the_fix(self, flag):
        with pytest.raises(ValueError) as ei:
            _client()
        msg = str(ei.value)
        assert "LUCIDIC_REQUIRE_BASE_URL" in msg
        assert "LUCIDIC_BASE_URL" in msg
        assert "will not fall back to Lucidic SaaS" in msg

    def test_legacy_get_config_raises(self, flag):
        # The legacy module-level paths (sdk/event.py, sdk/session.py) read config here.
        with pytest.raises(ValueError, match=_NO_URL):
            get_config()

    def test_legacy_http_client_without_config_raises(self, flag):
        # HttpClient() with no config reads get_config(), so it can't default to SaaS.
        with pytest.raises(ValueError, match=_NO_URL):
            HttpClient()


class TestFlagSetWithURL:
    """Ticket case 2: flag set and a URL given, the SDK uses that URL."""

    @respx.mock
    def test_env_url_is_used(self, flag, monkeypatch, built):
        monkeypatch.setenv("LUCIDIC_BASE_URL", _BYOC_URL)
        route = respx.get(f"{_BYOC_URL}/ping").mock(return_value=httpx.Response(200, json={}))
        client = _client()
        try:
            assert client.config.network.base_url == _BYOC_URL
            assert client.config.network.region is None
            assert len(built) == 1 and built[0] is client.config
            # A real request goes to the BYOC URL (anything else would be unmocked and fail).
            client._http.get("ping")
            assert route.called
        finally:
            client.close()

    def test_env_url_is_used_in_production(self, flag, production, monkeypatch, built):
        monkeypatch.setenv("LUCIDIC_BASE_URL", _BYOC_URL)
        client = _client()
        try:
            assert client.config.network.base_url == _BYOC_URL
            assert len(built) == 1
        finally:
            client.close()

    def test_kwarg_and_base_url_arg(self, built):
        client = _client(require_base_url=True, base_url=_BYOC_URL)
        try:
            assert client.config.network.base_url == _BYOC_URL
            assert client._http.base_url == _BYOC_URL
        finally:
            client.close()

    def test_env_flag_and_base_url_arg(self, flag, built):
        client = _client(base_url=_BYOC_URL)
        try:
            assert client.config.network.base_url == _BYOC_URL
        finally:
            client.close()

    def test_legacy_get_config_uses_env_url(self, flag, monkeypatch):
        monkeypatch.setenv("LUCIDIC_BASE_URL", _BYOC_URL)
        assert get_config().network.base_url == _BYOC_URL


class TestRegionRejected:
    """region only ever picks a SaaS URL, so the flag rejects it, base URL or not."""

    def test_region_arg(self, flag, built):
        with pytest.raises(ValueError, match="region 'india' is not allowed"):
            _client(region="india")
        assert built == []

    def test_region_env(self, flag, monkeypatch, built):
        monkeypatch.setenv("LUCIDIC_REGION", "us")
        with pytest.raises(ValueError, match="region 'us' is not allowed"):
            _client()
        assert built == []

    def test_region_with_base_url(self, flag, monkeypatch, built):
        monkeypatch.setenv("LUCIDIC_BASE_URL", _BYOC_URL)
        with pytest.raises(ValueError, match="region 'us' is not allowed"):
            _client(region="us")
        assert built == []

    def test_region_with_kwarg_in_production(self, production, built):
        with pytest.raises(ValueError, match="region 'us' is not allowed"):
            _client(require_base_url=True, base_url=_BYOC_URL, region="us", production=True)
        assert built == []


class TestFlagUnset:
    """Ticket case 3: flag unset, behaviour is unchanged."""

    def test_default_is_saas_us(self):
        net = NetworkConfig.from_env()
        assert net.base_url == _SAAS_US
        assert net.region is Region.US

    def test_region_arg(self):
        net = NetworkConfig.from_env(region="india")
        assert net.base_url == _SAAS_INDIA
        assert net.region is Region.INDIA

    def test_region_env(self, monkeypatch):
        monkeypatch.setenv("LUCIDIC_REGION", "india")
        assert NetworkConfig.from_env().base_url == _SAAS_INDIA

    def test_invalid_region_still_raises(self):
        with pytest.raises(ValueError, match="Invalid region"):
            NetworkConfig.from_env(region="mars")

    def test_env_url(self, monkeypatch):
        monkeypatch.setenv("LUCIDIC_BASE_URL", _BYOC_URL)
        assert NetworkConfig.from_env().base_url == _BYOC_URL

    def test_base_url_beats_region_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="Lucidic"):
            net = NetworkConfig.from_env(region="india", base_url=_BYOC_URL)
        assert net.base_url == _BYOC_URL
        assert net.region is None
        assert "ignoring region 'india'" in caplog.text

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "  "])
    def test_falsy_env_values(self, monkeypatch, value):
        monkeypatch.setenv("LUCIDIC_REQUIRE_BASE_URL", value)
        assert NetworkConfig.from_env().base_url == _SAAS_US

    def test_client_defaults_to_saas(self, built):
        client = _client()
        try:
            assert client.config.network.base_url == _SAAS_US
            assert client.config.network.region is Region.US
            assert len(built) == 1
        finally:
            client.close()

    def test_client_region(self, built):
        client = _client(region="india")
        try:
            assert client.config.network.base_url == REGION_URLS[Region.INDIA]
        finally:
            client.close()


class TestDebugStillWins:
    def test_debug_uses_localhost_without_raising(self, flag, monkeypatch):
        monkeypatch.setenv("LUCIDIC_DEBUG", "true")
        config = SDKConfig.from_env()
        assert config.network.base_url == DEBUG_URL


class TestEnvParsing:
    @pytest.mark.parametrize("value", ["true", "TRUE", "True", "1", "yes", "on", " true "])
    def test_truthy_values_turn_it_on(self, monkeypatch, value):
        monkeypatch.setenv("LUCIDIC_REQUIRE_BASE_URL", value)
        with pytest.raises(ValueError, match=_NO_URL):
            NetworkConfig.from_env()

    @pytest.mark.parametrize("value", ["ture", "enabled", "2"])
    def test_unrecognized_value_raises_instead_of_reading_as_off(self, monkeypatch, value):
        monkeypatch.setenv("LUCIDIC_REQUIRE_BASE_URL", value)
        with pytest.raises(ValueError, match="must be true or false"):
            NetworkConfig.from_env()
