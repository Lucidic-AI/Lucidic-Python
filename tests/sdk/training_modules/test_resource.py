"""LUC-801: ``client.training_modules`` per-edit-site tool discovery.

Exercises the full high-level -> HTTP path with respx mocking the backend at the
request boundary, so we verify the real query-param shape (session vs checkpoint,
prompt_name scoping), the edit_site passthrough, and provider-spec scoping.
"""
import httpx
import pytest
import respx

from lucidicai.api.client import HttpClient
from lucidicai.core.config import NetworkConfig, SDKConfig
from lucidicai.core.errors import LucidicError
from lucidicai.sdk.context import current_session_id
from lucidicai.sdk.training_modules.resource import TrainingModulesResource

_BASE_URL = "https://stub.lucidic.test"
_ENDPOINT = f"{_BASE_URL}/sdk/training-modules/tools"
_BODY = {
    "checkpoint_id": "cp-1",
    "tools": [
        {
            "module_key": "style_tuner",
            "tool_name": "get_style_guide",
            "tool_description": "style",
            "input_schema": {"type": "object"},
            "edit_site": "Prompt 1",
        },
    ],
}


class _FakeClient:
    """TrainingModulesResource only touches ``client._http``."""

    def __init__(self, http):
        self._http = http


@pytest.fixture
def resource() -> TrainingModulesResource:
    config = SDKConfig(
        api_key="test-key",
        agent_id="00000000-0000-0000-0000-000000000000",
        network=NetworkConfig(base_url=_BASE_URL),
    )
    return TrainingModulesResource(client=_FakeClient(HttpClient(config=config)))


@pytest.fixture
def in_session():
    token = current_session_id.set("sess-ctx")
    try:
        yield "sess-ctx"
    finally:
        current_session_id.reset(token)


@respx.mock
def test_tools_uses_active_session_and_passes_edit_site_through(resource, in_session):
    route = respx.get(_ENDPOINT).mock(return_value=httpx.Response(200, json=_BODY))
    tools = resource.tools()
    assert dict(route.calls.last.request.url.params) == {"session_id": "sess-ctx"}
    assert tools[0]["edit_site"] == "Prompt 1"


@respx.mock
def test_tools_prompt_name_scopes_server_side(resource, in_session):
    route = respx.get(_ENDPOINT).mock(return_value=httpx.Response(200, json=_BODY))
    resource.tools(prompt_name="Prompt 1")
    assert dict(route.calls.last.request.url.params) == {
        "session_id": "sess-ctx",
        "prompt_name": "Prompt 1",
    }


@respx.mock
def test_tools_checkpoint_id_needs_no_session(resource):
    # no active session bound; checkpoint_id path must not require one.
    route = respx.get(_ENDPOINT).mock(return_value=httpx.Response(200, json=_BODY))
    resource.tools(checkpoint_id="cp-1", prompt_name="Prompt 2")
    params = dict(route.calls.last.request.url.params)
    assert params == {"checkpoint_id": "cp-1", "prompt_name": "Prompt 2"}
    assert "session_id" not in params


def test_tools_without_session_or_checkpoint_raises(resource):
    with pytest.raises(LucidicError):
        resource.tools()


@respx.mock
def test_openai_tools_scopes_and_shapes(resource, in_session):
    route = respx.get(_ENDPOINT).mock(return_value=httpx.Response(200, json=_BODY))
    specs = resource.openai_tools(prompt_name="Prompt 1")
    assert dict(route.calls.last.request.url.params) == {
        "session_id": "sess-ctx",
        "prompt_name": "Prompt 1",
    }
    assert specs == [{
        "type": "function",
        "function": {
            "name": "get_style_guide",
            "description": "style",
            "parameters": {"type": "object"},
        },
    }]


@respx.mock
def test_anthropic_tools_checkpoint_scope(resource):
    route = respx.get(_ENDPOINT).mock(return_value=httpx.Response(200, json=_BODY))
    specs = resource.anthropic_tools(checkpoint_id="cp-1")
    assert dict(route.calls.last.request.url.params) == {"checkpoint_id": "cp-1"}
    assert specs == [{
        "name": "get_style_guide",
        "description": "style",
        "input_schema": {"type": "object"},
    }]


@respx.mock
@pytest.mark.asyncio
async def test_atools_threads_prompt_name(resource, in_session):
    route = respx.get(_ENDPOINT).mock(return_value=httpx.Response(200, json=_BODY))
    tools = await resource.atools(prompt_name="Prompt 1")
    assert dict(route.calls.last.request.url.params) == {
        "session_id": "sess-ctx",
        "prompt_name": "Prompt 1",
    }
    assert tools[0]["edit_site"] == "Prompt 1"
