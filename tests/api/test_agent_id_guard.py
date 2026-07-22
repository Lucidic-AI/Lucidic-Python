"""LUC-926 — agent_id optional at construction + AgentIdRequiredError guard.

agent_id is no longer required to build a client (org-/id-scoped work runs
without one); agent-scoped operations raise a clear, typed error — even in
production — when it's actually needed but absent.
"""
import pytest

from lucidicai.api.resources.evaluators import EvaluatorsResource
from lucidicai.api.resources.evosim import EvoSimResource
from lucidicai.api.resources.experiment import ExperimentResource
from lucidicai.api.resources.prompt import PromptResource
from lucidicai.api.resources.session import SessionResource
from lucidicai.core.config import NetworkConfig, SDKConfig
from lucidicai.core.errors import AgentIdRequiredError, require_agent_id
from lucidicai.sdk.session import _build_session_params

_BASE = "https://stub.lucidic.test"


class _FakeClient:
    """Minimal stand-in: the write guard fires before any real client use."""
    is_valid = True


class TestValidateRelaxed:
    def test_no_agent_id_is_valid(self):
        # api_key alone is a valid config now (LUC-926).
        assert SDKConfig(api_key="k", agent_id=None).validate() == []

    def test_api_key_still_required(self):
        errors = SDKConfig(api_key=None, agent_id=None).validate()
        assert any("API key" in e for e in errors)


class TestHelper:
    def test_returns_when_present(self):
        assert require_agent_id("a1", "op") == "a1"

    def test_raises_when_absent(self):
        with pytest.raises(AgentIdRequiredError) as ei:
            require_agent_id(None, "create_session")
        assert "create_session" in str(ei.value)
        assert "agent_id" in str(ei.value)


class TestIngestionGuard:
    def test_build_session_params_requires_agent(self):
        # Telemetry chokepoint: a session must attach to an agent. This helper
        # has no production branch, so it raises unconditionally.
        with pytest.raises(AgentIdRequiredError):
            _build_session_params(None, "run", None, None, None, None, None, None, False)

    def test_build_session_params_ok_with_agent(self):
        real_id, params = _build_session_params(
            None, "run", "a1", None, None, None, None, None, False)
        assert params["agent_id"] == "a1"

    def test_evosim_train_raises_even_in_production(self, http):
        # Guard is before the try/except swallow, so production doesn't eat it.
        res = EvoSimResource(http, agent_id=None, production=True)
        with pytest.raises(AgentIdRequiredError):
            res.train("does-not-matter.json")  # raises before reading the file


class TestListReadGuards:
    def test_experiments_list_raises(self, http):
        with pytest.raises(AgentIdRequiredError):
            ExperimentResource(http, agent_id=None).list()

    def test_evaluators_list_raises(self, http):
        with pytest.raises(AgentIdRequiredError):
            EvaluatorsResource(http, agent_id=None).list()

    def test_prompts_list_raises(self, http):
        cfg = SDKConfig(api_key="k", agent_id=None, network=NetworkConfig(base_url=_BASE))
        with pytest.raises(AgentIdRequiredError):
            PromptResource(http, cfg).list()

    def test_prompts_labels_raises(self, http):
        cfg = SDKConfig(api_key="k", agent_id=None, network=NetworkConfig(base_url=_BASE))
        with pytest.raises(AgentIdRequiredError):
            PromptResource(http, cfg).labels()

    def test_explicit_agent_id_bypasses_guard(self, http):
        # Passing agent_id explicitly is fine even with no configured agent —
        # the resolver never reaches the guard (no HTTP asserted; just no raise).
        import respx
        import httpx
        with respx.mock:
            respx.get(f"{_BASE}/sdk/v2/experiments").mock(
                return_value=httpx.Response(200, json={"results": [], "next": None}))
            assert list(ExperimentResource(http, agent_id=None).list("a2")) == []


class TestWriteGuards:
    """The write chokepoints (the live public telemetry / provisioning paths)
    must raise BEFORE their production error-swallow, not silently no-op."""

    def test_session_create_raises_even_in_production(self, http):
        cfg = SDKConfig(api_key="k", agent_id=None, network=NetworkConfig(base_url=_BASE))
        res = SessionResource(http, client=_FakeClient(), config=cfg, production=True)
        with pytest.raises(AgentIdRequiredError):
            res.create(session_name="x")

    def test_experiment_create_raises_even_in_production(self, http):
        res = ExperimentResource(http, agent_id=None, production=True)
        with pytest.raises(AgentIdRequiredError):
            res.create("my-exp")
