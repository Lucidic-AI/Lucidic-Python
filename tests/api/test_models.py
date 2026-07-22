"""LUC-905 — typed response-model base."""
from dataclasses import dataclass
from typing import List, Optional

import pytest

from lucidicai.api.models.base import APIModel


@dataclass
class _Agent(APIModel):
    agent_id: str
    name: Optional[str] = None
    tags: Optional[List[str]] = None


class TestFromDict:
    def test_maps_known_fields(self):
        a = _Agent.from_dict({"agent_id": "a1", "name": "bot", "tags": ["x"]})
        assert a.agent_id == "a1"
        assert a.name == "bot"
        assert a.tags == ["x"]

    def test_unknown_fields_go_to_extra(self):
        a = _Agent.from_dict({"agent_id": "a1", "created_at": "2026", "future_flag": True})
        assert a.extra == {"created_at": "2026", "future_flag": True}
        # ...and don't blow up construction (forward-compat).
        assert a.agent_id == "a1"

    def test_missing_optional_uses_default(self):
        a = _Agent.from_dict({"agent_id": "a1"})
        assert a.name is None and a.tags is None

    def test_missing_required_raises(self):
        with pytest.raises(TypeError):
            _Agent.from_dict({"name": "no-id"})

    def test_non_dict_raises_typeerror(self):
        with pytest.raises(TypeError):
            _Agent.from_dict(["not", "a", "dict"])

    def test_extra_defaults_empty(self):
        a = _Agent.from_dict({"agent_id": "a1"})
        assert a.extra == {}


class TestFromList:
    def test_builds_list(self):
        items = _Agent.from_list([{"agent_id": "1"}, {"agent_id": "2", "name": "b"}])
        assert [i.agent_id for i in items] == ["1", "2"]
        assert items[1].name == "b"


class TestToDict:
    def test_excludes_extra(self):
        a = _Agent.from_dict({"agent_id": "a1", "name": "bot", "unknown": 9})
        d = a.to_dict()
        assert d == {"agent_id": "a1", "name": "bot", "tags": None}
        assert "unknown" not in d
