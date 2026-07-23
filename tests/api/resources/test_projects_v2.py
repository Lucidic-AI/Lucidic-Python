"""LUC-913 — client.projects CRUD (list / create / get / update / delete)."""
import json

import httpx
import pytest
import respx

from lucidicai.api.models.project import Project
from lucidicai.api.resources.projects import ProjectsResource
from lucidicai.core.errors import NotFoundError, ValidationError

_BASE = "https://stub.lucidic.test"
_PROJECTS = f"{_BASE}/sdk/v2/projects"


@pytest.fixture
def projects(http):
    return ProjectsResource(http)


def _project(i, **over):
    d = {"project_id": f"p{i}", "name": f"proj-{i}", "description": "", "icon": "folder"}
    d.update(over)
    return d


class TestList:
    @respx.mock
    def test_follows_pages_typed(self, projects):
        respx.get(_PROJECTS).mock(side_effect=[
            httpx.Response(200, json={"results": [_project(1), _project(2)],
                                      "next": f"{_PROJECTS}?cursor=c2", "previous": None}),
            httpx.Response(200, json={"results": [_project(3)], "next": None}),
        ])
        got = list(projects.list())
        assert [p.project_id for p in got] == ["p1", "p2", "p3"]
        assert all(isinstance(p, Project) for p in got)

    @respx.mock
    def test_no_agent_id_param(self, projects):
        # Projects are org-scoped: no agent_id is sent.
        route = respx.get(_PROJECTS).mock(
            return_value=httpx.Response(200, json={"results": [], "next": None}))
        list(projects.list(ordering="id"))
        p = route.calls.last.request.url.params
        assert "agent_id" not in p and p["ordering"] == "id"


class TestGet:
    @respx.mock
    def test_get(self, projects):
        respx.get(f"{_PROJECTS}/p1").mock(return_value=httpx.Response(200, json=_project(1)))
        p = projects.get("p1")
        assert isinstance(p, Project) and p.project_id == "p1" and p.icon == "folder"

    @respx.mock
    def test_404(self, projects):
        respx.get(f"{_PROJECTS}/missing").mock(
            return_value=httpx.Response(404, json={"error": "not found"}))
        with pytest.raises(NotFoundError):
            projects.get("missing")


class TestCreate:
    @respx.mock
    def test_create_requires_name_and_icon(self, projects):
        route = respx.post(_PROJECTS).mock(return_value=httpx.Response(201, json=_project(1)))
        p = projects.create("proj-1", "rocket")
        assert isinstance(p, Project) and p.project_id == "p1"
        body = json.loads(route.calls.last.request.read())
        assert body["name"] == "proj-1" and body["icon"] == "rocket"
        assert "description" not in body

    @respx.mock
    def test_create_with_description(self, projects):
        route = respx.post(_PROJECTS).mock(return_value=httpx.Response(201, json=_project(1)))
        projects.create("proj-1", "rocket", description="my project")
        body = json.loads(route.calls.last.request.read())
        assert body["description"] == "my project"

    @respx.mock
    def test_create_over_length_400(self, projects):
        respx.post(_PROJECTS).mock(return_value=httpx.Response(
            400, json={"error": "A field exceeds its maximum length."}))
        with pytest.raises(ValidationError):
            projects.create("x" * 999, "i")


class TestUpdate:
    @respx.mock
    def test_only_provided_fields(self, projects):
        route = respx.put(f"{_PROJECTS}/p1").mock(return_value=httpx.Response(200, json=_project(1)))
        p = projects.update("p1", name="renamed")
        assert isinstance(p, Project)
        body = json.loads(route.calls.last.request.read())
        assert body["name"] == "renamed"
        assert "icon" not in body and "description" not in body

    @respx.mock
    def test_all_fields(self, projects):
        route = respx.put(f"{_PROJECTS}/p1").mock(return_value=httpx.Response(200, json=_project(1)))
        projects.update("p1", name="n", description="d", icon="i")
        body = json.loads(route.calls.last.request.read())
        assert body == {"name": "n", "description": "d", "icon": "i", **_current_time(body)}


class TestDelete:
    @respx.mock
    def test_delete_returns_none_on_204(self, projects):
        route = respx.delete(f"{_PROJECTS}/p1").mock(return_value=httpx.Response(204))
        assert projects.delete("p1") is None
        assert route.called

    @respx.mock
    def test_delete_404_raises(self, projects):
        respx.delete(f"{_PROJECTS}/missing").mock(
            return_value=httpx.Response(404, json={"error": "not found"}))
        with pytest.raises(NotFoundError):
            projects.delete("missing")


class TestAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_acreate(self, projects):
        respx.post(_PROJECTS).mock(return_value=httpx.Response(201, json=_project(1)))
        p = await projects.acreate("proj-1", "rocket")
        assert p.project_id == "p1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_aupdate(self, projects):
        respx.put(f"{_PROJECTS}/p1").mock(return_value=httpx.Response(200, json=_project(1)))
        p = await projects.aupdate("p1", icon="star")
        assert p.project_id == "p1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_adelete(self, projects):
        route = respx.delete(f"{_PROJECTS}/p1").mock(return_value=httpx.Response(204))
        assert await projects.adelete("p1") is None
        assert route.called


def _current_time(body):
    # PUT/POST bodies carry an auto-injected current_time; ignore it in equality.
    return {"current_time": body["current_time"]} if "current_time" in body else {}
