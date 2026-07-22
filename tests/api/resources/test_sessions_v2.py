"""LUC-907 — client.sessions v2 reads (list/count/tags, detail+trace,
evaluator-results, event raw)."""
import httpx
import pytest
import respx

from lucidicai.api.models.session import (
    EvalResult,
    Event,
    EventEval,
    Session,
    SessionEvaluatorResults,
    SessionTrace,
)
from lucidicai.api.resources.session import SessionResource
from lucidicai.core.errors import NotFoundError

_BASE = "https://stub.lucidic.test"
_SESSIONS = f"{_BASE}/sdk/v2/sessions"


@pytest.fixture
def sessions(http):
    # v2 reads only use self.http; client/config/production are irrelevant here.
    return SessionResource(http, client=None, config=None, production=False)


def _session(i, **over):
    d = {
        "session_id": f"s{i}", "custom_session_id": None, "name": f"run-{i}",
        "start_time": "2026-07-22T00:00:00Z", "duration": 1.5, "is_finished": True,
        "production_monitoring": False, "task": None, "cost": 0.01, "tags": ["prod"],
        "eval_previews": [], "num_events": 3, "status": "finished", "datasetitem_id": None,
    }
    d.update(over)
    return d


class TestList:
    @respx.mock
    def test_follows_pages_typed(self, sessions):
        respx.get(_SESSIONS).mock(side_effect=[
            httpx.Response(200, json={"results": [_session(1), _session(2)],
                                      "next": f"{_SESSIONS}?cursor=c2", "previous": None}),
            httpx.Response(200, json={"results": [_session(3)], "next": None}),
        ])
        got = list(sessions.list("a1"))
        assert [s.session_id for s in got] == ["s1", "s2", "s3"]
        assert all(isinstance(s, Session) for s in got)

    @respx.mock
    def test_filters_are_normalized(self, sessions):
        route = respx.get(_SESSIONS).mock(
            return_value=httpx.Response(200, json={"results": [], "next": None}))
        list(sessions.list(
            "a1", experiment_id="e1", production=True, cost="0:10", duration=":5",
            num_events="1:", tags=["prod", "canary"], status="finished",
            eval_bool="quality:true", ordering="-start_time", page_size=25,
        ))
        p = route.calls.last.request.url.params
        assert p["agent_id"] == "a1"
        assert p["experiment_id"] == "e1"
        assert p["production"] == "true"       # bool -> "true"
        assert p["cost"] == "0:10"
        assert p["duration"] == ":5"
        assert p["num_events"] == "1:"
        assert p["tags"] == "prod,canary"      # list -> comma
        assert p["status"] == "finished"
        assert p["eval_bool"] == "quality:true"
        assert p["ordering"] == "-start_time"
        assert p["page_size"] == "25"

    @respx.mock
    def test_production_false_serialized(self, sessions):
        route = respx.get(_SESSIONS).mock(
            return_value=httpx.Response(200, json={"results": [], "next": None}))
        list(sessions.list("a1", production=False))
        assert route.calls.last.request.url.params["production"] == "false"

    @respx.mock
    def test_list_page(self, sessions):
        respx.get(_SESSIONS).mock(return_value=httpx.Response(
            200, json={"results": [_session(1)], "next": f"{_SESSIONS}?cursor=c9"}))
        page = sessions.list_page("a1")
        assert isinstance(page.results[0], Session) and page.next_cursor == "c9"


class TestCountTags:
    @respx.mock
    def test_count(self, sessions):
        respx.head(_SESSIONS).mock(return_value=httpx.Response(
            200, headers={"X-Total-Count": "42"}))
        assert sessions.count("a1", production=True) == 42

    @respx.mock
    def test_tags(self, sessions):
        respx.head(_SESSIONS).mock(return_value=httpx.Response(
            200, headers={"X-Tags": "prod,canary"}))
        assert sessions.tags("a1") == ["prod", "canary"]

    @respx.mock
    def test_tags_empty(self, sessions):
        respx.head(_SESSIONS).mock(return_value=httpx.Response(200, headers={"X-Tags": ""}))
        assert sessions.tags("a1") == []


class TestGetTrace:
    @respx.mock
    def test_detail_and_tree(self, sessions):
        body = {
            "session": _session(1),
            "events": [
                {"event_id": "e1", "parent_event_id": None, "occurred_at": "1", "type": "llm"},
                {"event_id": "e2", "parent_event_id": "e1", "occurred_at": "2", "type": "tool"},
                {"event_id": "e3", "parent_event_id": "e1", "occurred_at": "3", "type": "tool"},
            ],
            "num_events": 3,
        }
        respx.get(f"{_SESSIONS}/s1").mock(return_value=httpx.Response(200, json=body))
        trace = sessions.get("s1")
        assert isinstance(trace, SessionTrace)
        assert isinstance(trace.session, Session) and trace.session.session_id == "s1"
        assert trace.num_events == 3
        assert all(isinstance(e, Event) for e in trace.events)
        roots = trace.tree()
        assert [r.event_id for r in roots] == ["e1"]
        assert [c.event_id for c in roots[0].children] == ["e2", "e3"]

    @respx.mock
    def test_accepts_custom_session_id(self, sessions):
        respx.get(f"{_SESSIONS}/my-custom-id").mock(return_value=httpx.Response(
            200, json={"session": _session(1), "events": [], "num_events": 0}))
        trace = sessions.get("my-custom-id")
        assert trace.events == []

    @respx.mock
    def test_404_raises(self, sessions):
        respx.get(f"{_SESSIONS}/missing").mock(return_value=httpx.Response(
            404, json={"error": "not found"}))
        with pytest.raises(NotFoundError):
            sessions.get("missing")

    @respx.mock
    def test_orphan_event_surfaces_as_root(self, sessions):
        # e2's parent e1 is absent from the returned set (purged/truncated) —
        # it must surface as a root, not silently vanish from tree().
        respx.get(f"{_SESSIONS}/s1").mock(return_value=httpx.Response(200, json={
            "session": _session(1),
            "events": [{"event_id": "e2", "parent_event_id": "e1", "occurred_at": "1"}],
            "num_events": 1,
        }))
        roots = sessions.get("s1").tree()
        assert [r.event_id for r in roots] == ["e2"]

    @respx.mock
    def test_self_cycle_event_is_root(self, sessions):
        respx.get(f"{_SESSIONS}/s1").mock(return_value=httpx.Response(200, json={
            "session": _session(1),
            "events": [{"event_id": "e1", "parent_event_id": "e1", "occurred_at": "1"}],
        }))
        roots = sessions.get("s1").tree()
        assert [r.event_id for r in roots] == ["e1"]
        assert roots[0].children == []  # not its own child

    @respx.mock
    def test_custom_session_id_path_encoded(self, sessions):
        # A client-supplied custom_session_id with URL-significant chars must be
        # percent-encoded so it doesn't break routing (false NotFound otherwise).
        route = respx.route(method="GET").mock(return_value=httpx.Response(
            200, json={"session": _session(1), "events": [], "num_events": 0}))
        sessions.get("a/b c")
        raw = route.calls.last.request.url.raw_path.decode()
        assert "/sessions/a/b" not in raw   # the raw '/' did not leak
        assert "a%2Fb" in raw


class TestFilterValidation:
    def test_unknown_filter_raises(self, sessions):
        # A typo in a **filters method must fail loudly, not silently return the
        # unfiltered count/tags.
        with pytest.raises(TypeError):
            sessions.count("a1", experiment="e1")  # should be experiment_id


class TestEvaluatorResults:
    @respx.mock
    def test_typed_results(self, sessions):
        body = {
            "evals": [{"eval_id": "ev1", "evaluator_name": "quality", "result": 0.9,
                       "result_type": "number", "is_pending": False}],
            "event_evals": [{"eval_id": "ee1", "event_id": "e1", "criteria_name": "grounded",
                             "result": True}],
        }
        respx.get(f"{_SESSIONS}/s1/evaluator-results").mock(
            return_value=httpx.Response(200, json=body))
        res = sessions.evaluator_results("s1")
        assert isinstance(res, SessionEvaluatorResults)
        assert isinstance(res.evals[0], EvalResult) and res.evals[0].evaluator_name == "quality"
        assert isinstance(res.event_evals[0], EventEval) and res.event_evals[0].event_id == "e1"

    @respx.mock
    def test_empty_results(self, sessions):
        respx.get(f"{_SESSIONS}/s1/evaluator-results").mock(return_value=httpx.Response(
            200, json={"evals": [], "event_evals": []}))
        res = sessions.evaluator_results("s1")
        assert res.evals == [] and res.event_evals == []


class TestEvent:
    @respx.mock
    def test_non_raw_returns_typed_event(self, sessions):
        respx.get(f"{_SESSIONS}/s1/events/e1").mock(return_value=httpx.Response(
            200, json={"event_id": "e1", "type": "llm", "has_blob": True,
                       "payload": {"preview": "..."}}))
        ev = sessions.event("s1", "e1")
        assert isinstance(ev, Event) and ev.event_id == "e1" and ev.has_blob is True

    @respx.mock
    def test_raw_true_sends_param_and_returns_blob_url(self, sessions):
        route = respx.get(f"{_SESSIONS}/s1/events/e1").mock(return_value=httpx.Response(
            200, json={"event_id": "e1", "type": "llm", "has_blob": True,
                       "blob_url": "https://s3/presigned?sig=x"}))
        ev = sessions.event("s1", "e1", raw=True)
        assert route.calls.last.request.url.params["raw"] == "true"
        assert ev.blob_url == "https://s3/presigned?sig=x"

    @respx.mock
    def test_raw_true_inline_payload(self, sessions):
        respx.get(f"{_SESSIONS}/s1/events/e1").mock(return_value=httpx.Response(
            200, json={"event_id": "e1", "payload": {"full": [1, 2, 3]}}))
        ev = sessions.event("s1", "e1", raw=True)
        assert ev.payload == {"full": [1, 2, 3]}


class TestAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_aget(self, sessions):
        respx.get(f"{_SESSIONS}/s1").mock(return_value=httpx.Response(
            200, json={"session": _session(1), "events": [], "num_events": 0}))
        trace = await sessions.aget("s1")
        assert trace.session.session_id == "s1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_alist(self, sessions):
        respx.get(_SESSIONS).mock(side_effect=[
            httpx.Response(200, json={"results": [_session(1)], "next": f"{_SESSIONS}?cursor=c2"}),
            httpx.Response(200, json={"results": [_session(2)], "next": None}),
        ])
        got = [s.session_id async for s in sessions.alist("a1")]
        assert got == ["s1", "s2"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_acount(self, sessions):
        respx.head(_SESSIONS).mock(return_value=httpx.Response(200, headers={"X-Total-Count": "7"}))
        assert await sessions.acount("a1") == 7

    @respx.mock
    @pytest.mark.asyncio
    async def test_aevent(self, sessions):
        respx.get(f"{_SESSIONS}/s1/events/e1").mock(return_value=httpx.Response(
            200, json={"event_id": "e1"}))
        ev = await sessions.aevent("s1", "e1")
        assert ev.event_id == "e1"
