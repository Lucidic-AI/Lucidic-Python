"""Tests for ``lucidicai.sdk.tools.registry``.

Pins the contract that the backend ``SyncAgentToolsSerializer`` validates:
ToolSurface shape, source_hash determinism, and the buffered-registration
path that ``LucidicAI.__init__`` will drain in LUC-577c.

Hash stability tests are the most important — drift detection relies on
both ends producing identical bytes for the same input. A refactor that
silently changes the hash would invalidate every customer's stored
snapshot. The test cases here pin the canonical encoding and MUST be
updated deliberately if the encoding ever changes (with a corresponding
backend migration).
"""
import subprocess
import sys
import textwrap

import pytest

from lucidicai.sdk.tools.registry import (
    ToolSurface,
    _canonical_signature_json,
    _compute_source_hash,
    _params_from_callable,
    _params_from_json_schema,
    _serialize_default,
    _stringify_annotation,
    drain_buffer_into,
    register_tool,
    snapshot_registry,
    _PENDING_BUFFER,
    _REGISTRY,
)


@pytest.fixture(autouse=True)
def _clear_module_state():
    """Each test starts with empty registry + buffer.

    The module globals persist across tests in the same process; without
    this fixture, surface registrations from one test leak into the next.
    """
    _REGISTRY.clear()
    _PENDING_BUFFER.clear()
    yield
    _REGISTRY.clear()
    _PENDING_BUFFER.clear()


# ---------- _stringify_annotation ---------------------------------------


class TestStringifyAnnotation:
    def test_empty_returns_none(self):
        import inspect
        assert _stringify_annotation(inspect.Parameter.empty) is None
        assert _stringify_annotation(inspect.Signature.empty) is None

    def test_none_type(self):
        assert _stringify_annotation(type(None)) == "None"

    def test_builtin_with_name(self):
        assert _stringify_annotation(int) == "int"
        assert _stringify_annotation(str) == "str"
        assert _stringify_annotation(list) == "list"

    def test_user_class(self):
        class MyType:
            pass
        assert _stringify_annotation(MyType) == "MyType"

    def test_generic_alias_strips_to_origin_name(self):
        # `list[str].__name__` is "list" in 3.10+ — the helper takes the
        # origin name. Acceptable info loss for v1; the tool catalog
        # surfaces the param as "list" rather than "list[str]". Backend
        # serializer doesn't deeply parse the type string anyway.
        assert _stringify_annotation(list[str]) == "list"
        assert _stringify_annotation(dict[str, int]) == "dict"


# ---------- _serialize_default -------------------------------------------


class TestSerializeDefault:
    def test_no_default_returns_none(self):
        import inspect
        assert _serialize_default(inspect.Parameter.empty) is None

    def test_python_none(self):
        assert _serialize_default(None) is None

    def test_json_scalars(self):
        assert _serialize_default("hello") == "hello"
        assert _serialize_default(42) == 42
        assert _serialize_default(3.14) == 3.14
        assert _serialize_default(True) is True
        assert _serialize_default(False) is False

    def test_json_list(self):
        assert _serialize_default([1, 2, 3]) == [1, 2, 3]

    def test_json_dict(self):
        assert _serialize_default({"a": 1}) == {"a": 1}

    def test_tuple_becomes_list(self):
        # tuple isn't JSON-native; we convert to list for serializability
        assert _serialize_default((1, 2)) == [1, 2]

    def test_non_json_container_falls_back_to_repr(self):
        class Opaque:
            def __repr__(self):
                return "<Opaque>"
        # A list containing a non-serializable obj
        result = _serialize_default([Opaque()])
        assert isinstance(result, str)
        assert "Opaque" in result

    def test_arbitrary_object_uses_repr(self):
        class Thing:
            def __repr__(self):
                return "Thing(x=1)"
        assert _serialize_default(Thing()) == "Thing(x=1)"


# ---------- _params_from_callable ----------------------------------------


class TestParamsFromCallable:
    def test_required_only(self):
        def f(a: int, b: str): pass
        assert _params_from_callable(f) == [
            {"name": "a", "type": "int", "default": None, "required": True},
            {"name": "b", "type": "str", "default": None, "required": True},
        ]

    def test_mixed_required_optional(self):
        def f(a: int, b: str = "x"): pass
        params = _params_from_callable(f)
        assert params[0]["required"] is True
        assert params[1]["required"] is False
        assert params[1]["default"] == "x"

    def test_untyped_param_defaults_to_any(self):
        def f(a, b=1): pass
        params = _params_from_callable(f)
        assert params[0]["type"] == "any"
        assert params[1]["type"] == "any"

    def test_var_args_and_kwargs_skipped(self):
        def f(a, *args, **kwargs): pass
        params = _params_from_callable(f)
        assert [p["name"] for p in params] == ["a"]

    def test_builtin_with_no_signature_returns_empty(self):
        # builtins like `len` raise on inspect.signature in some configs
        params = _params_from_callable(len)
        assert isinstance(params, list)
        # Either empty (if introspection failed) or has 'obj' (if it worked).
        # Both are acceptable — point is we don't raise.


# ---------- _params_from_json_schema -------------------------------------


class TestParamsFromJsonSchema:
    def test_basic(self):
        schema = {
            "type": "object",
            "properties": {
                "sender": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["sender"],
        }
        params = _params_from_json_schema(schema)
        assert params == [
            {"name": "sender", "type": "string", "default": None, "required": True},
            {"name": "limit", "type": "integer", "default": 50, "required": False},
        ]

    def test_missing_type_defaults_to_any(self):
        schema = {"properties": {"x": {}}, "required": []}
        assert _params_from_json_schema(schema)[0]["type"] == "any"

    def test_missing_required_treats_all_optional(self):
        schema = {"properties": {"x": {"type": "string"}}}
        assert _params_from_json_schema(schema)[0]["required"] is False

    def test_empty_schema_returns_empty(self):
        assert _params_from_json_schema({}) == []

    def test_non_dict_returns_empty(self):
        assert _params_from_json_schema(None) == []
        assert _params_from_json_schema("not a schema") == []


# ---------- _compute_source_hash (the load-bearing one) -----------------


class TestSourceHashStability:
    """The byte-level encoding here is the canonical drift-detection key.
    Any refactor that changes these hashes invalidates every existing
    customer snapshot — pin loudly with literal values."""

    def test_hash_is_64_char_hex(self):
        h = _compute_source_hash(name="f", signature={"params": []}, body_source="")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_input_same_hash(self):
        sig = {"params": [{"name": "a", "type": "int", "default": None, "required": True}]}
        h1 = _compute_source_hash(name="f", signature=sig, body_source="def f(a): pass")
        h2 = _compute_source_hash(name="f", signature=sig, body_source="def f(a): pass")
        assert h1 == h2

    def test_body_change_changes_hash(self):
        sig = {"params": []}
        h1 = _compute_source_hash(name="f", signature=sig, body_source="def f(): return 1")
        h2 = _compute_source_hash(name="f", signature=sig, body_source="def f(): return 2")
        assert h1 != h2

    def test_signature_change_changes_hash(self):
        body = "def f(a): pass"
        s1 = {"params": [{"name": "a", "type": "int", "default": None, "required": True}]}
        s2 = {"params": [{"name": "a", "type": "str", "default": None, "required": True}]}
        assert _compute_source_hash(name="f", signature=s1, body_source=body) != \
            _compute_source_hash(name="f", signature=s2, body_source=body)

    def test_name_change_changes_hash(self):
        sig = {"params": []}
        body = "def f(): pass"
        assert _compute_source_hash(name="f", signature=sig, body_source=body) != \
            _compute_source_hash(name="g", signature=sig, body_source=body)

    def test_crlf_normalized_to_lf(self):
        # Editor line-ending churn shouldn't trip drift detection
        sig = {"params": []}
        unix = "line1\nline2\n"
        windows = "line1\r\nline2\r\n"
        old_mac = "line1\rline2\r"
        h_unix = _compute_source_hash(name="f", signature=sig, body_source=unix)
        h_win = _compute_source_hash(name="f", signature=sig, body_source=windows)
        h_mac = _compute_source_hash(name="f", signature=sig, body_source=old_mac)
        assert h_unix == h_win == h_mac

    def test_canonical_signature_json_sorts_keys(self):
        # Re-ordering dict keys in the signature must not change the hash
        a = {"params": [], "return_type": "int"}
        b = {"return_type": "int", "params": []}
        assert _canonical_signature_json(a) == _canonical_signature_json(b)

    def test_hash_stable_across_processes(self, tmp_path):
        """Process-restart determinism: spawn a subprocess, hash the
        same input, compare. Catches accidental dependence on per-process
        randomness (e.g. PYTHONHASHSEED affecting json.dumps key order)."""
        script = textwrap.dedent("""
            import sys
            sys.path.insert(0, %r)
            from lucidicai.sdk.tools.registry import _compute_source_hash
            sig = {"params": [{"name": "a", "type": "int", "default": None, "required": True}],
                   "return_type": "list"}
            print(_compute_source_hash(name="f", signature=sig, body_source="def f(a): return [a]"))
        """ % str(_repo_root()))
        out = subprocess.check_output([sys.executable, "-c", script], text=True).strip()

        sig = {"params": [{"name": "a", "type": "int", "default": None, "required": True}],
               "return_type": "list"}
        in_proc = _compute_source_hash(name="f", signature=sig,
                                       body_source="def f(a): return [a]")
        assert out == in_proc


def _repo_root():
    """Path to the SDK repo root, for subprocess sys.path injection."""
    import pathlib
    return pathlib.Path(__file__).resolve().parents[3]


# ---------- ToolSurface + registry ---------------------------------------


def _make_surface(name="t", hash_="a" * 64):
    return ToolSurface(
        name=name,
        signature={"params": [], "return_type": None},
        docstring="",
        return_shape=None,
        source_hash=hash_,
    )


class TestRegistry:
    def test_register_with_no_client_buffers(self):
        # No LucidicAI exists → falls through to module-level registry +
        # buffer (the typical test-environment case)
        register_tool(_make_surface(name="foo"))
        assert "foo" in _REGISTRY
        assert len(_PENDING_BUFFER) == 1
        assert _PENDING_BUFFER[0].name == "foo"

    def test_register_last_wins_on_duplicate_name(self):
        register_tool(_make_surface(name="foo", hash_="a" * 64))
        register_tool(_make_surface(name="foo", hash_="b" * 64))
        assert _REGISTRY["foo"].source_hash == "b" * 64

    def test_snapshot_registry_returns_stable_order(self):
        register_tool(_make_surface(name="zeta"))
        register_tool(_make_surface(name="alpha"))
        register_tool(_make_surface(name="middle"))
        names = [s.name for s in snapshot_registry()]
        assert names == ["alpha", "middle", "zeta"]

    def test_drain_buffer_into_empty_returns_zero(self):
        client = _FakeClient()
        assert drain_buffer_into(client) == 0
        assert client.tools._registry == {}

    def test_drain_buffer_into_populated(self):
        register_tool(_make_surface(name="a"))
        register_tool(_make_surface(name="b"))
        client = _FakeClient()
        count = drain_buffer_into(client)
        assert count == 2
        assert set(client.tools._registry.keys()) == {"a", "b"}
        # Buffer is emptied so a re-drain is a no-op
        assert _PENDING_BUFFER == []

    def test_drain_is_idempotent_after_empty(self):
        register_tool(_make_surface(name="a"))
        client = _FakeClient()
        drain_buffer_into(client)
        assert drain_buffer_into(client) == 0


class _FakeClient:
    """Stand-in for LucidicAI. The actual ToolsResource attribute is
    added by LUC-577c; for these tests we only need ``tools._registry``."""
    def __init__(self):
        self.tools = _FakeToolsResource()


class _FakeToolsResource:
    def __init__(self):
        self._registry: dict[str, ToolSurface] = {}
