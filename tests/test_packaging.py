"""Package metadata declared in setup.py.

setup.py is read with ast, so these tests never import or run setuptools.
"""
import ast
from pathlib import Path

import pytest

SETUP_PY = Path(__file__).resolve().parent.parent / "setup.py"

pytestmark = pytest.mark.skipif(not SETUP_PY.exists(), reason="setup.py not present (installed package)")


def _setup_kwargs():
    tree = ast.parse(SETUP_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup":
            return {kw.arg: kw.value for kw in node.keywords}
    raise AssertionError("no setup() call in setup.py")


def test_python_requires_matches_supported_versions():
    kwargs = _setup_kwargs()
    assert ast.literal_eval(kwargs["python_requires"]) == ">=3.10"
    classifiers = ast.literal_eval(kwargs["classifiers"])
    for minor in ("3.10", "3.11", "3.12", "3.13"):
        assert f"Programming Language :: Python :: {minor}" in classifiers


def test_long_description_is_the_readme_in_markdown():
    kwargs = _setup_kwargs()
    assert "long_description" in kwargs
    assert ast.literal_eval(kwargs["long_description_content_type"]) == "text/markdown"
    assert (SETUP_PY.parent / "README.md").is_file()
