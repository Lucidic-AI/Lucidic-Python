import functools

import pytest

from .harness import run_scenario


@pytest.fixture
def capture(tmp_path):
    return functools.partial(run_scenario, tmp_path)
