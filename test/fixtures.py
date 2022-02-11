from typing import Callable, Tuple

from pathlib import Path

import numpy as np
from pytest import FixtureRequest
from pytest_cases import fixture, fixture_ref, parametrize

from test.data import xy


@fixture(scope="function")
@parametrize("impl", [fixture_ref("tmp_path")])
def path(request: FixtureRequest, impl: Path) -> Path:
    """Forwards different kinds of paths we might want to use

    In the future we may want an AWSPath: Path to run tests with

    The should delete themselves after use
    """
    return impl


@fixture(scope="function")
def make_xy() -> Callable[..., Tuple[np.ndarray, np.ndarray]]:
    """Please see documentation of `def xy` in `test/data.py`"""
    return xy
