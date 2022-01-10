import os
import re
from pathlib import Path

from pytest import FixtureRequest
from pytest_cases import fixture

# Load in other pytest modules, in this case fixtures
here = os.path.dirname(os.path.realpath(__file__))

pytest_plugins = []


def _as_module(root: str, path: str) -> str:
    path = os.path.join(root, path)
    path = path.replace(here, "")
    path = path.replace(".py", "")
    path = path.replace(os.path.sep, ".")[1:]
    return "test." + path


for root, dirs, files in os.walk(here, topdown=True):
    dirs[:] = [d for d in dirs if d.startswith("test")]
    pytest_plugins += [_as_module(root, f) for f in files if f.endswith("fixtures.py")]


def test_id(request: FixtureRequest) -> str:
    """Gets a unique id for all tests, even parameterized tests

    Returns
    -------
    str
        A unique id for the test
    """
    return (
        re.match(r".*::(.*)$", request.node.nodeid)  # type: ignore
        .group(1)
        .replace("[", "_")
        .replace("]", "")
    )


@fixture(scope="function")
def tmpfile(request: FixtureRequest, tmpdir: Path) -> Path:
    """Returns the path to a tmpfile in a tmpdir

    /tmp/.../.../test_func_name_parametrization/test_func_name_parametrization
    """
    return tmpdir / test_id(request)
