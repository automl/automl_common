import os
import re
from glob import glob
from pathlib import Path

from pytest import fixture

from automl_common.backend.context import Context, LocalContext

# Load in other pytest modules, in this case fixtures
pytest_plugins = [
    "test_backend.fixtures"
]

def test_id(request) -> str:
    """Gets a unique id for all tests, even parameterized tests"""
    return re.match(r".*::(.*)$", request.node.nodeid).group(1).replace("[", "_").replace("]", "")


@fixture(scope="function")
def tmpfile(request, tmpdir) -> Path:
    """Returns the path to a tmpfile in a tmpdir

    /tmp/.../.../test_func_name_parametrization/test_func_name_parametrization
    """
    return tmpdir / test_id(request)
