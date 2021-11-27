import pytest
from pytest_lazyfixture import lazy_fixture

from pathlib import Path

from automl_common.backend.context import Context, LocalContext
from automl_common.backend.datamanager import DataManager

@pytest.fixture(scope="function")
def local_context() -> LocalContext:
    return LocalContext()


@pytest.fixture(scope="function")
def other_context() -> LocalContext:
    return LocalContext()


@pytest.fixture(scope="function", params=[lazy_fixture("local_context")])
def context(request) -> Context:
    """All Contexts collected together"""
    return request.param


@pytest.fixture(scope="function")
def datamanager(tmpdir: Path, context: Context) -> DataManager:
    return DataManager(dir=tmpdir, context=context)
