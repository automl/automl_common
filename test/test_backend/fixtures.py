import pytest
from pytest_lazyfixture import lazy_fixture

from automl_common.backend.context import Context, LocalContext

@pytest.fixture(scope="function")
def local_context() -> LocalContext:
    return LocalContext()

@pytest.fixture(
    scope="function",
    params=[lazy_fixture("local_context")]
)
def context(request) -> Context:
    """All Contexts"""
    return request.param


