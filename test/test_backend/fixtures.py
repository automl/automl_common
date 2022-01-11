from pathlib import Path

from pytest import FixtureRequest
from pytest_cases import fixture

from automl_common.backend import Backend
from automl_common.backend.contexts import Context

from test.conftest import test_id
from test.test_model.fixtures import MockModel


@fixture(scope="function")
def backend(
    request: FixtureRequest,
    tmp_path: Path,
    context: Context,
) -> Backend[MockModel]:
    """
    Parameters
    ----------
    test_id: str
        A unique test id to name the backend

    tmp_path: Path
        The path to create the backend at

    context: Context
        The context to use
    """
    return Backend(name=test_id(request), path=tmp_path, context=context)
