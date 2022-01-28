from typing import Callable

from pytest_cases import fixture

from automl_common.model import Model

from test.test_model.mocks import MockModel


@fixture(scope="function")
def make_model() -> Callable[[], Model]:
    """Model = make_model()"""

    def _make() -> MockModel:
        return MockModel()

    return _make
