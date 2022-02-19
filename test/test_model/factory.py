from typing import Callable

from pytest_cases import fixture
from test.test_model.mocks import MockModel


@fixture(scope="function")
def make_model() -> Callable[[], MockModel]:
    """Model = make_model()"""

    def _make() -> MockModel:
        return MockModel()

    return _make
