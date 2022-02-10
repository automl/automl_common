from typing import Callable

from pytest_cases import fixture

from test.test_model.mocks import MockModel, MockProbabilisticModel


@fixture(scope="function")
def make_model() -> Callable[[], MockModel]:
    """Model = make_model()"""

    def _make() -> MockModel:
        return MockModel()

    return _make


@fixture(scope="function")
def make_probabilistic_model() -> Callable[[int], MockProbabilisticModel]:
    """Model = make_model()"""

    def _make(n_classes: int = 2) -> MockProbabilisticModel:
        return MockProbabilisticModel(n_classes=n_classes)

    return _make
