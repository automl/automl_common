from typing import Callable

from pytest_cases import fixture
from test.test_sklearn.test_models.mocks import MockClassifier, MockRegressor


@fixture(scope="function")
def make_classifier() -> Callable[[], MockClassifier]:
    """Classifier = make_classifier()"""

    def _make() -> MockClassifier:
        return MockClassifier()

    return _make


@fixture(scope="function")
def make_regressor() -> Callable[[], MockRegressor]:
    """Regressor = make_regressor()"""

    def _make() -> MockRegressor:
        return MockRegressor()

    return _make
