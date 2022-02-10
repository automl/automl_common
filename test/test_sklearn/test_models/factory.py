from typing import Callable

from pytest_cases import fixture

from test.test_sklearn.test_models.mocks import (
    MockClassifier,
    MockPredictor,
    MockProbabilisticPredictor,
    MockRegressor,
)


@fixture(scope="function")
def make_predictor() -> Callable[[], MockPredictor]:
    """Model = make_predictor()"""

    def _make() -> MockPredictor:
        return MockPredictor()

    return _make


@fixture(scope="function")
def make_probabilistic_predictor() -> Callable[[int], MockProbabilisticPredictor]:
    """Model = make_probabilistic_predictor(n_classes: int =2)"""

    def _make(n_classes: int = 2) -> MockProbabilisticPredictor:
        return MockProbabilisticPredictor(n_classes=n_classes)

    return _make


@fixture(scope="function")
def make_classifier() -> Callable[[int], MockClassifier]:
    """Model = make_classifier(n_classes: int = 2)"""

    def _make(n_classes: int = 2) -> MockClassifier:
        return MockClassifier(n_classes=n_classes)

    return _make


@fixture(scope="function")
def make_regressor() -> Callable[[], MockRegressor]:
    """Model = make_regressor()"""

    def _make() -> MockRegressor:
        return MockRegressor()

    return _make
