import numpy as np

from automl_common.sklearn.model import (
    Classifier,
    Predictor,
    ProbabilisticPredictor,
    Regressor,
)


class MockPredictor(Predictor):
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x: np.ndarray
            The values to predict on

        Returns
        -------
        np.ndarray
            Just return what it got as input
        """
        return x


class MockProbabilisticPredictor(ProbabilisticPredictor):
    def __init__(self, n_classes: int = 2):
        assert n_classes > 1
        self.n_classes = n_classes

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x: np.ndarray
            The values to predict on

        Returns
        -------
        np.ndarray
            Just return what it got as input
        """
        x = np.random.random((len(x), self.n_classes))
        return x / x.sum(axis=1, keepdims=True)


class MockRegressor(MockPredictor, Regressor):
    pass


class MockClassifier(MockProbabilisticPredictor, Classifier):
    pass
