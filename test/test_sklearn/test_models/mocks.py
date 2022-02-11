from __future__ import annotations

import numpy as np

from automl_common.data.math import normalize
from automl_common.sklearn.model import Classifier, Regressor
from automl_common.util.random import as_random_state

from test.data import DEFAULT_SEED, arrhash


class MockRegressor(Regressor):
    """A MockRegressor which correctly gives output shapes"""

    def fit(self, x: np.ndarray, y: np.ndarray) -> MockRegressor:
        """Fit to the data

        Parameters
        ----------
        x : np.ndarray
            The x data to fit to

        y : np.ndarray
            The y data to fit to

        Returns
        -------
        MockRegressor
            Self
        """
        assert len(x) == len(y)
        self.x_shape = x.shape
        self.y_shape = y.shape
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions

        Parameters
        ----------
        x : np.ndarray
            The data to predict on

        Returns
        -------
        np.ndarray
            Predictions in the correct output shape
        """
        shape = self.y_shape
        if len(shape) == 1:
            return np.zeros(len(x))
        else:
            return np.zeros((len(x), *shape[1:]))


class MockClassifier(Classifier):
    """A MockClassifier which correctly gives output shapes and classes"""

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> MockClassifier:
        """Fit the MockClassifier, saving shapes and classes seen

        Parameters
        ----------
        x : np.ndarray
            The x data to fit on

        y : np.ndarray
            The y data to fit on

        Returns
        -------
        MockClassifier
            Self
        """
        assert len(x) == len(y)
        self.x_shape = x.shape
        self.y_shape = y.shape
        self.classes_ = np.unique(y, axis=0)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate Predictions, remeber the labels it's seen

        Parameters
        ----------
        x : np.ndarray
            The x data to predict on

        Returns
        -------
        np.ndarray
            The predicted classes
        """
        rs = as_random_state(self.seed + arrhash(x))
        idxs = rs.choice(len(self.classes_), size=len(x))
        return self.classes_[idxs]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Generate probability predictions

        Parameters
        ----------
        x : np.ndarray
            The data to predict on

        Returns
        -------
        np.ndarray
            The probability predictions
        """
        rs = as_random_state(self.seed + arrhash(x))
        shape = (len(x), len(self.classes_))
        return normalize(rs.random(size=shape), axis=1)
