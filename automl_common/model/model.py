from typing_extensions import (  # TODO: update with Python 3.8
    Protocol,
    runtime_checkable,
)

import numpy as np


@runtime_checkable
class Model(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Perform predictions on x

        Parameters
        ----------
        x: np.ndarray
            The values to predict on

        Returns
        -------
        np.ndarray
            The predictions of this model
        """
        ...


@runtime_checkable
class ProbabilisticModel(Model, Protocol):
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Perform probability predictions on x

        Parameters
        ----------
        x: np.ndarray
            The values to predict on

        Returns
        -------
        np.ndarray
            The predictions of this model
        """
        ...
