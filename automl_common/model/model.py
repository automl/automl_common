from typing import Protocol

import numpy as np


class Model(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Perform predictions on x, caching the results under `dataset`

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