import numpy as np

from automl_common.model import Model
from automl_common.util.random import as_random_state

from test.data import DEFAULT_SEED, arrhash


class MockModel(Model):
    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = seed

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
        rs = as_random_state(self.seed + arrhash(x))
        return rs.random(size=len(x))
