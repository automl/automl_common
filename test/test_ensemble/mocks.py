import numpy as np

from automl_common.ensemble import Ensemble
from automl_common.model import Model


class MockEnsemble(Ensemble[Model]):
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x: np.ndarray
            The values to predict on

        Returns
        -------
        np.ndarray
            Just return (x * len(self.models)) what it got as input
        """
        return x
