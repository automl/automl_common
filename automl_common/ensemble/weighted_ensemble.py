from typing import List, Mapping, TypeVar

from pathlib import Path

import numpy as np

from automl_common.data.math import weighted_sum
from automl_common.ensemble.ensemble import Ensemble
from automl_common.model import Model

ModelT = TypeVar("ModelT", bound=Model)


class WeightedEnsemble(Ensemble[ModelT]):
    """An ensemble that uses weights"""

    def __init__(
        self,
        model_dir: Path,
        weighted_identifiers: Mapping[str, float],
    ):
        """
        Parameters
        ----------
        model_dir: Path
            The backend object to use

        weighted_identifiers: Mapping[str, float]
            A mapping from model identifiers to their weights
        """
        super().__init__(
            model_dir=model_dir,
            identifiers=list(weighted_identifiers.keys()),
        )
        self.weighted_identifiers = weighted_identifiers

    @property
    def weights(self) -> List[float]:
        """The weights of this ensemble"""
        return list(self.weighted_identifiers.values())

    def weight(self, identifier: str) -> float:
        """Get the weight of a specific model

        Parameters
        ----------
        key: str
            The identifier of a particular model

        Returns
        -------
        float
            The models weight in the ensemble
        """
        return self.weighted_identifiers[identifier]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predictions of the ensemble on features x

        Parameters
        ----------
        x: np.ndarray
            The values to predict on

        Returns
        -------
        np.ndarray
            The predictions
        """
        predictions = iter(self.models[id].load().predict(x) for id in self.identifiers)
        return weighted_sum(self.weights, predictions)
