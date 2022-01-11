from __future__ import annotations

from typing import TYPE_CHECKING, Dict, TypeVar, cast

import numpy as np

from automl_common.ensemble.ensemble import Ensemble
from automl_common.model import Model

if TYPE_CHECKING:
    from automl_common.backend import Backend


ModelT = TypeVar("ModelT", bound=Model)


class WeightedEnsemble(Ensemble[ModelT]):
    """An ensemble that uses weights"""

    def __init__(
        self,
        backend: Backend,
        weighted_identifiers: Dict[str, float],
    ):
        """
        Parameters
        ----------
        backend: Backend
            The backend object to use

        weighted_identifiers: Dict[str, float]
            A dicitonary from model identifiers to their weights
        """
        super().__init__(backend=backend, identifiers=list(weighted_identifiers.keys()))
        self.weighted_identifiers = weighted_identifiers

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
        weighted_prediction = sum(
            weight * self[id].load().predict(x)
            for id, weight in self.weighted_identifiers.items()
            if weight > 0
        )

        # The models don't return scalers so we can be sure it's np.ndarray
        return cast(np.ndarray, weighted_prediction)
