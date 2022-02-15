from typing import Iterator, Mapping, TypeVar

import numpy as np

from automl_common.backend.stores.model_store import ModelStore
from automl_common.data.math import weighted_sum
from automl_common.ensemble.ensemble import Ensemble
from automl_common.model import Model

MT = TypeVar("MT", bound=Model)


class WeightedEnsemble(Ensemble[MT]):
    """An ensemble that uses weights"""

    def __init__(
        self,
        model_store: ModelStore[MT],
        weighted_ids: Mapping[str, float],
    ):
        """
        Parameters
        ----------
        model_store: ModelStore[MT]
            The backend object to use

        weighted_ids: Mapping[str, float]
            A mapping from model ids to their weights
        """
        if len(weighted_ids) == 0:
            raise ValueError("Instantiated ensemble with empty `weighted_ids`")

        missing = set(weighted_ids).difference(set(model_store))
        if len(missing) > 0:
            raise ValueError(f"Model(s) {missing} not in {model_store}")

        self._model_store = model_store
        self._weighted_ids = weighted_ids

    @property
    def weights(self) -> Mapping[str, float]:
        """The weights of this ensemble"""
        return self._weighted_ids

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
        ids, weights = zip(*self.weights.items())
        predictions = iter(self[id].predict(x) for id in ids)
        return weighted_sum(predictions, weights=np.asarray(weights))

    def __getitem__(self, model_id: str) -> MT:
        return self._model_store[model_id].load()

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._weighted_ids.keys()))
