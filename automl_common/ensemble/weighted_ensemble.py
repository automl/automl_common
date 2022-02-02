from typing import Iterator, Mapping, TypeVar

from pathlib import Path

import numpy as np

from automl_common.backend.stores.model_store import FilteredModelStore
from automl_common.data.math import weighted_sum
from automl_common.ensemble.ensemble import Ensemble
from automl_common.model import Model

MT = TypeVar("MT", bound=Model)


class WeightedEnsemble(Ensemble[MT]):
    """An ensemble that uses weights"""

    def __init__(
        self,
        model_dir: Path,
        weighted_ids: Mapping[str, float],
    ):
        """
        Parameters
        ----------
        model_dir: Path
            The backend object to use

        weighted_ids: Mapping[str, float]
            A mapping from model ids to their weights
        """
        if len(weighted_ids) == 0:
            raise ValueError("Instantiated ensemble with empty `weighted_ids`")

        self.model_dir = model_dir
        self._weighted_ids = weighted_ids

        self._store = FilteredModelStore[MT](
            dir=self.model_dir,
            ids=list(weighted_ids.keys()),
        )

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
        return weighted_sum(weights, predictions)

    def __getitem__(self, model_id: str) -> MT:
        return self._store[model_id].load()

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._weighted_ids.keys()))
