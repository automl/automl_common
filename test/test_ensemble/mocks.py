from typing import Collection, Iterator, TypeVar

import numpy as np

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble import Ensemble
from automl_common.model import Model

MT = TypeVar("MT", bound=Model)
ID = TypeVar("ID")


class MockEnsemble(Ensemble[ID, MT]):
    def __init__(self, model_store: ModelStore[ID, MT], ids: Collection[ID]):
        self.model_store = model_store
        self._ids = ids

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

    def __getitem__(self, model_id: ID) -> MT:
        return self.model_store[model_id].load()

    def __iter__(self) -> Iterator[ID]:
        return iter(self._ids)
