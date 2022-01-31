from typing import Collection, Iterator

from pathlib import Path

import numpy as np

from automl_common.backend.stores.model_store import FilteredModelStore
from automl_common.ensemble import Ensemble
from automl_common.model import Model


class MockEnsemble(Ensemble[Model]):
    def __init__(self, model_dir: Path, ids: Collection[str]):
        self.model_dir = model_dir
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

    def __getitem__(self, model_id: str) -> Model:
        store = FilteredModelStore[Model](self.model_dir, ids=self.ids)
        return store[model_id].load()

    def __iter__(self) -> Iterator[str]:
        return iter(self._ids)
