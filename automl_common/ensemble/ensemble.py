from abc import ABC, abstractmethod
from typing import Collection, Generic, TypeVar

from pathlib import Path

import numpy as np

from automl_common.backend.stores.model_store import FilteredModelStore
from automl_common.model import Model

ModelT = TypeVar("ModelT", bound=Model)


class Ensemble(ABC, Generic[ModelT]):
    """Manages functionality around using multiple models ensembled in some fashion"""

    def __init__(self, model_dir: Path, identifiers: Collection[str]):
        """
        Parameters
        ----------
        model_dir: Path
            Path to where the models are located

        identifiers: List[str]
            The identifiers of the models in the ensemble
        """
        if len(identifiers) == 0:
            raise ValueError("Instantiated ensemble with empty `identifiers`")

        self.identifiers = list(identifiers)
        self._model_store = FilteredModelStore[ModelT](
            dir=model_dir,
            ids=self.identifiers,
        )

        for id in self.identifiers:
            if not self.models[id].exists():
                raise ValueError(
                    f"No model for id `{id}` found at {self.models[id].path}"
                )

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """The prediction of the ensemble on values x

        Parameters
        ----------
        x: np.ndarray
            The values to predict on

        Returns
        -------
        np.ndarray
            The prediction for the given values
        """
        ...

    @property
    def models(self) -> FilteredModelStore[ModelT]:
        """Store of models in this ensemble"""
        return self._model_store
