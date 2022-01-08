from abc import ABC, abstractmethod
from typing import Iterator, Mapping, Sequence, TypeVar

import numpy as np

from automl_common.backend import Backend
from automl_common.backend.accessors import ModelAccessor
from automl_common.model import Model

ModelT = TypeVar("ModelT", bound=Model)


class Ensemble(ABC, Mapping[str, ModelAccessor[ModelT]]):
    def __init__(self, backend: Backend[ModelT], identifiers: Sequence[str]):
        """
        Parameters
        ----------
        identifiers: Sequence[str]
            The identifiers of the models in the ensemble

        backend: Backend
            The context to work from
        """
        self.backend = backend
        self.identifiers = identifiers

    def __getitem__(self, key: str) -> ModelAccessor[ModelT]:
        if key not in self.identifiers:
            raise ValueError(f"Model with {key} not in ensemble, {self.identifiers}")
        return self.backend.models[key]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False

        return key in self.identifiers

    def __iter__(self) -> Iterator[str]:
        return iter(self.identifiers)

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
