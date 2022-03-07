from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Mapping, TypeVar

import numpy as np

from automl_common.model import Model
from automl_common.util.types import EqualityMixin

MT = TypeVar("MT", bound=Model)
KT = TypeVar("KT")

x: List[str] = []


class Ensemble(ABC, Model, EqualityMixin, Mapping[KT, MT]):
    """Manages functionality around using multiple models ensembled in some fashion"""

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

    @abstractmethod
    def __getitem__(self, model_id: KT) -> MT:
        """Get a model with a given model_id

        Parameters
        ----------
        model_id : KT
            The id of the model to get

        Returns
        -------
        MT
            The model
        """
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[KT]:
        """Iter over the model ids in the ensemble

        Returns
        -------
        Iterator[KT]
            An iteratory over the model ids in this ensemble
        """
        ...

    @property
    def ids(self) -> List[KT]:
        """Get ths ids of models in this ensemble

        Returns
        -------
        List[KT]
            The identifiers of the models in this ensemble
        """
        return list(iter(self))

    def __len__(self) -> int:
        return len(list(iter(self)))  # Relying on self.ids causes recursion errors

    def __contains__(self, model_id: Any) -> bool:
        return model_id in self.ids
