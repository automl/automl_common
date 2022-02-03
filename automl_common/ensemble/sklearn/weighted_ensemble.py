from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import numpy as np

from automl_common.backend.stores.model_store import ModelStore
from automl_common.data.math import weighted_sum
from automl_common.ensemble.builders.weighted_ensemble import weighted_ensemble_caruana
from automl_common.ensemble.sklearn import SklearnEnsemble
from automl_common.model import Model
from automl_common.util.types import SupportsEqualty

MT = TypeVar("MT", bound=Model)
T = TypeVar("T", bound=SupportsEqualty)

Trajectory = List[Tuple[str, T]]


class SklearnWeightedEnsemble(SklearnEnsemble[MT]):
    """A weighted ensemble"""

    def __init__(self, model_store: ModelStore[MT], size: int):
        """
        Parameters
        ----------
        model_store : ModelStore[MT]
            The model store from which it can access models

        size : int
            The size of the ensemble to generate
        """
        super().__init__()
        self.model_store = model_store
        self.size = size

        self._trajectory: Optional[Trajectory] = None
        self._weights: Optional[Dict[str, float]] = None

    @property
    def tracjectory(self) -> List[Tuple[str, T]]:
        """The tracjectory of the fitting procedue

        Returns
        -------
        List[Tuple[str, T]]
            The trajectory of the ensemble fitting procedure with the model added and
            the overall ensemble performance with that model added.

            .. code-block:: python

                for model_id, perf in ensemble.tracjectory:
                    ...

        Raises
        ------
        AttributeError
            If the ensemble has not been fitted yet
        """
        if self._trajectory is None:
            raise AttributeError("`trajectory` is not set, please call `fit` first")

        return self._trajectory

    @property
    def weights(self) -> Dict[str, float]:
        """The weights of the ensemble

        Returns
        -------
        Dict[str, float]
            A dictionary mapping from model ids to weights

            .. code-block:: python

                {
                    "model_a": 0.6,
                    "model_b": 0.2,
                    "model_c": 0.2,
                }

        Raises
        ------
        AttributeError
            If the ensemble has not been fit yet
        """
        if self._weights is None:
            raise AttributeError("`weights` is not set, please call `fit` first")

        return self._weights

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        metric: Callable[[np.ndarray, np.ndarray], T],
        best: Union[str, Callable[[Iterable[T]], T]] = "max",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> SklearnWeightedEnsemble[MT]:
        """Fit the ensemble to the given targets

        Parameters
        ----------
        x : np.ndarray,
            Fit the ensemble to the given x data

        y : np.ndarray,
            The targets to fit to

        metric : (pred: np.ndarray, target: np.ndarry) -> T
            A metric to use for evaluating candidate models

        best : Union[str, Callable[[Iterable[T]], T]] = "max"
            Select a model member at each stage according to the "min" or "max"
            of the score when adding the model.

            Optionally, you can pass your own `best` function that accepts the output of
            `metric` and returns a `T` which supports equality `==`. This could be
            useful for using multiple metric and return a tuple such as `(x, y, z)`.

        random_state : Optional[Union[int, np.random.RandomState]] = None
            The random_state to use for breaking ties

        Returns
        -------
        np.ndarray
        """
        model_predictions = {
            name: model.load().predict(x) for name, model in self.model_store.items()
        }
        weighted_ids, trajectory = weighted_ensemble_caruana(
            model_predictions=model_predictions,
            targets=y,
            size=self.size,
            metric=metric,
            best=best,
            random_state=random_state,
        )
        self._trajectory = trajectory
        self._weights = weighted_ids
        self.ids = list(weighted_ids.keys())

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Perform ensemble predictions on an array

        Parameters
        ----------
        x : np.ndarray
            The array to perform predictions on

        Returns
        -------
        np.ndarray
            The predictions of the ensemble

        Raises
        ------
        AttributeError
            If the ensemble has not been fit yet
        """
        if self._weights is None:
            raise AttributeError("Please call `fit` first")

        # Keep it iterable as weighted sum can handle it
        ids, weights = zip(*self.weights.items())
        predictions = iter(self.model_store[id].load().predict(x) for id in ids)
        return weighted_sum(weights, predictions)

    def __getitem__(self, model_id: str) -> MT:
        if self._weights is None:
            raise AttributeError("Please call `fit` first")

        if model_id not in self._weights:
            raise KeyError(f"Model {model_id} not in ensemble")

        return self.model_store[model_id].load()
