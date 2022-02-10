from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from typing_extensions import Literal  # TODO, remove with Python 3.8

import numpy as np

from automl_common.backend.stores.model_store import ModelStore
from automl_common.data.math import weighted_sum
from automl_common.ensemble.builders.weighted_ensemble import weighted_ensemble_caruana
from automl_common.metrics import accuracy, rmse
from automl_common.sklearn.ensemble.ensemble import (
    ClassifierEnsemble,
    Ensemble,
    RegressorEnsemble,
)
from automl_common.sklearn.model import Classifier, Predictor, Regressor
from automl_common.util.random import as_random_state
from automl_common.util.types import Orderable

PredictorT = TypeVar("PredictorT", bound=Predictor)
RegressorT = TypeVar("RegressorT", bound=Regressor)
ClassifierT = TypeVar("ClassifierT", bound=Classifier)


class WeightedEnsemble(Ensemble[PredictorT]):
    """An ensemble of size ``n`` that selects a weighted ensemble.

    Parameters
    ----------
    metric : (np.ndarray, np.ndarray) -> Orderable
        A metric to evaluate models with. Should return an Orderable result

    select: "min" | "max"
        How to order results of the metric

    model_store : Optional[ModelStore[PredictorT]] = None
        A store of models to use during fit, won't really work without it sepcified.

    size: int = 10
        The size of the ensemble to fit

    random_state : Optional[Union[int, np.random.RandomState]] = None
        The random_state to use for breaking ties
    """

    def __init__(
        self,
        metric: Callable[[np.ndarray, np.ndarray], Orderable],
        select: Literal["min", "max"],
        *,
        model_store: Optional[ModelStore[PredictorT]] = None,
        size: int = 10,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        super().__init__(model_store=model_store)
        self.metric = metric
        self.select = select
        self.size = size
        self.random_state = random_state

        # Set in `fit`
        # self.trajectory_
        # self.weights_
        # self.random_state_

    @property
    def tracjectory(self) -> List[Tuple[str, Orderable]]:
        """The tracjectory of the fitting procedue

        Returns
        -------
        List[Tuple[str, Orderable]]
            The trajectory of the ensemble fitting procedure with the model added and
            the overall ensemble performance with that model added.

            Note
            ----


            .. code-block:: python

                for model_id, perf in ensemble.tracjectory:
                    ...

        Raises
        ------
        AttributeError
            If the ensemble has not been fitted yet
        """
        if not self.__sklearn_is_fitted__():
            raise AttributeError("Please call `fit` first")

        return self.trajectory_  # type: ignore

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
        if not self.__sklearn_is_fitted__():
            raise AttributeError("Please call `fit` first")

        return self.weights_

    @classmethod
    def _fit_attributes(cls) -> List[str]:
        """The attributes required by this ensemble.

        Implementing classes should take into account their superclasses attributes

        .. code-block:: python

            @classmethod
            def _attributes(cls) -> List[str]:
                return super()._attributes() + [...]

        Returns
        -------
        List[str]
            The attributes of this ensemble
        """
        return super()._fit_attributes() + ["weights_", "trajectory_", "random_state_"]

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get the parameters of this ensemble

        Parameters
        ----------
        deep : bool = True
            Whether to get the parameters of subestimators

        Returns
        -------
        Dict
            A dicitonary mapping from parameters to values
        """
        return {
            "metric": self.metric,
            "select": self.select,
            "size": self.size,
            "random_state": self.random_state,
            **super().get_params(deep=deep),
        }

    def _fit(self, x: np.ndarray, y: np.ndarray) -> List[str]:
        """Fit the ensemble to the given targets

        Parameters
        ----------
        x : np.ndarray,
            Fit the ensemble to the given x data

        y : np.ndarray,
            The targets to fit to

        Returns
        -------
        List[str]
            The ids of the models selected

        """
        model_predictions = {
            name: model.load().predict(x) for name, model in self.model_store.items()
        }

        self.random_state_ = as_random_state(self.random_state)

        weighted_ids, trajectory = weighted_ensemble_caruana(
            model_predictions=model_predictions,
            targets=y,
            size=self.size,
            metric=self.metric,
            select=self.select,
            random_state=self.random_state_,
        )
        self.trajectory_ = trajectory
        self.weights_ = weighted_ids

        ids = list(weighted_ids.keys())
        return ids

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Perform ensemble predictions on an array

        Parameters
        ----------
        x : np.ndarray
            The array to perform predictions on

        Returns
        -------
        np.ndarray
            The predictions of the ensemble
        """
        # Keep it iterable as weighted sum can handle it
        ids, weights = zip(*self.weights.items())
        predictions = iter(self.model_store[id].load().predict(x) for id in ids)
        return weighted_sum(weights, predictions)


class WeightedRegressorEnsemble(
    RegressorEnsemble[RegressorT],
    WeightedEnsemble[RegressorT],
):
    def __init__(
        self,
        *,
        model_store: Optional[ModelStore[RegressorT]] = None,
        size: int = 10,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = rmse,
        select: Literal["min", "max"] = "min",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        super().__init__(
            model_store=model_store,
            size=size,
            metric=metric,
            select=select,
            random_state=random_state,
        )


class WeightedClassifierEnsemble(
    ClassifierEnsemble[ClassifierT],
    WeightedEnsemble[ClassifierT],
):
    def __init__(
        self,
        *,
        model_store: Optional[ModelStore[ClassifierT]] = None,
        size: int = 10,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = accuracy,
        select: Literal["min", "max"] = "max",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        super().__init__(
            model_store=model_store,
            size=size,
            metric=metric,
            select=select,
            random_state=random_state,
        )
