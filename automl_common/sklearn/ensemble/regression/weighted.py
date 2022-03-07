from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from typing_extensions import Literal  # TODO, remove with Python 3.8

from functools import partial

import numpy as np
from sklearn.metrics import mean_squared_error

from automl_common.backend.stores.model_store import ModelStore
from automl_common.data.math import weighted_sum
from automl_common.ensemble.builders.weighted_ensemble import weighted_ensemble_caruana
from automl_common.sklearn.ensemble.regression.base import RegressorEnsemble
from automl_common.sklearn.ensemble.weighted import WeightedEnsemble
from automl_common.sklearn.model import Regressor
from automl_common.util.random import as_random_state
from automl_common.util.types import Orderable

RT = TypeVar("RT", bound=Regressor)
ID = TypeVar("ID")


rmse = partial(mean_squared_error, squared=True)


class WeightedRegressorEnsemble(RegressorEnsemble[ID, RT], WeightedEnsemble[ID, RT]):
    """TODO"""

    def __init__(
        self,
        *,
        model_store: ModelStore[ID, RT],
        tags: Optional[Dict[str, Any]] = None,
        size: int = 10,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = rmse,
        select: Literal["min", "max"] = "min",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        super().__init__(model_store=model_store, tags=tags)
        self.size = size
        self.metric = metric
        self.select = select
        self.random_state = random_state

    def _fit(self, x: np.ndarray, y: np.ndarray, pred_key: Optional[str] = None) -> List[ID]:
        """Fit the ensemble to the given targets

        Parameters
        ----------
        x : np.ndarray,
            Fit the ensemble to the given x data

        y : np.ndarray,
            The targets to fit to

        pred_key: Optional[str] = None
            The name of predictions to try and load instead of loading the full models

        Returns
        -------
        List[ID]
            The ids of the models selected
        """
        self.random_state_ = as_random_state(self.random_state)
        model_predictions = dict(self._model_predictions(x, pred_key))

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
        predictions = iter(
            self.model_store[id].load().predict(x) for id in ids
        )  # pragma: no cover, not sure why it's not covered
        return weighted_sum(predictions, weights=np.asarray(weights))

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
            "random_state": self.random_state,
            **super().get_params(deep=deep),
        }

    @classmethod
    def _fit_attributes(cls) -> List[str]:
        return super()._fit_attributes() + ["random_state_"]
