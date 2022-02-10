from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from typing_extensions import Literal

import numpy as np

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble.builders.single_best import single_best
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


class SingleEnsemble(Ensemble[PredictorT]):
    """An ensemble that select the single best.

    Parameters
    ----------
    metric : (np.ndarray, np.ndarray) -> Orderable
        A metric to evaluate models with. Should return an Orderable result

    select: "min" | "max"
        How to order results of the metric

    random_state : Optional[Union[int, np.random.RandomState]] = None
        The random_state to use for breaking ties

    model_store : Optional[ModelStore[PredictorT]] = None
        A store of models to use during fit
    """

    @abstractmethod
    def __init__(
        self,
        metric: Callable[[np.ndarray, np.ndarray], Orderable],
        select: Literal["min", "max"],
        *,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        model_store: ModelStore[PredictorT] = None,
    ) -> None:
        super().__init__(model_store=model_store)
        self.metric = metric
        self.select = select
        self.random_state = random_state

    @property
    def model(self) -> PredictorT:
        """Get the model of this ensemble

        Returns
        -------
        PredictorT
            Returns the model of this ensemble.

        Raises
        ------
        AttributeError
            If the ensemble has not been fit yet
        """
        return self.__getitem__(self.id)

    @property
    def id(self) -> str:
        """Get the id of the selected model
        Returns
        -------
        str
            The id of the selected model
        """
        if not self.__sklearn_is_fitted__():
            raise AttributeError("Please call `fit` first")

        return self.model_id_  # type: ignore

    @classmethod
    def _fit_attributes(cls) -> List[str]:
        return super()._fit_attributes() + ["random_state_", "model_id_"]

    def _fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> List[str]:
        """Fit the ensemble to the given targets

        Implementing classes must set self.ids

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
        model_predictions = iter(
            (model, self[model].predict(x)) for model in self.model_store
        )

        self.random_state_ = as_random_state(self.random_state)

        selected_id = single_best(
            model_predictions=model_predictions,
            targets=y,
            metric=self.metric,
            select=self.select,
            random_state=self.random_state_,
        )

        self.model_id_ = selected_id
        return [self.model_id_]

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

        Raises
        ------
        AttributeError
            If the ensemble has not been fit yet
        """
        return self.model.predict(x)

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


class SingleRegressorEnsemble(
    SingleEnsemble[RegressorT],
    RegressorEnsemble[RegressorT],
):
    def __init__(
        self,
        *,
        model_store: Optional[ModelStore[RegressorT]] = None,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = rmse,
        select: Literal["min", "max"] = "min",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        super().__init__(
            model_store=model_store,
            metric=metric,
            select=select,
            random_state=random_state,
        )


class SingleClassifierEnsemble(
    SingleEnsemble[ClassifierT],
    ClassifierEnsemble[ClassifierT],
):
    def __init__(
        self,
        *,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = accuracy,
        select: Literal["min", "max"] = "max",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        model_store: Optional[ModelStore[ClassifierT]] = None,
    ) -> None:
        super().__init__(
            metric=metric,
            select=select,
            random_state=random_state,
            model_store=model_store,
        )

    def _predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get probability predictions for the data x

        Underlying class must implement `_predict_proba()`

        Parameters
        ----------
        x : np.ndarray
            The data to predict on

        Returns
        -------
        np.ndarray
            The predictions for x
        """
        return self.model.predict_proba(x)
