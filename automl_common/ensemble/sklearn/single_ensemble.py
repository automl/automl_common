from __future__ import annotations

from typing import Callable, Iterable, Optional, TypeVar, Union

import numpy as np

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble.builders.single_best import single_best
from automl_common.ensemble.sklearn import SklearnEnsemble
from automl_common.model import Model
from automl_common.util.types import SupportsEqualty

MT = TypeVar("MT", bound=Model)
T = TypeVar("T", bound=SupportsEqualty)


class SklearnSingleEnsemble(SklearnEnsemble[MT]):
    """An ensemble that select the single best.

    Follows the sklearn style of having `fit`
    """

    def __init__(self, model_store: ModelStore[MT]) -> None:
        super().__init__()
        self.model_store = model_store

        self._model_id: Optional[str] = None

    @property
    def model(self) -> MT:
        """Get the model of this ensemble

        Returns
        -------
        MT
            Returns the model of this ensemble.

        Raises
        ------
        AttributeError
            If the ensemble has not been fit yet
        """
        if self._model_id is None:
            raise AttributeError("`model` is None, please call `fit` first")

        return self.model_store[self._model_id].load()

    @property
    def id(self) -> str:
        """Get the id of the selected model

        Returns
        -------
        str
            The id of the selected model
        """
        if self._model_id is None:
            raise AttributeError("`id` is None, please call `fit` first")

        return self._model_id

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        metric: Callable[[np.ndarray, np.ndarray], T],
        best: Union[str, Callable[[Iterable[T]], T]] = "max",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> SklearnSingleEnsemble[MT]:
        """Fit the ensemble to the given targets

        Implementing classes must set self.ids

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
        model_predictions = iter(
            (name, model.load().predict(x)) for name, model in self.model_store.items()
        )

        selected_id = single_best(
            model_predictions=model_predictions,
            targets=y,
            metric=metric,
            best=best,
            random_state=random_state,
        )

        self._model_id = selected_id
        self.ids = [selected_id]

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
        if self._model_id is None:
            raise AttributeError("Please call `fit` first")

        # Keep it iterable as weighted sum can handle it
        return self.model.predict(x)

    def __getitem__(self, model_id: str) -> MT:
        if self._model_id is None:
            raise AttributeError("Please call `fit` first")

        if model_id != self._model_id:
            raise KeyError("Model {model_id} not in ensemble")

        return self.model_store[model_id].load()
