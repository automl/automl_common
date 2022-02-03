from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Iterable, Iterator, List, Optional, TypeVar, Union

import numpy as np

from automl_common.ensemble.ensemble import Ensemble
from automl_common.model import Model
from automl_common.util.types import SupportsEqualty

MT = TypeVar("MT", bound=Model)
T = TypeVar("T", bound=SupportsEqualty)


class SklearnEnsemble(Ensemble[MT]):
    """An sklearn style ensemble which includes `fit`"""

    def __init__(self) -> None:
        self.__ids: Optional[List[str]] = None

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        metric: Callable[[np.ndarray, np.ndarray], T],
        best: Union[str, Callable[[Iterable[T]], T]] = "max",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> SklearnEnsemble[MT]:
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
        ...

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
    def __getitem__(self, model_id: str) -> MT:
        """Get a model with a given model_id

        Parameters
        ----------
        model_id : str
            The id of the model to get

        Returns
        -------
        MT
            The model
        """
        ...

    @property
    def ids(self) -> List[str]:
        """The ids of the fitted ensemble.

        Returns
        -------
        List[str]
            The ids of the models in the fitted ensemble

            .. code-block:: python

                for id in ensemble.ids
                    ...

        Raises
        ------
        AttributeError
            If the Ensemble is not fitted, will raise an attribute error
        """
        if self.__ids is None:
            raise AttributeError("`ids` is not set, please call `fit` first")

        return self.__ids

    @ids.setter
    def ids(self, values: List[str]) -> None:
        self.__ids = values

    @property
    def models(self) -> Iterator[MT]:
        """Get an iterator over the models in this ensemble

        Returns
        -------
        Iterator[MT]
            The iterator over models

            .. code-block:: python

                for model in ensemble:
                    ...

        Raises
        ------
        AttributeError
            If the ensemble is not fitted
        """
        if not self.__ids:
            raise AttributeError("`models` is not set, please call `fit` first")

        return iter(self[id] for id in self.ids)

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids)

    def __sklearn_is_fitted__(self) -> bool:
        """Sklearns check to see if a model is fitted.

        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        """  # noqa: E501
        return self.__ids is not None
