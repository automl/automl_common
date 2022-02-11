"""
Following:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeVar
from copy import copy

import logging

import numpy as np

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble.ensemble import Ensemble as BaseEnsemble
from automl_common.sklearn.model import Classifier, Predictor, Regressor

PredictorT = TypeVar("PredictorT", bound=Predictor)
RegressorT = TypeVar("RegressorT", bound=Regressor)
ClassifierT = TypeVar("ClassifierT", bound=Classifier)

# https://www.python.org/dev/peps/pep-0673/#motivation
SelfT = TypeVar("SelfT", bound="Ensemble")  # TODO Python 3.11 or typing_extensions

logger = logging.getLogger(__name__)


class Ensemble(BaseEnsemble[PredictorT]):
    """An sklearn style ensemble which includes `fit`.

    A base class for sklearn style ensembles.

    Must implement `_fit`, `_predict` and `_fit_attributes`.

    Any easy convention for `_fit_attributes` is

    .. code-block:: python

        @classmethod
        def _fit_attributes(cls) -> List[str]:
            return super()._fit_attributes() + ["myattribute_", "cat_"]

    On top of this, implementing classes should also implement `get_params` for full
    sklearn compatibility.

    Note
    ----
    Technically we shouldn't allow for `model_store = None` in `__init__` but for
    full compatibility with sklearn, we need to do so. If this is the case, when
    no model_store was recieved, the implementing object must call `self._fail_fit()`
    and it's predictions should return the shape of NaN as seen during fit. This is
    accessible in `self._target_shape`

    None but for sklearn compatibility, we have to allow it. If not specified, it
    must pretend nothing was chosen

    Parameters
    ----------
    model_store: Optional[ModelStore[PredictorT]] = None
        A model store from which models can be chosen.

        Note
        ----
        Will not operate properly without it.
    """

    @abstractmethod
    def __init__(self, *, model_store: Optional[ModelStore[PredictorT]] = None):
        self._model_store = model_store

        # Used if model_store is None to indicate it's failed
        self._on_fail_predict_shape: Optional[Tuple[int, ...]] = None

    @abstractmethod
    def _fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> List[str]:
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
            The list of models selected
        """
        ...

    @abstractmethod
    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Get predictions for the data x

        Underlying class must implement `_predict()`

        Parameters
        ----------
        x : np.ndarray
            The data to predict on

        Returns
        -------
        np.ndarray
            The predictions for x

        Raises
        ------
        AttributeError
            Raises if the ensemble has not been fit yet
        """
        ...

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
        return ["ids_"]

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
        if not self.__sklearn_is_fitted__():
            raise AttributeError("Please call `fit` first")

        return self.ids_  # type: ignore

    @property
    def model_store(self) -> ModelStore[PredictorT]:
        """Get the ModelStore this ensemble is using

        Returns
        -------
        ModelStore[PredictorT]
            The model store that the ensemble is using
        """
        if self._model_store is None:
            raise RuntimeError("Ensemble was built with `model_store` == None")

        return self._model_store

    def fit(self: SelfT, x: np.ndarray, y: np.ndarray) -> SelfT:
        """Fit the ensemble to the given targets

        Implementing classes must implement `_fit`.

        Parameters
        ----------
        x : np.ndarray,
            Fit the ensemble to the given x data

        y : np.ndarray,
            The targets to fit to

        Returns
        -------
        self
        """
        # Reset attributes
        for attr in self._fit_attributes():
            delattr(self, attr)
            self._on_fail_predict_shape = None

        # If we have no model_store, we can't fit anything
        if self._model_store is None:
            for attr in self._fit_attributes():
                setattr(self, attr, None)

            self._on_fail_predict_shape = y.shape
            self.ids_ = []

            return self

        # Call the underlying fit implementation
        self.ids_ = self._fit(x, y)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Get predictions for the data x

        Underlying class must implement `_predict()`

        Parameters
        ----------
        x : np.ndarray
            The data to predict on

        Returns
        -------
        np.ndarray
            The predictions for x

        Raises
        ------
        AttributeError
            Raises if the ensemble has not been fit yet
        """
        if not self.__sklearn_is_fitted__():
            raise AttributeError("Please call `fit` first")

        # If fitted with no model store
        if self._on_fail_predict_shape is not None:
            assert self._model_store is None
            logger.warning(
                "Predicting with Ensemble that was constructed with no ModelStore."
                "\nWill return all 0's."
            )

            # Copy the length and return the rest of the expected shape
            shape = self._on_fail_predict_shape
            if len(shape) == 1:
                return np.zeros(len(x))
            else:
                return np.zeros((len(x), *shape[1:]))

        return self._predict(x)

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
        return {"model_store": self._model_store}

    def set_params(self: SelfT, **parameters: Any) -> SelfT:
        """Set the params of this ensemble

        Parameters
        ----------
        **parameters : Any
            The parameters and their values to set

        Returns
        -------
        Ensemble[PredictorT]
            self with the parameters set
        """
        for param, value in parameters.items():
            if param == "model_store":
                param = "_model_store"  # We underscore prefix it

            setattr(self, param, value)

        return self

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids)

    def __getitem__(self, model_id: str) -> PredictorT:
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
        if self.model_store is None:
            raise RuntimeError("No ModelStore to retreive from")

        if not self.__sklearn_is_fitted__():
            raise AttributeError("Please call `fit` first")

        if model_id not in self.ids:  # type: ignore
            raise KeyError(f"Model {model_id} not in ensemble")

        return self.model_store[model_id].load()

    def __sklearn_is_fitted__(self) -> bool:
        """Sklearns check to see if a model is fitted.

        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        """  # noqa: E501
        return all(hasattr(self, attr) for attr in self._fit_attributes())


class RegressorEnsemble(Ensemble[RegressorT]):
    pass


class ClassifierEnsemble(Ensemble[ClassifierT]):
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
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

        Raises
        ------
        AttributeError
            Raises if the ensemble has not been fit yet
        """
        if not self.__sklearn_is_fitted__():
            raise AttributeError("Please call `fit` first")

        return self._predict_proba(x)

    @abstractmethod
    def _predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get probability predictions for the data x

        Parameters
        ----------
        x : np.ndarray
            The data to predict on

        Returns
        -------
        np.ndarray
            The predictions for x

        Raises
        ------
        AttributeError
            Raises if the ensemble has not been fit yet
        """
        ...
