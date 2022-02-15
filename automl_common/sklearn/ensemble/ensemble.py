"""
Following:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Optional, TypeVar

import logging
import warnings

import numpy as np

from automl_common.backend.stores.model_store import ModelStore
from automl_common.data.validate import jagged
from automl_common.ensemble.ensemble import Ensemble as BaseEnsemble
from automl_common.sklearn.ensemble.util import tag_accumulate
from automl_common.sklearn.model import Classifier, Predictor, Regressor
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

PredictorT = TypeVar("PredictorT", bound=Predictor)
RegressorT = TypeVar("RegressorT", bound=Regressor)
ClassifierT = TypeVar("ClassifierT", bound=Classifier)

# https://www.python.org/dev/peps/pep-0673/#motivation
SelfT = TypeVar("SelfT", bound="Ensemble")  # TODO Python 3.11 or typing_extensions

logger = logging.getLogger(__name__)


class Ensemble(Predictor, BaseEstimator, BaseEnsemble[PredictorT]):
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
    no model_store was recieved, we simple error on fit.

    Parameters
    ----------
    model_store: Optional[ModelStore[PredictorT]] = None
        A model store from which models can be chosen.
        Will not operate properly without it.
    """

    # Not sure if we then make the parameter Non-Optional?
    _required_parameters: List[str] = ["model_store"]

    @abstractmethod
    def __init__(self, *, model_store: Optional[ModelStore[PredictorT]] = None):
        self.model_store = model_store

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
        NotFittedError
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
        NotFittedError
            If the Ensemble is not fitted, will raise an attribute error
        """
        check_is_fitted(self)
        return self.ids_  # type: ignore

    @property
    def _model_store(self) -> ModelStore[PredictorT]:
        """Internal method to access the model store in a type safe way

        Returns
        -------
        ModelStore[PredictorT]
            The model store this object was constructed with
        """
        if self.model_store is None:
            raise AttributeError("Constructed without `model_store`")

        return self.model_store

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
        if self.model_store is None:
            raise RuntimeError("Can't fit without model store")

        x, y = check_X_y(x, y, accept_sparse=True, multi_output=True)

        # Reset attributes
        for attr in self._fit_attributes():
            if hasattr(self, attr):
                delattr(self, attr)

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
        NotFittedError
            Raises if the ensemble has not been fit yet
        """
        check_is_fitted(self)
        x = check_array(x, accept_sparse=True)

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
        return {"model_store": self.model_store}

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
        check_is_fitted(self)

        assert self.model_store is not None  # Protected by requiring `fit`

        if model_id not in self.ids:  # type: ignore
            raise KeyError(f"Model {model_id} not in ensemble")

        return self.model_store[model_id].load()

    def __sklearn_is_fitted__(self) -> bool:
        """Sklearns check to see if a model is fitted.

        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        """  # noqa: E501
        return all(hasattr(self, attr) for attr in self._fit_attributes())

    def _more_tags(self) -> Dict[str, Any]:
        """
        Note
        ----
        We assume this method is only called for testing purposes by sklearn.
        This is quite an expensive function so we raise a warning if this is
        not the case.
        """
        warnings.warn("Expensive function `_more_tags` called.")
        # Get all model tags
        model_tags: List[Dict[str, bool]] = []
        for model in self._model_store.values():
            m = model.load()
            _more_tags = getattr(m, "_more_tags", None)
            if callable(_more_tags):
                model_tags.append(_more_tags())

        return tag_accumulate(model_tags)


class RegressorEnsemble(Ensemble[RegressorT], Regressor):
    pass


class ClassifierEnsemble(Ensemble[ClassifierT], Classifier):
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
        NotFittedError
            Raises if the ensemble has not been fit yet

        RuntimeError
            Raises if it recieved a jagged set of probabilities
        """
        check_is_fitted(self)
        x = self._validate_data(x, reset=False)

        predictions = self._predict_proba(x)
        if jagged(predictions):
            raise RuntimeError(
                "Probability predictions were jagged, perhaps not all classifiers"
                " were trained with the same labels or one of them produces output"
                " in a different format from the rest."
                f"\n\t{predictions}"
            )

        return predictions

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
        NotFittedError
            Raises if the ensemble has not been fit yet
        """
        ...
