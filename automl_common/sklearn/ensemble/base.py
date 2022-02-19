"""
Following:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Optional, TypeVar

import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble.ensemble import Ensemble as BaseEnsemble
from automl_common.sklearn.ensemble.util import tag_accumulate
from automl_common.sklearn.model import Predictor

PredictorT = TypeVar("PredictorT", bound=Predictor)

# https://www.python.org/dev/peps/pep-0673/#motivation
SelfT = TypeVar("SelfT", bound="Ensemble")  # TODO Python 3.11 or typing_extensions


class Ensemble(BaseEnsemble[PredictorT], Predictor, BaseEstimator):
    """An sklearn style ensemble which includes `fit`.

    A base class for sklearn style ensembles.

    Any easy convention for `_fit_attributes` is

    .. code-block:: python

        @classmethod
        def _fit_attributes(cls) -> List[str]:
            return super()._fit_attributes() + ["myattribute_", "cat_"]

    On top of this, implementing classes should also implement `get_params` for full
    sklearn compatibility.

    Parameters
    ----------
    model_store: ModelStore[PredictorT]
        A model store from which models can be chosen.
        Will not operate properly without it.

    tags: Optional[Dict[str, Any]] = None
        What tags the ensemble advertises to sklearn
    """

    _required_parameters: List[str] = ["model_store"]

    @abstractmethod
    def __init__(
        self,
        *,
        model_store: ModelStore[PredictorT],
        tags: Optional[Dict[str, Any]] = None,
    ):
        self.model_store = model_store
        self.tags = tags

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
        return ["ids_", "n_outputs_"]

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

    @abstractmethod
    def fit(self: SelfT, x: np.ndarray, y: np.ndarray) -> SelfT:
        """Fit the ensemble to the given targets

        Implemented should set the attribute `self.ids_`

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
        ...

    @abstractmethod
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
        ...

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

        if model_id not in self.ids:  # type: ignore
            raise KeyError(f"Model {model_id} not in ensemble")

        return self.model_store[model_id].load()

    def _more_tags(self) -> Dict[str, Any]:
        """
        Note
        ----
        We assume this method is only called for testing purposes by sklearn.
        This is quite an expensive function so we raise a warning if this is
        not the case.
        """
        if self.tags is not None:
            return self.tags

        warnings.warn("Expensive function `_more_tags` called.")

        # Get all model tags
        model_tags: List[Dict[str, bool]] = []
        tag_f_attributes = ["_more_tags", "_get_tags"]

        for model in self.model_store.values():
            m = model.load()
            tag_fs = [getattr(m, attr, None) for attr in tag_f_attributes]
            for tag_f in [f for f in tag_fs if callable(f)]:
                model_tags.append(tag_f())

        return tag_accumulate(model_tags)
