from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Union, Tuple, Iterator

import numpy as np
from sklearn.utils.multiclass import check_classification_targets, class_distribution
from sklearn.utils.validation import check_is_fitted

from automl_common.backend.stores.model_store import ModelStore
from automl_common.sklearn.ensemble.base import Ensemble
from automl_common.sklearn.model import Classifier

CT = TypeVar("CT", bound=Classifier)
ID = TypeVar("ID")


class ClassifierEnsemble(Ensemble[ID, CT], Classifier):
    """An ensemble for Classifiers

    Parameters
    ----------
    model_store : ModelStore[ID, CT]
        Where to fit models from

    classes: Optional[List[Any]] = None
        The classes to use for this Ensemble.

        If None is provided, a LabelEncoder will be used on 1d targets and
        ``np.unique`` on 2d numerical data.

        Note
        ----
        This will not transform data before handing to classifiers, only to convert
        probabilities to classes.
    """

    def __init__(
        self,
        *,
        model_store: ModelStore[ID, CT],
        tags: Optional[Dict[str, Any]] = None,
        classes: Optional[Union[np.ndarray, List]] = None,
    ):
        super().__init__(model_store=model_store, tags=tags)
        self.classes = classes

    @classmethod
    def _fit_attributes(cls) -> List[str]:
        return super()._fit_attributes() + [
            "classes_",
            "n_classes_",
            "class_prior_",
            "n_features_in_",
        ]

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pred_key: Optional[str] = None,
    ) -> ClassifierEnsemble[ID, CT]:
        """Fit a classifier Ensemble

        This method ensures that `classes_` attribute is set

        Parameters
        ----------
        x : np.ndarray
            The data to fit to

        y : np.ndarray
            The targets to fit to

        pred_key: Optional[str] = None
            The name of predictions to try and load instead of loading the full models

        Returns
        -------
        ClassifierEnsemble[CT]
            The ClassifierEnsemble

        Raises
        ------
        RuntimeError

        """
        # Reset attributes
        for attr in self._fit_attributes():
            if hasattr(self, attr):
                delattr(self, attr)

        # Validate the data, set's the `n_features_in` attribute
        x, y = self._validate_data(x, y, accept_sparse=True, multi_output=True)
        check_classification_targets(y)

        # Fill in attributes, classes_, n_classes_ and class_prior_
        if self.classes is not None:
            self.classes_ = self.classes
            self.class_prior_ = None
            self.n_classes_: Union[int, List[int]]

            ndim = np.ndim(self.classes_)
            if ndim == 1:
                self.n_classes_ = len(self.classes_)
            elif ndim == 2:
                self.n_classes_ = [len(col_classes) for col_classes in self.classes]
            else:
                raise ValueError(f"`classes` must be 1 or 2 dimensional, {self.classes}")

        else:
            shape = np.shape(y)
            if len(shape) == 1 or shape[1] == 1:
                self.classes_, self.n_classes_, self.class_prior_ = class_distribution(
                    y=y.reshape((-1, 1))
                )
                self.classes_ = self.classes_[0]  # Returns [np.ndarray(..classes...)]
            else:
                self.classes_, self.n_classes_, self.class_prior_ = class_distribution(y)

        # Get the output shape
        shape = np.shape(y)
        if len(shape) == 1:
            self.n_outputs_ = 1
        else:
            self.n_outputs_ = shape[1]

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
        x = self._validate_data(X=x, accept_sparse=True, reset=False)

        return self._predict(x)

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
        x = self._validate_data(X=x, accept_sparse=True, reset=False)
        return self._predict_proba(x)

    @abstractmethod
    def _fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pred_key: Optional[str] = None,
    ) -> List[ID]:
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

        Parameters
        ----------
        x : np.ndarray
            The data to predict on

        Returns
        -------
        np.ndarray
            The predictions for x

        """
        ...

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
        return {**super().get_params(deep=deep), "classes": self.classes}

    def _model_probas(
        self,
        x: np.ndarray,
        pred_key: Optional[str] = None,
    ) -> Iterator[Tuple[ID, np.ndarray]]:
        """Helper to get the model proba predicitons, loading from pred_key if it can.

        Parameters
        ----------
        x : np.ndarray,
            Fit the ensemble to the given x data

        pred_key: Optional[str] = None
            The name of predictions to try and load instead of loading the full models

        Returns
        -------
        Dict[ID, np.ndarray]
            Mapping from model id to their predictions
        """
        ids = list(self.model_store.keys())
        model_accessors = iter(self.model_store[id] for id in ids)
        predictions = iter(
            m.predictions[pred_key]
            if pred_key is not None and pred_key in m.predictions
            else np.asarray(m.load().predict_proba(x))
            for m in model_accessors
        )
        return zip(ids, predictions)
