from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Union

import numpy as np
from sklearn.utils.multiclass import class_distribution
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from automl_common.backend.stores.model_store import ModelStore
from automl_common.data.validate import jagged
from automl_common.sklearn.ensemble.base import Ensemble
from automl_common.sklearn.model import Classifier

CT = TypeVar("CT", bound=Classifier)


class ClassifierEnsemble(Ensemble[CT], Classifier):
    """An ensemble for Classifiers

    Parameters
    ----------
    model_store : ModelStore[CT]
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
        model_store: ModelStore[CT],
        classes: Optional[Union[np.ndarray, List]] = None,
    ):
        self.model_store = model_store
        self.classes = classes

    @classmethod
    def _fit_attributes(cls) -> List[str]:
        return super()._fit_attributes() + ["classes_", "n_classes_", "class_prior_"]

    def fit(self, x: np.ndarray, y: np.ndarray) -> ClassifierEnsemble[CT]:
        """Fit a classifier Ensemble

        This method ensures that `classes_` attribute is set

        Parameters
        ----------
        x : np.ndarray
            The data to fit to

        y : np.ndarray
            The targets to fit to

        Returns
        -------
        ClassifierEnsemble[CT]
            The ClassifierEnsemble

        Raises
        ------
        RuntimeError

        """
        if self.model_store is None:
            raise RuntimeError("Can't fit without model store")

        x, y = check_X_y(x, y, accept_sparse=True, multi_output=True)

        # Reset attributes
        for attr in self._fit_attributes():
            if hasattr(self, attr):
                delattr(self, attr)

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
                raise ValueError(f"`classes` mut be 1 or 2 dimensional, {self.classes}")

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
        x = check_array(x, accept_sparse=True)

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
        x = check_array(x, accept_sparse=True)

        probs = self._predict_proba(x)
        if jagged(probs):
            raise RuntimeError(
                "Probability predictions were jagged, perhaps not all classifiers"
                " were trained with the same labels or one of them produces output"
                " in a different format from the rest."
                f"\n\t{probs}"
            )

        return probs

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

    def _more_tags(self) -> Dict[str, Any]:
        return {**super()._more_tags(), "multioutput_only": False, "no_validation": True}
