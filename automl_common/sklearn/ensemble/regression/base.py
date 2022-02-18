from __future__ import annotations

from abc import abstractmethod
from typing import List, TypeVar

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from automl_common.sklearn.ensemble.base import Ensemble
from automl_common.sklearn.model import Regressor

RT = TypeVar("RT", bound=Regressor)


class RegressorEnsemble(Ensemble[RT], Regressor):
    """TODO"""

    def fit(self, x: np.ndarray, y: np.ndarray) -> RegressorEnsemble[RT]:
        """Fit a Regressor Ensemble

        Parameters
        ----------
        x : np.ndarray
            The data to fit to

        y : np.ndarray
            The targets to fit to

        Returns
        -------
        ClassifierEnsemble[ClassifierT]
            The ClassifierEnsemble
        """
        if self.model_store is None:
            raise RuntimeError("Can't fit without model store")

        x, y = check_X_y(x, y, accept_sparse=True, multi_output=True)

        # Reset attributes
        for attr in self._fit_attributes():
            if hasattr(self, attr):
                delattr(self, attr)

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
