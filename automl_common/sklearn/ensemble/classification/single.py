from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from typing_extensions import Literal

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble.builders.single_best import single_best
from automl_common.sklearn.ensemble.classification.base import ClassifierEnsemble
from automl_common.sklearn.ensemble.single import SingleEnsemble
from automl_common.sklearn.model import Classifier
from automl_common.util.types import Orderable

CT = TypeVar("CT", bound=Classifier)
ID = TypeVar("ID")


class SingleClassifierEnsemble(SingleEnsemble[ID, CT], ClassifierEnsemble[ID, CT]):
    """TODO"""

    def __init__(
        self,
        *,
        model_store: ModelStore[ID, CT],
        tags: Optional[Dict[str, Any]] = None,
        classes: Optional[Union[np.ndarray, List]] = None,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = accuracy_score,
        select: Literal["min", "max"] = "max",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        super().__init__(
            model_store=model_store,
            classes=classes,
            tags=tags,
        )
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
        self.random_state_ = check_random_state(self.random_state)
        model_predictions = self._model_predictions(x, pred_key)

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
        NotFittedError
            If the ensemble has not been fit yet
        """
        return np.asarray(self.model.predict(x))

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
        return np.asarray(self.model.predict_proba(x))

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
    def _fit_attributes(self) -> List[str]:
        return super()._fit_attributes() + ["random_state_"]
