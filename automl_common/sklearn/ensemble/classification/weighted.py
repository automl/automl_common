from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from typing_extensions import Literal  # TODO, remove with Python 3.8

import numpy as np
from sklearn.metrics import accuracy_score

from automl_common.backend.stores.model_store import ModelStore
from automl_common.data.convert import probabilities_to_classes
from automl_common.data.math import majority_vote, weighted_sum
from automl_common.ensemble.builders.weighted_ensemble import weighted_ensemble_caruana
from automl_common.sklearn.ensemble.classification.base import ClassifierEnsemble
from automl_common.sklearn.ensemble.weighted import WeightedEnsemble
from automl_common.sklearn.model import Classifier
from automl_common.util.random import as_random_state
from automl_common.util.types import Orderable

CT = TypeVar("CT", bound=Classifier)
ID = TypeVar("ID")


class WeightedClassifierEnsemble(ClassifierEnsemble[ID, CT], WeightedEnsemble[ID, CT]):
    def __init__(
        self,
        *,
        model_store: ModelStore[ID, CT],
        tags: Optional[Dict[str, Any]] = None,
        classes: Optional[Union[List, np.ndarray]] = None,
        size: int = 10,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = accuracy_score,
        select: Literal["min", "max"] = "max",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        voting: Literal["majority", "probability"] = "probability",
    ):
        super().__init__(
            model_store=model_store,
            tags=tags,
            classes=classes,
        )
        self.size = size
        self.metric = metric
        self.select = select
        self.random_state = random_state
        self.voting = voting

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
        self.random_state_ = as_random_state(self.random_state)
        model_predictions = dict(self._model_probas(x, pred_key))
        print(model_predictions)

        weighted_ids, trajectory = weighted_ensemble_caruana(
            model_predictions=model_predictions,
            is_probabilities=True,
            classes=self.classes_,
            targets=y,
            size=self.size,
            metric=self.metric,
            select=self.select,
            random_state=self.random_state_,
        )
        self.trajectory_ = trajectory
        self.weights_ = weighted_ids

        ids = list(weighted_ids.keys())
        return ids

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
        """
        # Keep it iterable as weighted sum can handle it
        ids, weights = zip(*self.weights.items())

        if self.voting == "majority":
            preds = iter(self.model_store[id].load().predict(x) for id in ids)
            return majority_vote(preds, weights=np.asarray(weights))

        elif self.voting == "probability":
            probabilities = self._predict_proba(x)
            predictions = probabilities_to_classes(probabilities, self.classes_)

            if predictions.ndim == 2 and predictions.shape[1] == 1:
                predictions = predictions.flatten()

            return predictions

        else:
            raise NotImplementedError(self.voting)

    def _predict_proba(self, x: np.ndarray) -> np.ndarray:
        ids, weights = zip(*self.weights.items())
        probs = iter(self.model_store[id].load().predict_proba(x) for id in ids)  # pragma: no cover

        weighted_probs = weighted_sum(probs, weights=np.asarray(weights))

        return weighted_probs

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
            **super().get_params(deep=deep),
            "metric": self.metric,
            "select": self.select,
            "random_state": self.random_state,
            "voting": self.voting,
        }

    @classmethod
    def _fit_attributes(cls) -> List[str]:
        return super()._fit_attributes() + ["random_state_"]