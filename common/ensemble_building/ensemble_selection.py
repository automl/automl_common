from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np

from sklearn.pipeline import Pipeline

from .abstract_ensemble import AbstractEnsemble


DType = TypeVar("DType")


class EnsembleSelection(AbstractEnsemble):
    def __init__(
        self,
        ensemble_size: int,
        loss_fn: Callable[..., float],
        loss_fn_args: Dict[str, Any],
        random_state: np.random.RandomState,
        precision: Optional[DType] = None,
    ) -> None:
        self.ensemble_size = ensemble_size
        self.loss_fn: Optional[Callable[..., float]] = loss_fn
        self.loss_fn_args = loss_fn_args
        self.random_state = random_state
        self.precision = precision

    def __getstate__(self) -> Dict[str, Any]:
        # Cannot serialize a metric if
        # it is user defined.
        # That is, if doing pickle dump
        # the metric won't be the same as the
        # one in __main__. we don't use the metric
        # in the EnsembleSelection so this should
        # be fine
        self.loss_fn = None
        self.loss_fn_args = {}
        return self.__dict__

    def fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        identifiers: List[Tuple[int, int, float]],
    ) -> AbstractEnsemble:
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError("Ensemble size cannot be less than one!")

        self._fit(predictions, labels)
        self._calculate_weights()
        self.identifiers_ = identifiers
        return self

    def _fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
    ) -> AbstractEnsemble:
        """Fast version of Rich Caruana's ensemble selection method."""

        if self.loss_fn is None:
            raise ValueError(
                "A function with signature Callable[[labels, predictions, ...], float]"
                "is needed to perform ensemble selection."
            )

        self.num_input_models_ = len(predictions)

        ensemble: List[np.ndarray] = []
        trajectory: List[float] = []
        order: List[int] = []

        ensemble_size = self.ensemble_size

        weighted_ensemble_prediction = np.zeros(
            predictions[0].shape,
            dtype=self.precision,
        )
        fant_ensemble_prediction = np.zeros(
            weighted_ensemble_prediction.shape,
            dtype=self.precision,
        )
        # We will find the next model to add to the ensemble
        # based on a minimization problem
        # On every iteration, self.loss_fn computes the contribution of each
        # candidate model to be added to the ensemble
        losses = np.ones(
            (len(predictions)),
            dtype=self.precision,
        )
        for _ in range(ensemble_size):
            s = len(ensemble)
            if s > 0:
                np.add(
                    weighted_ensemble_prediction,
                    ensemble[-1],
                    out=weighted_ensemble_prediction,
                )

            # Memory-efficient averaging!
            for j, pred in enumerate(predictions):
                # fant_ensemble_prediction is the prediction of the current ensemble
                # and should be ([predictions[selected_prev_iterations] + predictions[j])/(s+1)
                # We overwrite the contents of fant_ensemble_prediction
                # directly with weighted_ensemble_prediction + new_prediction and then scale for avg
                np.add(weighted_ensemble_prediction, pred, out=fant_ensemble_prediction)
                np.multiply(
                    fant_ensemble_prediction, (1.0 / float(s + 1)), out=fant_ensemble_prediction
                )

                losses[j] = self.loss_fn(labels, fant_ensemble_prediction, **self.loss_fn_args)

            all_best = np.argwhere(losses == np.nanmin(losses)).flatten()
            best = self.random_state.choice(all_best)
            ensemble.append(predictions[best])
            trajectory.append(losses[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_loss_ = trajectory[-1]
        return self

    def _calculate_weights(self) -> None:
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros(
            (self.num_input_models_,),
            dtype=self.precision,
        )
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

    def predict(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:

        average = np.zeros_like(predictions[0], dtype=self.precision)
        tmp_predictions = np.empty_like(predictions[0], dtype=self.precision)

        # if predictions.shape[0] == len(self.weights_),
        # Then it means that the predictions of non_zero_weight models
        # is included.
        if len(predictions) == len(self.weights_):
            for pred, weight in zip(predictions, self.weights_):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif len(predictions) == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            for pred, weight in zip(predictions, non_null_weights):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError(
                "The dimensions of ensemble predictions" " and ensemble weights do not match!"
            )
        del tmp_predictions
        return average

    def __str__(self) -> str:
        return (
            "Ensemble Selection:\n\tTrajectory: %s\n\tMembers: %s"
            "\n\tWeights: %s\n\tIdentifiers: %s"
            % (
                " ".join(
                    [
                        "%d: %5f" % (idx, performance)
                        for idx, performance in enumerate(self.trajectory_)
                    ]
                ),
                self.indices_,
                self.weights_,
                " ".join(
                    [
                        str(identifier)
                        for idx, identifier in enumerate(self.identifiers_)
                        if self.weights_[idx] > 0
                    ]
                ),
            )
        )

    def get_models_with_weights(self, models: Pipeline) -> List[Tuple[float, Pipeline]]:
        output = []
        for i, weight in enumerate(self.weights_):
            if weight > 0.0:
                identifier = self.identifiers_[i]
                model = models[identifier]
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output

    def get_validation_performance(self) -> float:
        return self.trajectory_[-1]
