from typing import Any, Dict, List, Tuple, Mapping, Optional, Union
from collections import Counter

import numpy as np

from automl_common.metric import MetricProtocol
from automl_common.util import as_random_state


def weighted_ensemble(
    model_predictions: Mapping[str, np.ndarray],
    targets: np.ndarray,
    size: int,
    metric: MetricProtocol,
    method: str = "caruana",
    metric_args: Optional[Mapping[str, Any]] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None
) -> Tuple[Dict[str, float], List[float]]:
    """Calculate a weighted ensemble of `n` models

    Parameters
    ----------
    model_predictions: Mapping[str, np.ndarray]
        The model predictions to use, mapping from id to their predictions

    targets: np.ndarray
        The targets

    size: int
        The size of the ensemble to create

    metric: MetricProtocol
        The metric to use in calculating which models to add to the ensemble

    method: str = "caruana"
        The method to generate the weighted ensemble with

        * ``"caruana"`` - Use a fast version of Rich Caruna's method

    metric_args: Optional[Mapping[str, Any]] = None
        Arguments to forward to the metric

    random_state: Optional[Union[int, np.random.RandomState]] = None
        The random_state to use

    Returns
    -------
    (Dict[str, float], List[float])
        A dictionary mapping from id's to the weight as well as the trajectory
        during training.
    """
    if not size > 0:
        raise ValueError(f"Ensemble size ({size}) must be greater than 0")

    funcs = {"caruana": weighted_ensemble_caruana}
    func = funcs.get(method)

    if func is None:
        raise NotImplementedError(f"Not a supported method {method}")
    else:
        return func(
            model_predictions=model_predictions,
            targets=targets,
            metric=metric,
            metric_args=metric_args,
            random_state=random_state
        )


def weighted_ensemble_caruana(
    model_predictions: Mapping[str, np.ndarray],
    targets: np.ndarray,
    size: int,
    metric: Metric,
    metric_args: Optional[Mapping[str, Any]] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None
) -> Tuple[Dict[str, float], List[float]]:
    """Calculate a weighted ensemble of `n` models

    Parameters
    ----------
    model_predictions: Mapping[str, np.ndarray]
        The model predictions to use, mapping from id to their predictions

    targets: np.ndarray
        The targets

    size: int
        The size of the ensemble to create

    metric: Metric
        The metric to use in calculating which models to add to the ensemble

    method: str
        The method to generate the weighted ensemble with

        * ``"fast"`` - Use a fast version of Rich Caruna's method
        * ``"slow"`` - Use a slow version of Rich Caruna's method

    metric_args: Optional[Mapping[str, Any]] = None
        Arguments to forward to the metric

    random_state: Optional[Union[int, np.random.RandomState]] = None
        The random_state to use

    Returns
    -------
    (Dict[str, float], List[float])
        A dictionary mapping from id's to the weight as well as the trajectory
        during training.
    """
    rand = as_random_state(random_state)

    ids = list(model_predictions.keys())
    predictions = list(model_predictions.values())

    # We store each added best into current
    current = np.zeros(predictions[0].shape, dtype=np.float64)

    # Buffer is used to store candidate models weighted sum
    buffer = np.empty(predictions[0].shape, dtype=np.float64)

    ensemble: List[str] = []
    trajectory: List[float] = []

    def loss_if_added(_id: str) -> float:
        # Get the loss if the model was added to the current set of predicitons
        np.add(current, model_predictions[_id], out=buffer)
        np.divide(buffer, len(ensemble) + 1, out=buffer)
        return metric.loss(buffer, targets, **metric_args)

    for i in range(size):
        # Get the loss for each model
        losses = {id: loss_if_added(id) for id in ids}

        # Choose one of the best models
        minimum = min(losses.values())
        best = [id for id, loss in losses.items() if loss == minimum]
        chosen = rand.choice(best)

        # Add the predictions of the chosen model
        np.add(current, model_predictions[chosen], out=current)

        # Record it's addition and the over all trajectory of loss
        ensemble.append(chosen)
        trajectory.append(losses[chosen])

        # In the case of only one model, have calculated it's loss
        # and it's the only available model to add to the ensemble
        if len(model_predictions) == 1:
            trajectory *= size
            ensemble *= size
            break

    # Calculate weights
    weighted_ensemble = {id: count / size for id, count in Counter(ensemble)}

    return weighted_ensemble, trajectory
